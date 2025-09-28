# child_logic.py
import cv2
import yaml
import numpy as np
from collections import defaultdict
import time
import math
from shapely.geometry import Polygon
import paho.mqtt.client as mqtt


class ChildMonitor:
    def __init__(self, config_path="config.yaml"):
        # load config
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        # camera size from config
        cam = self.cfg.get("camera", {})
        self.img_w = int(cam.get("width", 640))
        self.img_h = int(cam.get("height", 640))

        # thresholds & classes
        thr = self.cfg.get("thresholds", {})
        self.height_thresh = float(thr.get("height_m", 2.2))
        self.frames_needed = int(thr.get("consecutive_frames", 3))
        self.classes = set(thr.get("classes_trigger", ["person"]))

        # calibration
        self.cal = self.cfg.get("calibration", {})
        # expected keys: ref_box_norm [x1,y1,x2,y2], ref_height_m, focal_px, depth_scale, vertical_vp
        # convert null -> None
        if self.cal.get("vertical_vp") is None:
            self.cal["vertical_vp"] = None

        # rois converted to pixel polygons (list of arrays)
        self.rois_px = self._load_rois(self.cfg.get("rois", []))

        # debounce counters keyed by coarse quantized bbox center
        self.counts = defaultdict(int)
        # last triggered timestamp for cooldown
        self.last_trigger_ts = {}
        self.trigger_cooldown = float(self.cfg.get("gpio", {}).get("cooldown_s", 5.0))

    def _load_rois(self, rois_norm):
        rois_px = []
        for poly in rois_norm:
            pts = []
            for (x, y) in poly:
                px = int(round(x * self.img_w))
                py = int(round(y * self.img_h))
                pts.append([px, py])
            if len(pts) >= 3:
                rois_px.append(np.array(pts, dtype=np.int32))
        return rois_px

    # ------------ height estimation helpers ------------
    def height_from_vp(self, bbox):
        """Single-view metrology estimate using vertical vanishing point + ref box + ref height."""
        cal = self.cal
        if not cal or cal.get("ref_box_norm") is None or cal.get("vertical_vp") is None or cal.get("ref_height_m") is None:
            return float("nan")

        ref = cal["ref_box_norm"]
        x1r = int(ref[0] * self.img_w)
        y1r = int(ref[1] * self.img_h)
        x2r = int(ref[2] * self.img_w)
        y2r = int(ref[3] * self.img_h)
        y_tr, y_br = y1r, y2r

        # Person top/bottom
        y_tp, y_bp = bbox[1], bbox[3]

        vp = cal["vertical_vp"]
        # vp may be [x,y] even if outside image
        yv = float(vp[1])

        denom_ref = (y_tr - yv)
        denom_p = (y_tp - yv)
        if abs(denom_ref) < 1e-8 or abs(denom_p) < 1e-8:
            return float("nan")

        scale_ref = (y_br - yv) / denom_ref
        scale_p = (y_bp - yv) / denom_p
        if abs(scale_p) < 1e-12:
            return float("nan")

        H_ref = float(cal["ref_height_m"])
        H_p = H_ref * (scale_ref / scale_p)
        return float(H_p)

    def height_from_depth(self, bbox, depth_map):
        """Depth-based metric estimate: H = (h_px * Z_person) / f_px.
           depth_map must be in meters or be scaled previously via calibration.depth_scale."""
        if depth_map is None:
            return float("nan")
        cal = self.cal
        if cal.get("focal_px") is None:
            return float("nan")

        x1, y1, x2, y2 = bbox
        h_px = max(1, y2 - y1)
        cx = int((x1 + x2) / 2)
        cy = int(y2)  # use foot or bottom for distance
        H = float("nan")
        # safe index
        if 0 <= cy < depth_map.shape[0] and 0 <= cx < depth_map.shape[1]:
            Z_person = float(np.median(depth_map[max(0, cy-2):min(depth_map.shape[0], cy+3),
                                                max(0, cx-2):min(depth_map.shape[1], cx+3)]))
            if Z_person > 0 and not math.isnan(Z_person):
                f_px = float(cal["focal_px"])
                H = (h_px * Z_person) / f_px
        return H

    # ------------ PV helpers ------------
    def _is_point_in_any_roi(self, pt):
        x, y = pt
        for poly in self.rois_px:
            # poly is np.array Nx2 dtype=int32
            if cv2.pointPolygonTest(poly, (int(x), int(y)), False) >= 0:
                return True
        return False

    # ------------ main processing ------------


    def process_detections(self, detections, depth_map=None, iou_thresh=0.02, mqtt_client=None):
        """
        detections: list of dicts from Inference.predict
        depth_map: numpy array scaled to meters if available
        iou_thresh: overlap ratio threshold
        mqtt_client: instance of MQTTSafeClient (optional)
        Returns:
            events: list of event dicts
            debug: list of per-detection debug dicts
        """
        from shapely.geometry import Polygon
        import time
        import math

        events = []
        debug = []

        for det in detections:
            cname = det.get("class_name", "")
            bbox = det.get("bbox_px", [0,0,0,0])
            conf = float(det.get("conf", 0.0))

            d = {"class": cname, "bbox": bbox, "conf": conf}

            x1, y1, x2, y2 = bbox
            box_poly = Polygon([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])

            # ROI check only for person (ROI is irrelevant for other classes)
            in_roi = False
            if cname == "person":
                for poly in self.rois_px:
                    roi_poly = Polygon(poly)
                    if not roi_poly.is_valid or not box_poly.is_valid:
                        continue
                    inter_area = roi_poly.intersection(box_poly).area
                    box_area = box_poly.area
                    if box_area > 0 and (inter_area / box_area) >= iou_thresh:
                        in_roi = True
                        break
            d["in_roi"] = in_roi

            # --- Height estimation (used only to decide child_in_water) ---
            H_svm = self.height_from_vp(bbox)
            H_depth = self.height_from_depth(bbox, depth_map) if depth_map is not None else float("nan")
            if not math.isnan(H_svm):
                H_est = H_svm
                d["height_method"] = "svm"
            elif not math.isnan(H_depth):
                H_est = H_depth
                d["height_method"] = "depth"
            else:
                H_est = float("nan")
                d["height_method"] = "none"

            d["height_m"] = (None if math.isnan(H_est) else H_est)

            # debounce logic (IMPORTANT: person increments counter wherever detected)
            key = (int((x1+x2)/2) // 40, int(y2) // 40)

            if cname == "person":
                # **Record (count) person anywhere in the frame** — ignore ROI/height for counting.
                # This ensures recordings are created for persons even if they are outside ROI or tall.
                self.counts[key] += 1
            else:
                # other classes: no ROI / no height check
                self.counts[key] += 1

            d["count"] = int(self.counts[key])

            # Determine child_in_water (height + ROI still matters for this event type)
            child_in_water = (
                cname == "person"
                and in_roi
                and (not math.isnan(H_est))
                and (H_est < self.height_thresh)
            )
            # person_in_roi = any person in ROI but not child_in_water (tall person)
            person_in_roi = (cname == "person" and in_roi and not child_in_water)

            # --- Trigger events with per-class cooldown ---
            if self.counts[key] >= self.frames_needed:
                now = time.time()
                last_ts = self.last_trigger_ts.get(cname, 0)
                if now - last_ts >= self.trigger_cooldown:
                    # Choose event type
                    if child_in_water:
                        ev_type = "child_in_water"
                    elif person_in_roi:
                        ev_type = "person_in_roi"
                    else:
                        ev_type = "other_detected"

                    ev = {
                        "type": ev_type,
                        "class": cname,
                        "bbox": bbox,
                        "height_m": d["height_m"],
                        "in_roi": in_roi,
                        "key": key,
                        "timestamp": now
                    }
                    events.append(ev)
                    # update per-class cooldown timestamp
                    self.last_trigger_ts[cname] = now

                    # Publish MQTT (respects per-class cooldown)
                    if mqtt_client:
                        try:
                            mqtt_client.publish("pond/detections", ev)
                        except Exception:
                            # fallback: publish string payload if client requires it
                            mqtt_client.publish("pond/detections", str(ev))
                        print(f"[MQTT] Published {ev_type} event for class '{cname}': {ev}")

                # reset counter for this spatial key after firing (avoids immediate re-trigger)
                self.counts[key] = 0

            debug.append(d)

        return events, debug
