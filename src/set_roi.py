import cv2
import yaml
import numpy as np
from pathlib import Path

# Define config path
project_root = Path(__file__).parent.parent
CONFIG_PATH = project_root / "config" / "config.yaml"

points = []

def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if points:
            points.pop()

def main():
    global points
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    cam_index = cfg["camera"]["device"]
    width, height = cfg["camera"]["width"], cfg["camera"]["height"]

    # picam2 = Picamera2()
    # camera_config = picam2.create_preview_configuration(main={"size": (2460, 2460)})
    # camera_config["transform"] = Transform(vflip=1)
    # picam2.configure(camera_config)
    # picam2.start()

    cv2.namedWindow("ROI Selector")
    cv2.setMouseCallback("ROI Selector", click_event)

    print("[INFO] Left click = add point, Right click = undo, ENTER = save, R = reset, ESC = quit")
    picam2 = cv2.VideoCapture(0)
    while True:
        ret, frame = picam2.read()
        if ret:
            frame = cv2.resize(frame, (width, height))
            # Draw polygon preview
            temp = frame.copy()
            if points:
                for i, p in enumerate(points):
                    cv2.circle(temp, p, 4, (0, 255, 0), -1)
                    if i > 0:
                        cv2.line(temp, points[i-1], p, (255, 0, 0), 2)
                if len(points) > 2:
                    cv2.line(temp, points[-1], points[0], (255, 0, 0), 2)

            cv2.imshow("ROI Selector", temp)
            key = cv2.waitKey(1) & 0xFF

            if key == 13:  # ENTER
                if len(points) >= 3:
                    # Normalize coords
                    norm_poly = [[x / width, y / height] for (x, y) in points]
                    cfg["rois"] = [norm_poly]

                    with open(CONFIG_PATH, "w") as f:
                        yaml.safe_dump(cfg, f)

                    print("✅ ROI saved to config.yaml")
                    break
                else:
                    print("⚠️ Need at least 3 points for polygon.")

            elif key in [ord("r"), ord("R")]:  # Reset
                points = []
                print("[INFO] Polygon reset.")

            elif key == 27:  # ESC
                print("[INFO] Exiting without saving.")
                break
                
    cv2.destroyAllWindows()

    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(cfg, f)

if __name__ == "__main__":
    main()