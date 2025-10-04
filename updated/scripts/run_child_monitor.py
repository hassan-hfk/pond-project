import sys
from pathlib import Path
from datetime import datetime
import cv2
import yaml
import time
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from inference import Inference
from child_logic import ChildMonitor
from db_handler import insert_detection
from video_recorder import VideoRecorder

# Define paths relative to project root
CONFIG_PATH = project_root / 'config' / 'config.yaml'
RECORDINGS_PATH = project_root / 'data' / 'recordings'
LOGS_PATH = project_root / 'logs'

detection_active = False

def set_detection_status(status):
    global detection_active
    detection_active = status

def generate_frame():
    """Generate video frames with detection overlay"""
    print("generate_frame function called")
    
    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)

    cam_w = int(cfg['camera']['width'])
    cam_h = int(cfg['camera']['height'])
    fps = int(cfg['camera'].get('fps', 25))
    cam_dev = int(cfg['camera'].get('device', 0))

    inf = Inference(cfg)
    mon = ChildMonitor(str(CONFIG_PATH))

    # Ensure directories exist
    RECORDINGS_PATH.mkdir(parents=True, exist_ok=True)
    LOGS_PATH.mkdir(parents=True, exist_ok=True)

    # Video recorder setup
    recorder = VideoRecorder(
        output_dir=str(RECORDINGS_PATH),
        fps=fps,
        pre_secs=4,
        post_secs=4,
        frame_size=(cam_w, cam_h)
    )
    
    last_event_time = {}
    cooldown = 15
    picam2 = cv2.VideoCapture(cam_dev)
    
    log_file = LOGS_PATH / cfg['logging'].get('events_file', 'events.log')
    logf = open(log_file, 'a')

    # MiDaS depth optional
    use_midas = False
    midas_model = None
    midas_transform = None
    depth_scale = cfg.get('calibration', {}).get('depth_scale', None)
    
    if cfg.get('model', {}).get('use_midas', False):
        use_midas = True
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        midas.to(device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        midas_transform = transforms.small_transform if "MiDaS_small" in "MiDaS_small" else transforms.default_transform
        midas_model = midas

    try:
        frame_count = 0
        while True:
            start_time = time.time()
            frame_count += 1
            print(f"Frame {frame_count} - detection_active: {detection_active}")
            
            ret, frame = picam2.read()
            if not ret:
                continue
                
            frame = cv2.resize(frame, (cam_w, cam_h))
            
            # ALWAYS update recorder buffer
            recorder.update(frame)

            # Only run detection if active
            if detection_active:
                # Re-load config to get latest settings
                with open(CONFIG_PATH, 'r') as f:
                    cfg = yaml.safe_load(f)
                
                print("Detection ACTIVE - Running detection logic")
                
                # Manual test trigger
                if cv2.waitKey(1) & 0xFF == ord('r'):
                    print("[MANUAL] Triggered recording")
                    recorder.trigger("manual_test")

                # Run YOLO inference
                dets = inf.predict(frame, cfg)

                # Build depth map if MiDaS is enabled
                depth_map = None
                if use_midas and midas_model is not None:
                    pass  # MiDaS code here if needed

                # Process detections
                events, debug = mon.process_detections(dets, depth_map)

                # Handle events
                now = time.time()
                for e in events:
                    cls = e.get("class") or e.get("class_name", "unknown")
                    ts = time.ctime(e["timestamp"])
                    logf.write(f"{ts} - EVENT - {e}\n")
                    logf.flush()
                    print(f"[EVENT] {ts} - {cls} - {e}")

                    last_ts = last_event_time.get(cls, 0)
                    if now - last_ts >= cooldown:
                        video_filename = recorder.trigger(cls)
                        if video_filename:
                            last_event_time[cls] = now
                            insert_detection(video_filename, e, output_dir=str(RECORDINGS_PATH))
                            print(f"[DB] Inserted detection for video: {video_filename}")
                    else:
                        remaining = int(cooldown - (now - last_ts))
                        print(f"[COOLDOWN] {cls} cooling down ({remaining}s left)")

                # Build overlay with detection results
                overlay = frame.copy()
                for poly in mon.rois_px:
                    cv2.polylines(overlay, [poly], isClosed=True, color=(0, 255, 0), thickness=2)

                for d in debug:
                    x1, y1, x2, y2 = d["bbox"]
                    cls = d["class"]
                    conf = d["conf"]
                    color = (0, 255, 0) if cls in mon.classes else (160, 160, 160)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                    label = f"{cls} {conf:.2f}"
                    cv2.putText(overlay, label, (x1, max(12, y1 - 6)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                    h = d.get("height_m", None)
                    if h is not None:
                        cv2.putText(overlay, f"H:{h:.2f}m", (x1, y1 - 22), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    if d.get("in_roi", False):
                        cx = int((x1 + x2) // 2)
                        cy = int(y2)
                        cv2.circle(overlay, (cx, cy), 5, (0, 0, 255), -1)
                        cnt = d.get("count", 0)
                        cv2.putText(overlay, f"cnt:{cnt}", (x2 - 60, y2 + 15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                fps_display = 1.0 / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
                cv2.putText(overlay, f"FPS: {fps_display:.2f}", (cam_w - 120, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            else:
                # When detection is inactive, just show plain video
                print("Detection INACTIVE - Showing plain video")
                overlay = frame.copy()
                cv2.putText(overlay, "DETECTION INACTIVE - Press Start Detection", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # ALWAYS show the video feed and yield frames
            cv2.imshow("monitor", overlay)
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
            ret, overlay_encoded = cv2.imencode(".jpg", overlay)
            overlay_bytes = overlay_encoded.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + overlay_bytes + b'\r\n')
                   
    except Exception as e:
        print(f"Error in generate_frame: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logf.close()
        cv2.destroyAllWindows()
        picam2.release()