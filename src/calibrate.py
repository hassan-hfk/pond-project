import cv2
import yaml
import numpy as np
import torch
from pathlib import Path

# Define config path relative to project
project_root = Path(__file__).parent.parent
CONFIG_PATH = project_root / "config" / "config.yaml"

# Load config
if CONFIG_PATH.exists():
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
else:
    raise SystemExit("config.yaml not found. Create it from the template.")

W = cfg['camera']['width']
H = cfg['camera']['height']

# --- Setup PiCamera2 ---
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(main={"size": (2460, 2460)})
camera_config["transform"] = Transform(vflip=1)
picam2.configure(camera_config)
picam2.start()

print("Calibration: draw a reference box (top-left -> bottom-right) around an object of known height.")
print("Then you'll be asked to enter the object's real height (m) and distance to camera (m).")
print("After that press 'v' to click two vertical lines (4 clicks total) to compute vertical VP.")

ref_box = None
start = None
curr = None

def mouse_cb(ev, x, y, fl, p):
    global start, ref_box, curr
    if ev == cv2.EVENT_LBUTTONDOWN:
        start = (x, y)
    elif ev == cv2.EVENT_MOUSEMOVE and start is not None:
        curr = (x, y)
    elif ev == cv2.EVENT_LBUTTONUP:
        ref_box = (start[0], start[1], x, y)
        start = None

cv2.namedWindow('cal')
cv2.setMouseCallback('cal', mouse_cb)

# --- Frame loop for ref_box selection ---
while True:
    frame = picam2.capture_array()
    frame = cv2.resize(frame, (W, H))
    vis = frame.copy()
    if start is not None and curr is not None:
        cv2.rectangle(vis, start, curr, (0, 255, 0), 2)
    if ref_box is not None:
        x1, y1, x2, y2 = ref_box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('cal', vis)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # ESC
        break
    if k == 32 and ref_box is not None:  # SPACE
        break

if ref_box is None:
    print('No ref box set. Exiting.')
    picam2.stop()
    cv2.destroyAllWindows()
    raise SystemExit

x1, y1, x2, y2 = ref_box
# normalize
ref_box_norm = [x1 / W, y1 / H, x2 / W, y2 / H]
ref_px_h = abs(y2 - y1)

# ask for real-world numbers
ref_h = float(input('Reference real height (meters): '))
ref_z = float(input('Reference distance from camera (meters): '))

# compute focal length px
f_px = (ref_px_h * ref_z) / ref_h
print(f'Computed focal px: {f_px:.2f}')

# vanishing point collection
print('Now press v and click 4 points: two vertical lines (2 pts each). Press ENTER when done.')
vp_points = []
vp_mode = False

def vp_mouse(ev, x, y, fl, p):
    global vp_points, vp_mode
    if not vp_mode:
        return
    if ev == cv2.EVENT_LBUTTONDOWN:
        vp_points.append((x, y))
        print('vp click', len(vp_points), (x, y))

cv2.setMouseCallback('cal', vp_mouse)

# --- VP loop ---
while True:
    frame = picam2.capture_array()
    frame = cv2.resize(frame, (W, H))
    vis = frame.copy()
    if ref_box is not None:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for p in vp_points:
        cv2.circle(vis, p, 4, (0, 0, 255), -1)
    cv2.imshow('cal', vis)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('v'):
        vp_mode = True
        print('VP mode ON: click 4 points')
    if k == 13 or k == 10:  # ENTER
        break

vp_mode = False
if len(vp_points) < 4:
    print('Not enough VP points. Skipping VP computation.')
    vertical_vp = None
else:
    def line(a, b):
        return np.cross(np.array([a[0], a[1], 1.0]), np.array([b[0], b[1], 1.0]))
    l1 = line(vp_points[0], vp_points[1])
    l2 = line(vp_points[2], vp_points[3])
    vp_h = np.cross(l1, l2)
    if abs(vp_h[2]) < 1e-8:
        vertical_vp = None
    else:
        vertical_vp = [float(vp_h[0] / vp_h[2]), float(vp_h[1] / vp_h[2])]

# save to config
cfg['calibration']['ref_box_norm'] = ref_box_norm
cfg['calibration']['ref_height_m'] = float(ref_h)
cfg['calibration']['ref_distance_m'] = float(ref_z)
cfg['calibration']['focal_px'] = float(f_px)
cfg['calibration']['vertical_vp'] = vertical_vp
cfg['calibration']['img_size'] = [W, H]

CONFIG_PATH.write_text(yaml.dump(cfg))
print('Calibration saved to config.yaml')

picam2.stop()
cv2.destroyAllWindows()
cv2.destroyAllWindows()
