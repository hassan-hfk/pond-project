from ultralytics import YOLO
import cv2
import numpy as np

class Inference:
    def __init__(self, cfg):
        self.model = YOLO(cfg['model']['yolov8_path'])
        self.conf = float(cfg['model'].get('conf_thresh', 0.45))

    def predict(self, frame, cfg):
        # frame: BGR image
        self.conf = float(cfg['model'].get('conf_thresh', 0.45))
        results = self.model.predict(source=frame, imgsz=640, conf=self.conf, verbose=False)
        res = results[0]
        boxes = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes.xyxy, 'cpu') else np.array(res.boxes.xyxy)
        classes = res.boxes.cls.cpu().numpy().astype(int) if hasattr(res.boxes.cls, 'cpu') else np.array(res.boxes.cls).astype(int)
        confs = res.boxes.conf.cpu().numpy() if hasattr(res.boxes.conf, 'cpu') else np.array(res.boxes.conf)
        print(confs)
        detections = []
        for i, b in enumerate(boxes):
            x1,y1,x2,y2 = b.astype(int)
            detections.append({
                'class_id': int(classes[i]),
                'class_name': self.model.names[int(classes[i])],
                'conf': float(confs[i]),
                'bbox_px': [int(x1),int(y1),int(x2),int(y2)],
            })
        return detections