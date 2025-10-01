from flask import Flask, render_template, Response, jsonify, request
from run_child_monitor import set_detection_status, generate_frame, detection_active
import yaml
import os
import cv2
import numpy as np

app = Flask(__name__)

print(f"🚀 App starting - initial detection_active: {detection_active}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/roi_editor')
def roi_editor():
    return render_template('roi_editor.html')

@app.route('/video_feed')
def video_feed():
    print("🎥 Video feed requested")
    return Response(generate_frame(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/roi_video_feed')
def roi_video_feed():
    """Video stream specifically for ROI editing (no detection overlay)"""
    print("🎥 ROI Video feed requested")
    return Response(generate_roi_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection')
def start_detection():
    print("🟢 Start detection endpoint called")
    set_detection_status(True)
    return jsonify({'status': 'success', 'message': 'Detection started'})

@app.route('/stop_detection')
def stop_detection():
    print("🔴 Stop detection endpoint called")
    set_detection_status(False)
    return jsonify({'status': 'success', 'message': 'Detection stopped'})

@app.route('/get_confidence')
def get_confidence():
    """Get current confidence threshold from config.yaml"""
    try:
        with open('config.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
        conf_thresh = cfg['model'].get('conf_thresh', 0.45)
        return jsonify({'confidence': conf_thresh})
    except Exception as e:
        print(f"Error reading confidence: {e}")
        return jsonify({'confidence': 0.45})

@app.route('/update_confidence/<float:confidence>')
def update_confidence(confidence):
    """Update confidence threshold in config.yaml"""
    try:
        # Read current config
        with open('config.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
        
        # Update confidence threshold
        if 'model' not in cfg:
            cfg['model'] = {}
        cfg['model']['conf_thresh'] = float(confidence)
        
        # Write back to file
        with open('config.yaml', 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)
        
        print(f"✅ Confidence threshold updated to: {confidence}")
        return jsonify({'status': 'success', 'message': f'Confidence updated to {confidence}'})
    except Exception as e:
        print(f"Error updating confidence: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/save_rois', methods=['POST'])
def save_rois():
    """Save ROI polygons to config.yaml - matching set_roi.py logic exactly"""
    try:
        data = request.get_json()
        points = data.get('points', [])
        
        if len(points) < 3:
            return jsonify({'status': 'error', 'message': 'Need at least 3 points for polygon'})
        
        # Read current config
        with open('config.yaml', 'r') as f:
            cfg = yaml.safe_load(f)

        # Get camera dimensions for normalization (EXACTLY like your set_roi.py)
        width = cfg["camera"]["width"]
        height = cfg["camera"]["height"]
        
        # Normalize coordinates EXACTLY like your set_roi.py
        # This creates: [[x1, y1], [x2, y2], [x3, y3], ...]
        norm_poly = [[x, y] for (x, y) in points]
        
        # Store EXACTLY like your set_roi.py: cfg["rois"] = [norm_poly]
        # This creates the nested structure: rois: [[[x1,y1], [x2,y2], ...]]
        cfg["rois"] = [norm_poly]

        # Write back to config.yaml
        with open('config.yaml', 'w') as f:
            yaml.safe_dump(cfg, f, default_flow_style=False)

        print(f"✅ ROIs saved using set_roi.py logic: {len(points)} points")
        print(f"✅ Normalized polygon: {norm_poly}")
        return jsonify({'status': 'success', 'message': f'ROIs saved with {len(points)} points'})
        
    except Exception as e:
        print(f"Error saving ROIs: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

def generate_roi_frames():
    """Generate video frames for ROI editing (simple video without detection)"""
    try:
        with open('config.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
        
        cam_w = int(cfg['camera']['width'])
        cam_h = int(cfg['camera']['height'])
        cam_dev = int(cfg['camera'].get('device', 0))
        
        # Use separate camera instance for ROI editing
        camera = cv2.VideoCapture(cam_dev)
        
        while True:
            success, frame = camera.read()
            if not success:
                break
                
            frame = cv2.resize(frame, (cam_w, cam_h))
            
            # Add ROI editor watermark
            cv2.putText(frame, "ROI Editor - Draw polygons on this feed", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Encode and yield frame
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    except Exception as e:
        print(f"Error in ROI video feed: {e}")
    finally:
        if 'camera' in locals():
            camera.release()

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)