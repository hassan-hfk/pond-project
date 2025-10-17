import os
import sys
from pathlib import Path

# Add project root and scripts to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'scripts'))

from flask import Flask, render_template, Response, jsonify, request
import yaml
import cv2
import numpy as np
import glob
import sqlite3
import json
from datetime import datetime
import shutil

# Define paths relative to project root
TEMPLATE_FOLDER = project_root / 'web' / 'templates'
STATIC_FOLDER = project_root / 'web' / 'static'
CONFIG_PATH = project_root / 'config' / 'config.yaml'
DB_PATH = project_root / 'data' / 'detections.db'
RECORDINGS_PATH = project_root / 'data' / 'recordings'
LOGS_PATH = project_root / 'logs'

app = Flask(__name__, 
            template_folder=str(TEMPLATE_FOLDER),
            static_folder=str(STATIC_FOLDER),
            static_url_path='/static')

# Import detection module from scripts folder
from run_child_monitor import set_detection_status, generate_frame, detection_active

print(f"🚀 App starting - initial detection_active: {detection_active}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/roi_editor')
def roi_editor():
    return render_template('roi_editor.html')

@app.route('/config_editor')
def config_editor():
    return render_template('config_editor.html')

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
        with open(CONFIG_PATH, 'r') as f:
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
        with open(CONFIG_PATH, 'r') as f:
            cfg = yaml.safe_load(f)
        
        if 'model' not in cfg:
            cfg['model'] = {}
        cfg['model']['conf_thresh'] = float(confidence)
        
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)
        
        print(f"✅ Confidence threshold updated to: {confidence}")
        return jsonify({'status': 'success', 'message': f'Confidence updated to {confidence}'})
    except Exception as e:
        print(f"Error updating confidence: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/save_rois', methods=['POST'])
def save_rois():
    """Save ROI polygons to config.yaml"""
    try:
        data = request.get_json()
        points = data.get('points', [])
        
        if len(points) < 3:
            return jsonify({'status': 'error', 'message': 'Need at least 3 points for polygon'})
        
        with open(CONFIG_PATH, 'r') as f:
            cfg = yaml.safe_load(f)

        width = cfg["camera"]["width"]
        height = cfg["camera"]["height"]
        
        norm_poly = [[x, y] for (x, y) in points]
        cfg["rois"] = [norm_poly]

        with open(CONFIG_PATH, 'w') as f:
            yaml.safe_dump(cfg, f, default_flow_style=False)

        print(f"✅ ROIs saved: {len(points)} points")
        return jsonify({'status': 'success', 'message': f'ROIs saved with {len(points)} points'})
        
    except Exception as e:
        print(f"Error saving ROIs: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

def generate_roi_frames():
    """Generate video frames for ROI editing"""
    try:
        with open(CONFIG_PATH, 'r') as f:
            cfg = yaml.safe_load(f)
        
        cam_w = int(cfg['camera']['width'])
        cam_h = int(cfg['camera']['height'])
        cam_dev = int(cfg['camera'].get('device', 0))
        
        camera = cv2.VideoCapture(cam_dev)
        
        while True:
            success, frame = camera.read()
            if not success:
                break
                
            frame = cv2.resize(frame, (cam_w, cam_h))
            
            cv2.putText(frame, "ROI Editor - Draw polygons on this feed", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
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

@app.route('/get_full_config')
def get_full_config():
    """Get complete config.yaml content"""
    try:
        with open(CONFIG_PATH, 'r') as f:
            cfg = yaml.safe_load(f)
        return jsonify({'config': cfg})
    except Exception as e:
        print(f"Error reading full config: {e}")
        return jsonify({'config': {}})

@app.route('/save_full_config', methods=['POST'])
def save_full_config():
    """Save complete configuration to config.yaml"""
    try:
        data = request.get_json()
        new_config = data.get('config', {})
        
        with open(CONFIG_PATH, 'r') as f:
            original_config = yaml.safe_load(f)
        
        updated_config = update_config_structure(original_config, new_config)
        
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(updated_config, f, default_flow_style=False, sort_keys=False)
        
        print("✅ Full configuration saved successfully")
        return jsonify({'status': 'success', 'message': 'Configuration saved successfully'})
        
    except Exception as e:
        print(f"Error saving full config: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

def update_config_structure(original, new):
    """Update original config structure with new values"""
    updated = original.copy()
    sections = ['calibration', 'camera', 'gpio', 'logging', 'model', 'rois', 'thresholds']
    
    for section in sections:
        if section in new:
            if section in updated:
                if isinstance(updated[section], dict) and isinstance(new[section], dict):
                    updated[section].update(new[section])
                else:
                    updated[section] = new[section]
            else:
                updated[section] = new[section]
    
    return updated

def get_events_from_db():
    """Get all events from database"""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, filename, type, class, bbox, height_m, in_roi, key_x, key_y, timestamp, is_correct
        FROM events 
        ORDER BY timestamp DESC
    ''')
    
    events = []
    for row in cursor.fetchall():
        filename = row[1] if row[1] else 'unknown'
        recordings_path = RECORDINGS_PATH / filename
        video_exists = recordings_path.exists()
        
        events.append({
            'id': row[0],
            'filename': filename,
            'type': row[2],
            'class': row[3],
            'bbox': json.loads(row[4]) if row[4] else [],
            'height_m': row[5],
            'in_roi': bool(row[6]),
            'key': (row[7], row[8]),
            'timestamp': row[9],
            'is_correct': row[10],
            'video_exists': video_exists,
            'basename': filename
        })
    
    conn.close()
    return events

def generate_thumbnail(video_path, thumbnail_path):
    """Generate thumbnail from video file"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False
            
        success, frame = cap.read()
        if not success:
            cap.release()
            return False
        
        frame = cv2.resize(frame, (320, 240))
        cv2.imwrite(str(thumbnail_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        cap.release()
        
        print(f"✅ Thumbnail generated: {thumbnail_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error generating thumbnail: {e}")
        return False

@app.route('/api/recordings')
def get_recordings():
    """Get recordings from database"""
    events = get_events_from_db()
    videos = []
    
    # Ensure directories exist
    RECORDINGS_PATH.mkdir(parents=True, exist_ok=True)
    (STATIC_FOLDER / 'videos').mkdir(parents=True, exist_ok=True)
    (STATIC_FOLDER / 'thumbnails').mkdir(parents=True, exist_ok=True)
    
    for event in events:
        if event['video_exists'] and event['filename']:
            filename = event['filename']
            source_path = RECORDINGS_PATH / filename
            static_video_path = STATIC_FOLDER / 'videos' / filename
            
            if source_path.exists():
                stat = source_path.stat()
                
                size_kb = stat.st_size // 1024
                size_mb = size_kb // 1024
                size_str = f"{size_mb} MB" if size_mb > 0 else f"{size_kb} KB"
                
                if not static_video_path.exists():
                    try:
                        shutil.copy2(source_path, static_video_path)
                        print(f"✅ Copied {filename} to static/videos/")
                    except Exception as e:
                        print(f"❌ Error copying {filename}: {e}")
                        continue
                
                thumbnail_filename = filename.rsplit('.', 1)[0] + '.jpg'
                thumbnail_path = STATIC_FOLDER / 'thumbnails' / thumbnail_filename
                
                if not thumbnail_path.exists():
                    generate_thumbnail(source_path, thumbnail_path)
                
                videos.append({
                    'id': event['id'],
                    'name': filename,
                    'url': f"/static/videos/{filename}",
                    'thumbnail': f"/static/thumbnails/{thumbnail_filename}",
                    'size': size_str,
                    'modified': datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    'type': event['type'],
                    'class': event['class'],
                    'in_roi': event['in_roi'],
                    'height_m': event['height_m'],
                    'is_correct': event['is_correct'],
                    'timestamp': datetime.fromtimestamp(event['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
                })
    
    return jsonify(videos)

@app.route('/update_feedback/<int:event_id>/<is_correct>')
def update_feedback(event_id, is_correct):
    """Update feedback for an event"""
    try:
        is_correct_bool = is_correct.lower() == 'true'
        
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        cursor = conn.cursor()
        
        cursor.execute('UPDATE events SET is_correct = ? WHERE id = ?', (is_correct_bool, event_id))
        conn.commit()
        conn.close()
        
        print(f"✅ Feedback updated for event {event_id}")
        return jsonify({'status': 'success', 'message': f'Feedback updated: {is_correct_bool}'})
        
    except Exception as e:
        print(f"Error updating feedback: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    # Create necessary directories
    RECORDINGS_PATH.mkdir(parents=True, exist_ok=True)
    (STATIC_FOLDER / 'videos').mkdir(parents=True, exist_ok=True)
    (STATIC_FOLDER / 'thumbnails').mkdir(parents=True, exist_ok=True)
    LOGS_PATH.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Starting Flask application...")
    app.run(debug=True, host='127.0.0.1', port=5000)