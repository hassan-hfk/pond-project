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

# Import camera manager
from camera_manager import camera_manager

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

@app.route('/calibration')
def calibration():
    return render_template('calibration.html')

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

@app.route('/calibration_feed')
def calibration_feed():
    """Video stream for calibration interface"""
    print("🎥 Calibration feed requested")
    return Response(generate_calibration_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

from db_handler import update_feedback, resend_email, get_email_stats

@app.route('/api/feedback/<int:event_id>/<feedback_type>')
def api_feedback(event_id, feedback_type):
    """
    Handle feedback from email links
    Routes: /api/feedback/123/correct or /api/feedback/123/incorrect
    """
    try:
        is_correct = feedback_type.lower() == 'correct'
        
        # Update feedback in database
        update_feedback(event_id, is_correct)
        
        # Return a nice HTML page
        feedback_text = "Correct" if is_correct else "Incorrect"
        emoji = "✅" if is_correct else "❌"
        color = "#27ae60" if is_correct else "#e74c3c"
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Feedback Received</title>
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    font-family: Arial, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }}
                .container {{
                    background: white;
                    padding: 50px;
                    border-radius: 15px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                    text-align: center;
                    max-width: 500px;
                }}
                .emoji {{
                    font-size: 80px;
                    margin-bottom: 20px;
                }}
                h1 {{
                    color: {color};
                    margin: 0 0 20px 0;
                    font-size: 32px;
                }}
                p {{
                    color: #666;
                    font-size: 18px;
                    line-height: 1.6;
                    margin: 0 0 30px 0;
                }}
                .btn {{
                    display: inline-block;
                    background: #3498db;
                    color: white;
                    text-decoration: none;
                    padding: 15px 40px;
                    border-radius: 8px;
                    font-weight: bold;
                    transition: background 0.3s;
                }}
                .btn:hover {{
                    background: #2980b9;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="emoji">{emoji}</div>
                <h1>Thank You!</h1>
                <p>
                    Your feedback has been recorded.<br>
                    You marked this detection as <strong>{feedback_text}</strong>.
                </p>
                <p style="font-size: 14px; color: #999;">
                    Event ID: #{event_id}
                </p>
                <a href="/" class="btn">View Dashboard</a>
            </div>
        </body>
        </html>
        """
        
        return html
        
    except Exception as e:
        print(f"[API] Error processing feedback: {e}")
        return f"""
        <html>
        <body style="font-family: Arial; text-align: center; padding: 50px;">
            <h1 style="color: #e74c3c;">❌ Error</h1>
            <p>Could not process feedback: {str(e)}</p>
            <a href="/" style="color: #3498db;">Go to Dashboard</a>
        </body>
        </html>
        """, 500

@app.route('/api/email/test')
def test_email():
    """Test email configuration"""
    try:
        from email_handler import EmailNotifier
        
        notifier = EmailNotifier(str(CONFIG_PATH))
        success, message = notifier.test_email()
        
        return jsonify({
            'status': 'success' if success else 'error',
            'message': message
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/email/resend/<int:event_id>')
def resend_email_route(event_id):
    """Resend email for a specific event"""
    success, message = resend_email(event_id)
    
    return jsonify({
        'status': 'success' if success else 'error',
        'message': message
    })

@app.route('/api/email/stats')
def email_stats():
    """Get email statistics"""
    stats = get_email_stats()
    
    if stats:
        return jsonify(stats)
    else:
        return jsonify({'error': 'Could not retrieve stats'}), 500

@app.route('/email_settings')
def email_settings():
    """Email settings configuration page"""
    return render_template('email_settings.html')

def generate_calibration_frames():
    """Generate video frames for calibration"""
    try:
        with open(CONFIG_PATH, 'r') as f:
            cfg = yaml.safe_load(f)
        
        cam_w = int(cfg['camera']['width'])
        cam_h = int(cfg['camera']['height'])
        
        # Get camera from manager
        camera = camera_manager.get_camera()
        
        try:
            while True:
                ret, frame = camera_manager.capture_frame()
                if not ret:
                    continue
                
                # Add helpful text
                cv2.putText(frame, "Calibration Mode - Follow instructions", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        finally:
            camera_manager.release_camera()
    
    except Exception as e:
        print(f"Error in calibration video feed: {e}")
        import traceback
        traceback.print_exc()

def generate_roi_frames():
    """Generate video frames for ROI editing"""
    try:
        with open(CONFIG_PATH, 'r') as f:
            cfg = yaml.safe_load(f)
        
        cam_w = int(cfg['camera']['width'])
        cam_h = int(cfg['camera']['height'])
        
        # Get camera from manager
        camera = camera_manager.get_camera()
        
        try:
            while True:
                ret, frame = camera_manager.capture_frame()
                if not ret:
                    continue
                
                cv2.putText(frame, "ROI Editor - Draw polygons on this feed", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        finally:
            camera_manager.release_camera()
    
    except Exception as e:
        print(f"Error in ROI video feed: {e}")
        import traceback
        traceback.print_exc()

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

@app.route('/update_config', methods=['POST'])
def update_config():
    """Handle config updates from config editor - PROTECTS ROIs"""
    try:
        data = request.get_json()
        print(f"[CONFIG] Received update request")
        
        # Load current config
        with open(CONFIG_PATH, 'r') as f:
            cfg = yaml.safe_load(f)
        
        # Store original ROIs - CRITICAL: Don't let form data overwrite ROIs
        original_rois = cfg.get('rois', [])
        print(f"[CONFIG] Protecting {len(original_rois)} ROIs from being overwritten")
        
        # Update with new values
        for section, values in data.items():
            # Skip ROIs - they should only be edited via ROI Editor
            if section == 'rois':
                print("[CONFIG] Skipping ROIs update (use ROI Editor)")
                continue
                
            if section in cfg:
                if isinstance(values, dict) and isinstance(cfg[section], dict):
                    cfg[section].update(values)
                else:
                    cfg[section] = values
            else:
                cfg[section] = values
        
        # Restore original ROIs
        cfg['rois'] = original_rois
        
        # Ensure critical fields have correct types
        # Fix classes_trigger if it got corrupted
        if 'thresholds' in cfg:
            if 'classes_trigger' in cfg['thresholds']:
                # Ensure it's a list
                classes = cfg['thresholds']['classes_trigger']
                if isinstance(classes, dict):
                    # It got corrupted, fix it
                    cfg['thresholds']['classes_trigger'] = ['person']
                    print("[CONFIG] Fixed corrupted classes_trigger")
                elif not isinstance(classes, list):
                    cfg['thresholds']['classes_trigger'] = [str(classes)]
        
        # Create backup before saving
        backup_path = CONFIG_PATH.parent / 'config.yaml.backup'
        with open(backup_path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"[CONFIG] Backup created at {backup_path}")
        
        # Save updated config
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        
        print("[CONFIG] ✅ Configuration saved successfully")
        print(f"[CONFIG] ROIs preserved: {len(cfg.get('rois', []))} polygons")
        
        return jsonify({'status': 'success', 'message': 'Configuration saved successfully'})
        
    except Exception as e:
        print(f"[CONFIG] ❌ Error saving config: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/save_full_config', methods=['POST'])
def save_full_config():
    """Save complete configuration - WITH ROI PROTECTION"""
    try:
        data = request.get_json()
        
        if not data or 'config' not in data:
            return jsonify({'status': 'error', 'message': 'No config data received'}), 400
        
        new_config = data.get('config', {})
        
        # Load original config
        with open(CONFIG_PATH, 'r') as f:
            original_config = yaml.safe_load(f)
        
        # CRITICAL: Preserve ROIs - they should only be edited via ROI Editor
        original_rois = original_config.get('rois', [])
        print(f"[CONFIG] Protecting {len(original_rois)} ROIs")
        
        # Update sections, but skip ROIs
        for section, values in new_config.items():
            if section == 'rois':
                print("[CONFIG] Skipping ROIs (use ROI Editor to modify)")
                continue
                
            if section in original_config:
                if isinstance(values, dict) and isinstance(original_config[section], dict):
                    original_config[section].update(values)
                else:
                    original_config[section] = values
            else:
                original_config[section] = values
        
        # Restore ROIs
        original_config['rois'] = original_rois
        
        # Fix any corrupted data structures
        if 'thresholds' in original_config:
            # Ensure classes_trigger is a list
            if 'classes_trigger' in original_config['thresholds']:
                classes = original_config['thresholds']['classes_trigger']
                if not isinstance(classes, list):
                    if isinstance(classes, dict):
                        # Corrupted - restore default
                        original_config['thresholds']['classes_trigger'] = ['person']
                        print("[CONFIG] Fixed corrupted classes_trigger")
                    else:
                        original_config['thresholds']['classes_trigger'] = [str(classes)]
        
        # Create backup
        backup_path = CONFIG_PATH.parent / 'config.yaml.backup'
        with open(backup_path, 'w') as f:
            yaml.dump(original_config, f, default_flow_style=False, sort_keys=False)
        
        # Save updated config
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(original_config, f, default_flow_style=False, sort_keys=False)
        
        print("[CONFIG] ✅ Full configuration saved successfully")
        print(f"[CONFIG] ROIs preserved: {len(original_config.get('rois', []))} polygons")
        
        return jsonify({'status': 'success', 'message': 'Configuration saved successfully'})
        
    except Exception as e:
        print(f"[CONFIG] ❌ Error saving full config: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

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

@app.route('/save_calibration', methods=['POST'])
def save_calibration():
    """Save calibration data to config.yaml"""
    try:
        data = request.get_json()
        
        with open(CONFIG_PATH, 'r') as f:
            cfg = yaml.safe_load(f)
        
        if 'calibration' not in cfg:
            cfg['calibration'] = {}
        
        cfg['calibration']['ref_box_norm'] = data.get('ref_box_norm')
        cfg['calibration']['ref_height_m'] = float(data.get('ref_height_m'))
        cfg['calibration']['ref_distance_m'] = float(data.get('ref_distance_m'))
        cfg['calibration']['focal_px'] = float(data.get('focal_px'))
        cfg['calibration']['vertical_vp'] = data.get('vertical_vp')
        cfg['calibration']['img_size'] = data.get('img_size')
        
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)
        
        print("✅ Calibration saved successfully")
        return jsonify({'status': 'success', 'message': 'Calibration saved successfully'})
        
    except Exception as e:
        print(f"Error saving calibration: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)})

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
        if isinstance(is_correct, str):
            is_correct_bool = is_correct.lower() == 'true'
        else:
            is_correct_bool = bool(is_correct)

        
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