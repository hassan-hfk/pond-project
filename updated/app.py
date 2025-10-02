from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from run_child_monitor import set_detection_status, generate_frame, detection_active
import yaml
import os
import cv2
import numpy as np
import glob
import sqlite3
import json
from datetime import datetime
import shutil

app = Flask(__name__, static_folder='static', static_url_path='/static')

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

@app.route('/config_editor')
def config_editor():
    return render_template('config_editor.html')

@app.route('/get_full_config')
def get_full_config():
    """Get complete config.yaml content"""
    try:
        with open('config.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
        
        return jsonify({
            'config': cfg
        })
    except Exception as e:
        print(f"Error reading full config: {e}")
        return jsonify({'config': {}})

@app.route('/save_full_config', methods=['POST'])
def save_full_config():
    """Save complete configuration to config.yaml while preserving structure"""
    try:
        data = request.get_json()
        new_config = data.get('config', {})
        
        # Read the original config to preserve comments and order (as much as possible)
        with open('config.yaml', 'r') as f:
            original_content = f.read()
            original_config = yaml.safe_load(original_content)
        
        # Update the original config with new values while preserving structure
        updated_config = update_config_structure(original_config, new_config)
        
        # Write back to file
        with open('config.yaml', 'w') as f:
            yaml.dump(updated_config, f, default_flow_style=False, sort_keys=False)
        
        print("✅ Full configuration saved successfully")
        return jsonify({'status': 'success', 'message': 'Configuration saved successfully'})
        
    except Exception as e:
        print(f"Error saving full config: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

def update_config_structure(original, new):
    """Update original config structure with new values"""
    updated = original.copy()
    
    # Update each section
    sections = ['calibration', 'camera', 'gpio', 'logging', 'model', 'rois', 'thresholds']
    
    for section in sections:
        if section in new:
            if section in updated:
                # Update existing section
                if isinstance(updated[section], dict) and isinstance(new[section], dict):
                    updated[section].update(new[section])
                else:
                    updated[section] = new[section]
            else:
                # Add new section
                updated[section] = new[section]
    
    return updated

# Database helper function
def get_events_from_db():
    """Get all events from database - now expecting only filename in database"""
    conn = sqlite3.connect('detections.db', check_same_thread=False)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, filename, type, class, bbox, height_m, in_roi, key_x, key_y, timestamp, is_correct
        FROM events 
        ORDER BY timestamp DESC
    ''')
    
    events = []
    for row in cursor.fetchall():
        # Now filename should be just the filename, not full path
        filename = row[1] if row[1] else 'unknown'
        
        # Check if video file exists in recordings folder
        recordings_path = os.path.join('recordings', filename)
        video_exists = os.path.exists(recordings_path)
        
        events.append({
            'id': row[0],
            'filename': filename,  # Just filename now
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
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return False
            
        # Read first frame
        success, frame = cap.read()
        if not success:
            print(f"❌ Could not read frame from: {video_path}")
            cap.release()
            return False
        
        # Resize to thumbnail size
        frame = cv2.resize(frame, (320, 240))
        
        # Save as JPEG
        cv2.imwrite(thumbnail_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        cap.release()
        
        print(f"✅ Thumbnail generated: {thumbnail_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error generating thumbnail for {video_path}: {e}")
        return False

@app.route('/api/recordings')
def get_recordings():
    """Get recordings from database and copy them from recordings to static/videos for serving"""
    events = get_events_from_db()
    videos = []
    
    # Ensure directories exist
    os.makedirs('recordings', exist_ok=True)
    os.makedirs('static/videos', exist_ok=True)
    os.makedirs('static/thumbnails', exist_ok=True)
    
    for event in events:
        if event['video_exists'] and event['filename']:
            filename = event['filename']
            source_path = os.path.join('recordings', filename)
            static_video_path = os.path.join('static/videos', filename)
            
            # Get file stats if file exists
            if os.path.exists(source_path):
                stat = os.stat(source_path)
                
                # Format file size
                size_kb = stat.st_size // 1024
                size_mb = size_kb // 1024
                if size_mb > 0:
                    size_str = f"{size_mb} MB"
                else:
                    size_str = f"{size_kb} KB"
                
                # Copy video from recordings to static/videos if not already there
                if not os.path.exists(static_video_path):
                    try:
                        shutil.copy2(source_path, static_video_path)
                        print(f"✅ Copied {filename} from recordings/ to static/videos/")
                    except Exception as e:
                        print(f"❌ Error copying {filename}: {e}")
                        continue  # Skip this video if copy fails
                
                # Generate thumbnail for the video
                thumbnail_filename = filename.rsplit('.', 1)[0] + '.jpg'
                thumbnail_path = os.path.join('static/thumbnails', thumbnail_filename)
                
                # Generate thumbnail if it doesn't exist
                if not os.path.exists(thumbnail_path):
                    if generate_thumbnail(source_path, thumbnail_path):
                        print(f"✅ Generated thumbnail: {thumbnail_filename}")
                    else:
                        print(f"❌ Failed to generate thumbnail for {filename}")
                        # Use default thumbnail if generation fails
                        thumbnail_filename = "default-thumbnail.jpg"
                else:
                    print(f"✅ Thumbnail already exists: {thumbnail_filename}")
                
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
        
        conn = sqlite3.connect('detections.db', check_same_thread=False)
        cursor = conn.cursor()
        
        cursor.execute('UPDATE events SET is_correct = ? WHERE id = ?', (is_correct_bool, event_id))
        conn.commit()
        conn.close()
        
        print(f"✅ Feedback updated for event {event_id}: is_correct = {is_correct_bool}")
        return jsonify({'status': 'success', 'message': f'Feedback updated: {is_correct_bool}'})
        
    except Exception as e:
        print(f"Error updating feedback: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

# Create default thumbnail if it doesn't exist
def create_default_thumbnail():
    """Create a default thumbnail image"""
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/thumbnails', exist_ok=True)
    default_thumb_path = 'static/default-thumbnail.jpg'
    default_thumb_path2 = 'static/thumbnails/default-thumbnail.jpg'
    
    if not os.path.exists(default_thumb_path):
        # Create a simple gray placeholder image
        img = np.ones((240, 320, 3), dtype=np.uint8) * 128
        cv2.putText(img, "No Thumbnail", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, "Click to play video", (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(default_thumb_path, img)
        print("✅ Default thumbnail created at static/default-thumbnail.jpg")
    
    # Also copy to thumbnails folder
    if os.path.exists(default_thumb_path) and not os.path.exists(default_thumb_path2):
        shutil.copy2(default_thumb_path, default_thumb_path2)
        print("✅ Default thumbnail copied to static/thumbnails/default-thumbnail.jpg")

# Create a test video if none exists
def create_test_video():
    """Create a test video file if no videos exist"""
    test_video_path = 'recordings/test_video.webm'
    if not os.path.exists(test_video_path):
        try:
            os.makedirs('recordings', exist_ok=True)
            # Create a simple test video using OpenCV
            fourcc = cv2.VideoWriter_fourcc(*'VP80')  # Use VP80 for WebM
            out = cv2.VideoWriter(test_video_path, fourcc, 20.0, (640, 480))
            
            for i in range(100):  # 5 seconds at 20 fps
                # Create a frame with moving text
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                frame[:] = (i * 2 % 255, 100, 100)  # Changing color
                cv2.putText(frame, f'Test Video Frame {i}', (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, 'Child Monitor System', (50, 300), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                out.write(frame)
            
            out.release()
            print("✅ Test video created: recordings/test_video.webm")
            
            # Also copy to static/videos for immediate access
            static_test_path = 'static/videos/test_video.webm'
            shutil.copy2(test_video_path, static_test_path)
            print("✅ Test video copied to static/videos/test_video.webm")
            
            # Generate thumbnail for test video
            thumbnail_path = 'static/thumbnails/test_video.jpg'
            generate_thumbnail(test_video_path, thumbnail_path)
            
        except Exception as e:
            print(f"❌ Could not create test video: {e}")

# Debug endpoint to check static files
@app.route('/api/debug/static')
def debug_static():
    """Debug endpoint to check what static files are available"""
    static_info = {
        'static_folder': app.static_folder,
        'static_url_path': app.static_url_path,
        'files': {}
    }
    
    # Check default thumbnail
    thumb_path = 'static/default-thumbnail.jpg'
    static_info['files']['default_thumbnail'] = {
        'path': thumb_path,
        'exists': os.path.exists(thumb_path),
        'url': '/static/default-thumbnail.jpg'
    }
    
    # Check recordings folder
    recordings_path = 'recordings'
    if os.path.exists(recordings_path):
        recording_files = glob.glob(os.path.join(recordings_path, '*'))
        static_info['files']['recordings'] = []
        for rf in recording_files:
            static_info['files']['recordings'].append({
                'name': os.path.basename(rf),
                'size': os.path.getsize(rf),
                'path': rf
            })
    
    # Check videos in static/videos
    videos_path = 'static/videos'
    if os.path.exists(videos_path):
        video_files = glob.glob(os.path.join(videos_path, '*'))
        static_info['files']['static_videos'] = []
        for vf in video_files:
            static_info['files']['static_videos'].append({
                'name': os.path.basename(vf),
                'size': os.path.getsize(vf),
                'url': f'/static/videos/{os.path.basename(vf)}'
            })
    
    # Check thumbnails in static/thumbnails
    thumbnails_path = 'static/thumbnails'
    if os.path.exists(thumbnails_path):
        thumbnail_files = glob.glob(os.path.join(thumbnails_path, '*.jpg'))
        static_info['files']['thumbnails'] = []
        for tf in thumbnail_files:
            static_info['files']['thumbnails'].append({
                'name': os.path.basename(tf),
                'size': os.path.getsize(tf),
                'url': f'/static/thumbnails/{os.path.basename(tf)}'
            })
    
    return jsonify(static_info)

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('recordings', exist_ok=True)
    os.makedirs('static/videos', exist_ok=True)
    os.makedirs('static/thumbnails', exist_ok=True)
    
    # Create default thumbnail
    create_default_thumbnail()
    
    # Create test video if no videos exist
    create_test_video()
    
    print("🚀 Starting Flask application...")
    print("📁 Checking static files...")
    
    # Check if default thumbnail exists
    if os.path.exists('static/default-thumbnail.jpg'):
        print("✅ Default thumbnail: static/default-thumbnail.jpg")
    else:
        print("❌ Default thumbnail missing!")
    
    # List available video files in recordings
    recording_files = glob.glob('recordings/*.webm') + glob.glob('recordings/*.mp4') + glob.glob('recordings/*.avi')
    print(f"🎥 Found {len(recording_files)} video files in recordings/")
    for rf in recording_files:
        size_mb = os.path.getsize(rf) / (1024 * 1024)
        print(f"   - {os.path.basename(rf)} ({size_mb:.1f} MB)")
    
    # List available video files in static/videos
    static_video_files = glob.glob('static/videos/*.webm') + glob.glob('static/videos/*.mp4') + glob.glob('static/videos/*.avi')
    print(f"🌐 Found {len(static_video_files)} video files in static/videos/ (web accessible)")
    for svf in static_video_files:
        size_mb = os.path.getsize(svf) / (1024 * 1024)
        print(f"   - {os.path.basename(svf)} ({size_mb:.1f} MB)")
    
    # List available thumbnails
    thumbnail_files = glob.glob('static/thumbnails/*.jpg')
    print(f"🖼️  Found {len(thumbnail_files)} thumbnail files in static/thumbnails/")
    for tf in thumbnail_files:
        size_kb = os.path.getsize(tf) / 1024
        print(f"   - {os.path.basename(tf)} ({size_kb:.1f} KB)")
    
    if not recording_files and not static_video_files:
        print("❌ No video files found!")
    
    print("🔍 Debug URL: http://127.0.0.1:5000/api/debug/static")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
