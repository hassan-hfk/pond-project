import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path
import time

# Define DB path relative to project structure
project_root = Path(__file__).parent.parent
DB_FILE = project_root / 'data' / 'detections.db'
CONFIG_PATH = project_root / 'config' / 'config.yaml'

# Ensure data directory exists
DB_FILE.parent.mkdir(parents=True, exist_ok=True)

# Connect to the database
conn = sqlite3.connect(str(DB_FILE), check_same_thread=False)
cursor = conn.cursor()

# Create table for detections
cursor.execute('''
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    type TEXT,
    class TEXT,
    bbox TEXT,
    height_m REAL,
    in_roi BOOLEAN,
    key_x INTEGER,
    key_y INTEGER,
    timestamp REAL,
    is_correct BOOLEAN,
    email_sent BOOLEAN DEFAULT 0
)
''')
conn.commit()

# Import email notifier
EMAIL_AVAILABLE = False
email_notifier = None

try:
    from email_handler import EmailNotifier
    email_notifier = EmailNotifier(str(CONFIG_PATH))
    EMAIL_AVAILABLE = True
    print(f"[DB] Email handler initialized - Enabled: {email_notifier.enabled}")
except Exception as e:
    print(f"[DB] Warning: Email handler not available: {e}")
    import traceback
    traceback.print_exc()

def insert_detection(filename, event_dict, output_dir="data/recordings"):
    """Insert a detection into the DB and send email notification"""
    try:
        cursor.execute('''
            INSERT INTO events (filename, type, class, bbox, height_m, in_roi, key_x, key_y, timestamp, email_sent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            filename,
            event_dict.get('type'),
            event_dict.get('class'),
            json.dumps(event_dict.get('bbox')),
            event_dict.get('height_m'),
            int(event_dict.get('in_roi')),
            event_dict.get('key')[0],
            event_dict.get('key')[1],
            event_dict.get('timestamp'),
            0  # email_sent initially False
        ))
        conn.commit()
        
        event_id = cursor.lastrowid
        print(f"[DB] ✅ Inserted detection with ID: {event_id} for file: {filename}")
        
        # Send email notification in a separate thread to avoid blocking
        if EMAIL_AVAILABLE and email_notifier and email_notifier.enabled:
            import threading
            email_thread = threading.Thread(
                target=send_email_async,
                args=(event_id, filename, event_dict, output_dir),
                daemon=True
            )
            email_thread.start()
        else:
            if not EMAIL_AVAILABLE:
                print(f"[DB] ⚠️ Email handler not available")
            elif not email_notifier:
                print(f"[DB] ⚠️ Email notifier is None")
            elif not email_notifier.enabled:
                print(f"[DB] ℹ️ Email notifications disabled in config")
        
        return event_id
        
    except Exception as e:
        print(f"[DB] ❌ Error inserting detection: {e}")
        import traceback
        traceback.print_exc()
        return None

def send_email_async(event_id, filename, event_dict, output_dir):
    """Send email asynchronously to avoid blocking the main thread"""
    try:
        # Wait for video file to be ready (WebM conversion takes time)
        video_path = Path(output_dir) / filename
        
        # Wait up to 30 seconds for the file to exist
        max_wait = 30
        wait_count = 0
        while not video_path.exists() and wait_count < max_wait:
            print(f"[DB] Waiting for video file: {filename} ({wait_count}/{max_wait}s)")
            time.sleep(1)
            wait_count += 1
        
        if not video_path.exists():
            print(f"[DB] ⚠️ Video file not ready after {max_wait}s: {video_path}")
            # Try to find any related file (JPEG directory or partial WebM)
            recordings_dir = Path(output_dir)
            possible_files = list(recordings_dir.glob(f"*{filename.split('_')[0]}*"))
            if possible_files:
                print(f"[DB] Found alternative files: {[f.name for f in possible_files]}")
        
        print(f"[DB] 📧 Attempting to send email for event {event_id}...")
        print(f"[DB] Video path: {video_path}")
        print(f"[DB] File exists: {video_path.exists()}")
        
        email_success = email_notifier.send_detection_email(
            event_id=event_id,
            event_data=event_dict,
            video_path=str(video_path)
        )
        
        if email_success:
            # Mark email as sent in database
            cursor.execute('UPDATE events SET email_sent = 1 WHERE id = ?', (event_id,))
            conn.commit()
            print(f"[DB] ✅ Email sent successfully for event {event_id}")
        else:
            print(f"[DB] ❌ Failed to send email for event {event_id}")
            
    except Exception as e:
        print(f"[DB] ❌ Error in email async thread: {e}")
        import traceback
        traceback.print_exc()

def update_feedback(event_id, is_correct):
    """Update feedback for an event"""
    try:
        cursor.execute('UPDATE events SET is_correct = ? WHERE id = ?', (is_correct, event_id))
        conn.commit()
        print(f"[DB] ✅ Updated feedback for event {event_id} -> {is_correct}")
    except Exception as e:
        print(f"[DB] ❌ Error updating feedback: {e}")

def resend_email(event_id):
    """Resend email for a specific event"""
    if not EMAIL_AVAILABLE or not email_notifier or not email_notifier.enabled:
        return False, "Email not available or disabled"
    
    try:
        # Get event from database
        cursor.execute('SELECT * FROM events WHERE id = ?', (event_id,))
        row = cursor.fetchone()
        
        if not row:
            return False, "Event not found"
        
        # Reconstruct event data
        event_data = {
            'type': row[2],
            'class': row[3],
            'bbox': json.loads(row[4]) if row[4] else [],
            'height_m': row[5],
            'in_roi': bool(row[6]),
            'timestamp': row[9]
        }
        
        filename = row[1]
        video_path = project_root / 'data' / 'recordings' / filename
        
        if not video_path.exists():
            return False, f"Video file not found: {filename}"
        
        # Send email
        success = email_notifier.send_detection_email(
            event_id=event_id,
            event_data=event_data,
            video_path=str(video_path)
        )
        
        if success:
            cursor.execute('UPDATE events SET email_sent = 1 WHERE id = ?', (event_id,))
            conn.commit()
            return True, "Email sent successfully"
        else:
            return False, "Failed to send email"
            
    except Exception as e:
        print(f"[DB] Error in resend_email: {e}")
        import traceback
        traceback.print_exc()
        return False, f"Error: {str(e)}"

def get_email_stats():
    """Get statistics about email notifications"""
    try:
        cursor.execute('SELECT COUNT(*) FROM events WHERE email_sent = 1')
        sent_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM events WHERE email_sent = 0')
        pending_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM events')
        total_count = cursor.fetchone()[0]
        
        return {
            'total_events': total_count,
            'emails_sent': sent_count,
            'emails_pending': pending_count,
            'email_enabled': EMAIL_AVAILABLE and email_notifier.enabled if EMAIL_AVAILABLE else False
        }
    except Exception as e:
        print(f"[DB] Error getting email stats: {e}")
        return None

def close_db():
    conn.close()