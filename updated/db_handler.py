import sqlite3
import json
import os
from datetime import datetime

DB_FILE = 'detections.db'

# Connect to the database (creates if it doesn't exist)
conn = sqlite3.connect(DB_FILE)
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
    is_correct BOOLEAN
)
''')
conn.commit()

# Function to insert a detection
def insert_detection(filename, event_dict, output_dir="recordings"):
    """
    Insert a detection into the DB.
    filename: video filename (will store full path)
    event_dict: dict containing event info
    output_dir: folder where video is saved
    """
    video_path = os.path.abspath(os.path.join(output_dir, filename))

    cursor.execute('''
        INSERT INTO events (filename, type, class, bbox, height_m, in_roi, key_x, key_y, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        video_path,
        event_dict.get('type'),
        event_dict.get('class'),
        json.dumps(event_dict.get('bbox')),
        event_dict.get('height_m'),
        int(event_dict.get('in_roi')),
        event_dict.get('key')[0],
        event_dict.get('key')[1],
        event_dict.get('timestamp')
    ))
    conn.commit()
    print(f"Inserted detection for {video_path}")

# Function to update feedback (correct/incorrect)
def update_feedback(event_id, is_correct):
    cursor.execute('UPDATE events SET is_correct = ? WHERE id = ?', (is_correct, event_id))
    conn.commit()
    print(f"Updated feedback for event id {event_id} -> {is_correct}")

# Function to batch insert detections from frames folder (kept for backward compatibility)
def insert_detections_from_folder(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith('.jpg') or file.endswith('.png'):
            # Example: create a dummy detection for each frame
            dummy_event = {
                'type': 'other_detected',
                'class': 'cat',
                'bbox': [0, 0, 100, 100],
                'height_m': 2.0,
                'in_roi': False,
                'key': (1, 1),
                'timestamp': datetime.now().timestamp()
            }
            insert_detection(file, dummy_event, output_dir=folder_path)

# Close DB connection when done
def close_db():
    conn.close()
