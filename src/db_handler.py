import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path

# Define DB path relative to project structure
project_root = Path(__file__).parent.parent
DB_FILE = project_root / 'data' / 'detections.db'

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
    is_correct BOOLEAN
)
''')
conn.commit()

def insert_detection(filename, event_dict, output_dir="data/recordings"):
    """Insert a detection into the DB"""
    cursor.execute('''
        INSERT INTO events (filename, type, class, bbox, height_m, in_roi, key_x, key_y, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        filename,
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
    print(f"Inserted detection for {filename}")

def update_feedback(event_id, is_correct):
    cursor.execute('UPDATE events SET is_correct = ? WHERE id = ?', (is_correct, event_id))
    conn.commit()
    print(f"Updated feedback for event id {event_id} -> {is_correct}")

def close_db():
    conn.close()