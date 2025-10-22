import sqlite3
from pathlib import Path

# Define the path to the database
db_path = Path(__file__).parent / 'data' / 'detections.db'

# Connect to the SQLite database
conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

def column_exists(table_name, column_name):
    """Check if a column exists in the specified table."""
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]  # row[1] is column name
    return column_name in columns

def add_column_if_missing(table, column, column_def):
    """Add column to table if it doesn't exist."""
    if column_exists(table, column):
        print(f"[INFO] Column '{column}' already exists in '{table}'.")
    else:
        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_def}")
            conn.commit()
            print(f"[SUCCESS] Added column '{column}' to '{table}'.")
        except Exception as e:
            print(f"[ERROR] Failed to add column '{column}': {e}")

if __name__ == "__main__":
    print(f"[INFO] Updating schema in: {db_path}")
    add_column_if_missing('events', 'email_sent', 'BOOLEAN DEFAULT 0')
    conn.close()
    print("[INFO] Done.")
