"""
FastAPI Feedback Service
Handles email feedback from anywhere on the internet
Run on separate port (8000) for external access
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
from pathlib import Path
import yaml
from datetime import datetime

app = FastAPI(title="Detection Feedback API", version="1.0")

# Enable CORS so it works from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
project_root = Path(__file__).parent.parent
DB_FILE = project_root / 'data' / 'detections.db'
CONFIG_PATH = project_root / 'config' / 'config.yaml'

# Load config
with open(CONFIG_PATH, 'r') as f:
    cfg = yaml.safe_load(f)

DASHBOARD_URL = cfg.get('email', {}).get('app_url', 'http://localhost:5000')

def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(str(DB_FILE), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "service": "Detection Feedback API",
        "version": "1.0",
        "status": "running",
        "endpoints": {
            "feedback": "/feedback/{event_id}/{feedback_type}",
            "stats": "/stats",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM events")
        count = cursor.fetchone()[0]
        conn.close()
        
        return {
            "status": "healthy",
            "database": "connected",
            "total_events": count
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/feedback/{event_id}/{feedback_type}", response_class=HTMLResponse)
async def handle_feedback(event_id: int, feedback_type: str):
    """
    Handle feedback from email links
    Works from anywhere on the internet
    
    Args:
        event_id: Database event ID
        feedback_type: 'correct' or 'incorrect'
    
    Returns:
        HTML page confirming feedback
    """
    try:
        # Validate feedback type
        if feedback_type.lower() not in ['correct', 'incorrect']:
            raise HTTPException(status_code=400, detail="Invalid feedback type")
        
        is_correct = feedback_type.lower() == 'correct'
        
        # Update database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if event exists
        cursor.execute("SELECT id, type, class, timestamp FROM events WHERE id = ?", (event_id,))
        event = cursor.fetchone()
        
        if not event:
            conn.close()
            raise HTTPException(status_code=404, detail="Event not found")
        
        # Update feedback
        cursor.execute(
            "UPDATE events SET is_correct = ? WHERE id = ?",
            (1 if is_correct else 0, event_id)
        )
        conn.commit()
        conn.close()
        
        # Prepare response data
        feedback_text = "Correct" if is_correct else "Incorrect"
        emoji = "✅" if is_correct else "❌"
        color = "#27ae60" if is_correct else "#e74c3c"
        
        event_type = event['type'] if event else 'unknown'
        event_class = event['class'] if event else 'unknown'
        event_time = datetime.fromtimestamp(event['timestamp']).strftime("%Y-%m-%d %H:%M:%S") if event else 'unknown'
        
        # Log feedback
        print(f"[FEEDBACK] Event {event_id} marked as {feedback_text} at {datetime.now()}")
        
        # Return beautiful HTML response
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Feedback Received - Detection System</title>
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 20px;
                }}
                
                .container {{
                    background: white;
                    padding: 50px 40px;
                    border-radius: 20px;
                    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                    text-align: center;
                    max-width: 600px;
                    width: 100%;
                    animation: slideUp 0.5s ease;
                }}
                
                @keyframes slideUp {{
                    from {{
                        opacity: 0;
                        transform: translateY(30px);
                    }}
                    to {{
                        opacity: 1;
                        transform: translateY(0);
                    }}
                }}
                
                .emoji {{
                    font-size: 100px;
                    margin-bottom: 20px;
                    animation: bounce 0.6s ease;
                }}
                
                @keyframes bounce {{
                    0%, 100% {{ transform: scale(1); }}
                    50% {{ transform: scale(1.1); }}
                }}
                
                h1 {{
                    color: {color};
                    margin: 0 0 20px 0;
                    font-size: 36px;
                    font-weight: 700;
                }}
                
                .message {{
                    color: #555;
                    font-size: 20px;
                    line-height: 1.6;
                    margin: 0 0 30px 0;
                }}
                
                .event-details {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 10px;
                    margin: 30px 0;
                    text-align: left;
                }}
                
                .event-details table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                
                .event-details td {{
                    padding: 10px;
                    border-bottom: 1px solid #e9ecef;
                }}
                
                .event-details td:first-child {{
                    font-weight: 600;
                    color: #7f8c8d;
                    width: 40%;
                }}
                
                .event-details td:last-child {{
                    color: #2c3e50;
                }}
                
                .event-details tr:last-child td {{
                    border-bottom: none;
                }}
                
                .event-id {{
                    font-size: 14px;
                    color: #999;
                    margin: 20px 0;
                }}
                
                .buttons {{
                    display: flex;
                    gap: 15px;
                    justify-content: center;
                    margin-top: 30px;
                }}
                
                .btn {{
                    display: inline-block;
                    padding: 15px 40px;
                    border-radius: 10px;
                    font-weight: 600;
                    text-decoration: none;
                    transition: all 0.3s ease;
                    font-size: 16px;
                }}
                
                .btn-primary {{
                    background: #3498db;
                    color: white;
                }}
                
                .btn-primary:hover {{
                    background: #2980b9;
                    transform: translateY(-2px);
                    box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
                }}
                
                .btn-secondary {{
                    background: #ecf0f1;
                    color: #2c3e50;
                }}
                
                .btn-secondary:hover {{
                    background: #d5dbdb;
                    transform: translateY(-2px);
                }}
                
                .footer {{
                    margin-top: 30px;
                    padding-top: 20px;
                    border-top: 1px solid #ecf0f1;
                    color: #999;
                    font-size: 12px;
                }}
                
                @media (max-width: 600px) {{
                    .container {{
                        padding: 30px 20px;
                    }}
                    
                    h1 {{
                        font-size: 28px;
                    }}
                    
                    .message {{
                        font-size: 16px;
                    }}
                    
                    .buttons {{
                        flex-direction: column;
                    }}
                    
                    .btn {{
                        width: 100%;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="emoji">{emoji}</div>
                <h1>Feedback Received!</h1>
                <p class="message">
                    Thank you for your feedback.<br>
                    You marked this detection as <strong>{feedback_text}</strong>.
                </p>
                
                <div class="event-details">
                    <table>
                        <tr>
                            <td>Event Type:</td>
                            <td>{event_type.replace('_', ' ').title()}</td>
                        </tr>
                        <tr>
                            <td>Detected Class:</td>
                            <td>{event_class.upper()}</td>
                        </tr>
                        <tr>
                            <td>Timestamp:</td>
                            <td>{event_time}</td>
                        </tr>
                        <tr>
                            <td>Feedback:</td>
                            <td><strong style="color: {color};">{emoji} {feedback_text}</strong></td>
                        </tr>
                    </table>
                </div>
                
                <p class="event-id">Event ID: #{event_id}</p>
                
                <div class="buttons">
                    <a href="{DASHBOARD_URL}/?showEventList=true" class="btn btn-primary">
                        📹 View All Events
                    </a>
                    <a href="{DASHBOARD_URL}" class="btn btn-secondary">
                        🏠 Dashboard
                    </a>
                </div>
                
                <div class="footer">
                    <p>Child Detection System</p>
                    <p>Your feedback helps improve detection accuracy</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return HTMLResponse(content=html_content)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[FEEDBACK] Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get feedback statistics"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM events")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM events WHERE is_correct = 1")
        correct = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM events WHERE is_correct = 0")
        incorrect = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM events WHERE is_correct IS NULL")
        pending = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM events WHERE email_sent = 1")
        emails_sent = cursor.fetchone()[0]
        
        conn.close()
        
        accuracy = (correct / (correct + incorrect) * 100) if (correct + incorrect) > 0 else 0
        
        return {
            "total_events": total,
            "feedback_received": correct + incorrect,
            "feedback_pending": pending,
            "marked_correct": correct,
            "marked_incorrect": incorrect,
            "accuracy_percentage": round(accuracy, 2),
            "emails_sent": emails_sent
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/events")
async def get_events(limit: int = 50):
    """Get recent events with feedback status"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, filename, type, class, timestamp, is_correct, email_sent
            FROM events
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        events = []
        for row in cursor.fetchall():
            events.append({
                "id": row['id'],
                "filename": row['filename'],
                "type": row['type'],
                "class": row['class'],
                "timestamp": datetime.fromtimestamp(row['timestamp']).isoformat(),
                "is_correct": row['is_correct'],
                "email_sent": bool(row['email_sent'])
            })
        
        conn.close()
        
        return {"events": events, "count": len(events)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    print("="*50)
    print("🚀 Detection Feedback API Server")
    print("="*50)
    print("Starting server on http://0.0.0.0:8000")
    print("Accessible from anywhere on the internet")
    print("="*50)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")