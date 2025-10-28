from fastapi import FastAPI
import requests

app = FastAPI()

LOCAL_API = "http://127.0.0.1:5000"

@app.get("/api/feedback/{event_id}/{status}")
@app.get("/feedback/{event_id}/{status}")  # handle both
def forward_feedback(event_id: int, status: str):
    """Forward feedback to local Flask API"""
    try:
        r = requests.get(f"{LOCAL_API}/api/feedback/{event_id}/{status}")
        return r.text
    except Exception as e:
        return {"error": str(e)}
