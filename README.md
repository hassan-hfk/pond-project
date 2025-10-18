# Child Detection System

A real-time detection system for monitoring using YOLOv8 and computer vision.

## Installation

1. Install dependencies: `pip install -r requirements.txt`
2. Place your YOLO model in `models/best.pt`
3. Configure `config/config.yaml`
4. Run: `python src/app.py`

## Project Structure

- `src/` - Source code
- `models/` - YOLO model files
- `config/` - Configuration files
- `data/` - Data storage (recordings, database)
- `web/` - Web interface (templates, static files)
- `logs/` - Application logs
- `scripts/` - Utility scripts
- `tests/` - Test files

## Usage

Start the web application:
```bash
python src/app.py
```

Access the interface at: http://127.0.0.1:5000
