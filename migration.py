#!/usr/bin/env python3
"""
Automated Project Migration Script
Restructures the child detection system to the new directory layout
"""

import os
import shutil
from pathlib import Path
import sys

class ProjectMigration:
    def __init__(self):
        self.root = Path.cwd()
        self.errors = []
        self.warnings = []
        self.moved_files = []
        
    def log(self, message, type="info"):
        """Log messages with color coding"""
        colors = {
            "info": "\033[94m",      # Blue
            "success": "\033[92m",   # Green
            "warning": "\033[93m",   # Yellow
            "error": "\033[91m",     # Red
            "reset": "\033[0m"
        }
        print(f"{colors.get(type, colors['info'])}{message}{colors['reset']}")
    
    def create_directories(self):
        """Create the new directory structure"""
        self.log("\n📁 Creating directory structure...", "info")
        
        directories = [
            'src',
            'models',
            'config',
            'data/recordings',
            'data/thumbnails',
            'web/templates',
            'web/static/videos',
            'web/static/thumbnails',
            'logs',
            'scripts',
            'tests'
        ]
        
        for directory in directories:
            path = self.root / directory
            path.mkdir(parents=True, exist_ok=True)
            self.log(f"  ✓ Created: {directory}", "success")
    
    def backup_project(self):
        """Create a backup of the current project"""
        self.log("\n💾 Creating backup...", "info")
        backup_dir = self.root / "backup_before_migration"
        
        if backup_dir.exists():
            self.log("  ! Backup already exists, skipping", "warning")
            return
        
        try:
            # Files to backup
            important_files = [
                'config.yaml',
                'detections.db',
                'events.log',
                'best.pt'
            ]
            
            backup_dir.mkdir(exist_ok=True)
            
            for file in important_files:
                src = self.root / file
                if src.exists():
                    shutil.copy2(src, backup_dir / file)
                    self.log(f"  ✓ Backed up: {file}", "success")
            
            # Backup recordings if they exist
            recordings = self.root / "recordings"
            if recordings.exists():
                shutil.copytree(recordings, backup_dir / "recordings", dirs_exist_ok=True)
                self.log(f"  ✓ Backed up: recordings/", "success")
                
        except Exception as e:
            self.log(f"  ✗ Backup failed: {e}", "error")
            self.errors.append(f"Backup failed: {e}")
    
    def move_python_files(self):
        """Move Python source files to src/"""
        self.log("\n🐍 Moving Python files to src/...", "info")
        
        python_files = [
            'inference.py',
            'child_logic.py',
            'video_recorder.py',
            'gpio_control.py',
            'db_handler.py',
            'calibrate.py',
            'set_roi.py',
            'mqtt_safe.py'
        ]
        
        for file in python_files:
            src = self.root / file
            dst = self.root / 'src' / file
            
            if src.exists() and not dst.exists():
                shutil.move(str(src), str(dst))
                self.log(f"  ✓ Moved: {file} → src/", "success")
                self.moved_files.append(file)
            elif dst.exists():
                self.log(f"  ! Already exists: src/{file}", "warning")
            else:
                self.log(f"  ✗ Not found: {file}", "error")
                self.errors.append(f"File not found: {file}")
    
    def move_script_files(self):
        """Move script files to scripts/"""
        self.log("\n📜 Moving scripts...", "info")
        
        # Move run_child_monitor.py if it exists in root
        src = self.root / 'run_child_monitor.py'
        dst = self.root / 'scripts' / 'run_child_monitor.py'
        
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
            self.log(f"  ✓ Moved: run_child_monitor.py → scripts/", "success")
        elif dst.exists():
            self.log(f"  ! Already exists: scripts/run_child_monitor.py", "warning")
        else:
            self.log(f"  ℹ run_child_monitor.py not in root (may be in scripts already)", "info")
    
    def move_app_file(self):
        """Move app.py to src/"""
        self.log("\n🌐 Moving Flask app...", "info")
        
        src = self.root / 'app.py'
        dst = self.root / 'src' / 'app.py'
        
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
            self.log(f"  ✓ Moved: app.py → src/", "success")
        elif dst.exists():
            self.log(f"  ! Already exists: src/app.py", "warning")
        else:
            self.log(f"  ✗ Not found: app.py", "error")
    
    def move_templates(self):
        """Move HTML templates to web/templates/"""
        self.log("\n📄 Moving HTML templates...", "info")
        
        html_files = [
            'index.html',
            'roi_editor.html',
            'config_editor.html'
        ]
        
        for file in html_files:
            src = self.root / file
            dst = self.root / 'web' / 'templates' / file
            
            if src.exists() and not dst.exists():
                shutil.move(str(src), str(dst))
                self.log(f"  ✓ Moved: {file} → web/templates/", "success")
            elif dst.exists():
                self.log(f"  ! Already exists: web/templates/{file}", "warning")
            else:
                self.log(f"  ✗ Not found: {file}", "error")
    
    def move_config(self):
        """Move config.yaml to config/"""
        self.log("\n⚙️  Moving configuration...", "info")
        
        src = self.root / 'config.yaml'
        dst = self.root / 'config' / 'config.yaml'
        
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
            self.log(f"  ✓ Moved: config.yaml → config/", "success")
        elif dst.exists():
            self.log(f"  ! Already exists: config/config.yaml", "warning")
        else:
            self.log(f"  ✗ Not found: config.yaml", "error")
    
    def move_model(self):
        """Move model file to models/"""
        self.log("\n🤖 Moving YOLO model...", "info")
        
        src = self.root / 'best.pt'
        dst = self.root / 'models' / 'best.pt'
        
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
            self.log(f"  ✓ Moved: best.pt → models/", "success")
        elif dst.exists():
            self.log(f"  ! Already exists: models/best.pt", "warning")
        else:
            self.log(f"  ✗ Not found: best.pt", "error")
            self.warnings.append("Model file not found - you'll need to add it manually")
    
    def move_data_files(self):
        """Move data files to data/"""
        self.log("\n💾 Moving data files...", "info")
        
        # Move database
        db_src = self.root / 'detections.db'
        db_dst = self.root / 'data' / 'detections.db'
        
        if db_src.exists() and not db_dst.exists():
            shutil.move(str(db_src), str(db_dst))
            self.log(f"  ✓ Moved: detections.db → data/", "success")
        elif db_dst.exists():
            self.log(f"  ! Already exists: data/detections.db", "warning")
        
        # Move recordings folder
        rec_src = self.root / 'recordings'
        rec_dst = self.root / 'data' / 'recordings'
        
        if rec_src.exists():
            for file in rec_src.glob('*'):
                if file.is_file():
                    dst_file = rec_dst / file.name
                    if not dst_file.exists():
                        shutil.move(str(file), str(dst_file))
            self.log(f"  ✓ Moved recordings → data/recordings/", "success")
            
            # Remove old recordings directory if empty
            if not any(rec_src.iterdir()):
                rec_src.rmdir()
        
        # Move log file
        log_src = self.root / 'events.log'
        log_dst = self.root / 'logs' / 'events.log'
        
        if log_src.exists() and not log_dst.exists():
            shutil.move(str(log_src), str(log_dst))
            self.log(f"  ✓ Moved: events.log → logs/", "success")
    
    def create_requirements(self):
        """Create requirements.txt if it doesn't exist"""
        self.log("\n📦 Creating requirements.txt...", "info")
        
        req_file = self.root / 'requirements.txt'
        
        if not req_file.exists():
            requirements = """flask==2.3.0
opencv-python==4.8.0.76
numpy==1.24.3
pyyaml==6.0
ultralytics==8.0.134
shapely==2.0.1
paho-mqtt==1.6.1
torch==2.0.1
torchvision==0.15.2
"""
            req_file.write_text(requirements)
            self.log(f"  ✓ Created requirements.txt", "success")
        else:
            self.log(f"  ! requirements.txt already exists", "warning")
    
    def create_gitignore(self):
        """Create .gitignore if it doesn't exist"""
        self.log("\n🚫 Creating .gitignore...", "info")
        
        gitignore = self.root / '.gitignore'
        
        if not gitignore.exists():
            content = """# Python
__pycache__/
*.py[cod]
*.so
.Python
env/
venv/
*.egg-info/

# Data files
data/recordings/*.mp4
data/recordings/*.avi
data/recordings/*.webm
data/*.db
logs/*.log

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Backup
backup_before_migration/
"""
            gitignore.write_text(content)
            self.log(f"  ✓ Created .gitignore", "success")
        else:
            self.log(f"  ! .gitignore already exists", "warning")
    
    def create_readme(self):
        """Create README.md if it doesn't exist"""
        self.log("\n📖 Creating README.md...", "info")
        
        readme = self.root / 'README.md'
        
        if not readme.exists():
            content = """# Child Detection System

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
"""
            readme.write_text(content)
            self.log(f"  ✓ Created README.md", "success")
        else:
            self.log(f"  ! README.md already exists", "warning")
    
    def print_summary(self):
        """Print migration summary"""
        self.log("\n" + "="*60, "info")
        self.log("📊 MIGRATION SUMMARY", "info")
        self.log("="*60, "info")
        
        if self.moved_files:
            self.log(f"\n✓ Successfully moved {len(self.moved_files)} files", "success")
        
        if self.warnings:
            self.log(f"\n⚠️  {len(self.warnings)} Warning(s):", "warning")
            for warning in self.warnings:
                self.log(f"  • {warning}", "warning")
        
        if self.errors:
            self.log(f"\n✗ {len(self.errors)} Error(s):", "error")
            for error in self.errors:
                self.log(f"  • {error}", "error")
        
        self.log("\n📝 NEXT STEPS:", "info")
        self.log("  1. Update Python files with new import paths (see documentation)", "info")
        self.log("  2. Update config/config.yaml model path to: ../models/best.pt", "info")
        self.log("  3. Test the application: python src/app.py", "info")
        self.log("  4. Verify all features work correctly", "info")
        
        self.log("\n" + "="*60 + "\n", "info")
    
    def run(self):
        """Execute the full migration"""
        self.log("\n🚀 Starting Project Migration", "success")
        self.log("="*60 + "\n", "info")
        
        try:
            self.backup_project()
            self.create_directories()
            self.move_python_files()
            self.move_script_files()
            self.move_app_file()
            self.move_templates()
            self.move_config()
            self.move_model()
            self.move_data_files()
            self.create_requirements()
            self.create_gitignore()
            self.create_readme()
            self.print_summary()
            
            return len(self.errors) == 0
            
        except Exception as e:
            self.log(f"\n✗ Migration failed with error: {e}", "error")
            return False

def main():
    print("""
╔════════════════════════════════════════════════════════════╗
║        Child Detection System - Project Migration         ║
║                                                            ║
║  This script will reorganize your project to the new      ║
║  directory structure. A backup will be created first.     ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    response = input("Continue with migration? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("\n❌ Migration cancelled by user.")
        sys.exit(0)
    
    migration = ProjectMigration()
    success = migration.run()
    
    if success:
        print("\n✅ Migration completed successfully!")
        sys.exit(0)
    else:
        print("\n⚠️  Migration completed with errors. Please review the output above.")
        sys.exit(1)

if __name__ == '__main__':
    main()