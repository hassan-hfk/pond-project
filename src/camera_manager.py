"""
Singleton Camera Manager to prevent multiple Picamera2 instances
Ensures only one camera object exists across the entire application

Configuration:
- Set FORCE_OPENCV = True to always use OpenCV (cv2.VideoCapture)
- Set FORCE_OPENCV = False to prefer Picamera2 (falls back to OpenCV if unavailable)
"""
import cv2
import threading
from pathlib import Path
import yaml

# ============================================
# CONFIGURATION: Change this to switch camera backend
# ============================================
FORCE_OPENCV = False  # Set to True to always use cv2.VideoCapture
# ============================================

# Try to import Picamera2, fall back to regular camera if not available
try:
    from picamera2 import Picamera2
    from libcamera import Transform
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    print("[CameraManager] Picamera2 not available")

class CameraManager:
    """Singleton class to manage camera access"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.camera = None
        self.camera_lock = threading.Lock()
        self.reference_count = 0
        
        # Load config
        project_root = Path(__file__).parent.parent
        config_path = project_root / 'config' / 'config.yaml'
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        self.width = int(self.cfg['camera']['width'])
        self.height = int(self.cfg['camera']['height'])
        self.device = int(self.cfg['camera'].get('device', 0))
        self.fps = int(self.cfg['camera'].get('fps', 25))  # ✅ READ FPS from config
        
        # Check if backend is specified in config (overrides FORCE_OPENCV)
        config_backend = self.cfg.get('camera', {}).get('backend', 'auto').lower()
        
        if config_backend == 'opencv':
            self.use_picamera = False
            print("[CameraManager] Config specifies 'opencv' backend")
        elif config_backend == 'picamera2':
            if PICAMERA_AVAILABLE:
                self.use_picamera = True
                print("[CameraManager] Config specifies 'picamera2' backend")
            else:
                self.use_picamera = False
                print("[CameraManager] Config specifies 'picamera2' but not available, using OpenCV")
        else:  # 'auto' or not specified
            # Use FORCE_OPENCV setting
            if FORCE_OPENCV:
                self.use_picamera = False
                print("[CameraManager] FORCE_OPENCV=True, using OpenCV backend")
            else:
                self.use_picamera = PICAMERA_AVAILABLE
                if not PICAMERA_AVAILABLE:
                    print("[CameraManager] Picamera2 not available, using OpenCV backend")
        
        backend = "Picamera2" if self.use_picamera else "OpenCV"
        print(f"[CameraManager] Active backend: {backend}")
    
    def get_camera(self):
        """Get camera instance (creates if not exists)"""
        with self.camera_lock:
            if self.camera is None:
                self._create_camera()
            self.reference_count += 1
            return self.camera
    
    def release_camera(self):
        """Decrement reference count (but keep camera alive)"""
        with self.camera_lock:
            self.reference_count = max(0, self.reference_count - 1)
            # Don't actually release camera - keep it alive for next use
            print(f"[CameraManager] Reference count: {self.reference_count}")
    
    def _create_camera(self):
        """Internal method to create camera"""
        if self.use_picamera:
            try:
                print("[CameraManager] Creating Picamera2 instance...")
                self.camera = Picamera2()
                camera_config = self.camera.create_preview_configuration(
                    main={"size": (2460, 2460)}
                )
                camera_config["transform"] = Transform(vflip=1)
                self.camera.configure(camera_config)
                self.camera.start()
                print("[CameraManager] Picamera2 started successfully")
            except Exception as e:
                print(f"[CameraManager] Picamera2 failed: {e}, falling back to OpenCV")
                self.use_picamera = False
                self._create_opencv_camera()
        else:
            self._create_opencv_camera()
    
    def _create_opencv_camera(self):
        """Create OpenCV camera fallback"""
        print(f"[CameraManager] Creating OpenCV camera (device {self.device})...")
        self.camera = cv2.VideoCapture(self.device)
        if not self.camera.isOpened():
            raise RuntimeError(f"Failed to open camera device {self.device}")
        print("[CameraManager] OpenCV camera opened successfully")
    
    def capture_frame(self):
        """Capture a frame from camera"""
        with self.camera_lock:
            if self.camera is None:
                raise RuntimeError("Camera not initialized")
            
            if self.use_picamera:
                frame = self.camera.capture_array()
                frame = cv2.resize(frame, (self.width, self.height))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return True, frame
            else:
                ret, frame = self.camera.read()
                if ret:
                    frame = cv2.resize(frame, (self.width, self.height))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return ret, frame
    
    def force_release(self):
        """Force release camera (use with caution)"""
        with self.camera_lock:
            if self.camera is not None:
                print("[CameraManager] Force releasing camera...")
                try:
                    if self.use_picamera:
                        self.camera.stop()
                    else:
                        self.camera.release()
                except Exception as e:
                    print(f"[CameraManager] Error releasing camera: {e}")
                finally:
                    self.camera = None
                    self.reference_count = 0
                print("[CameraManager] Camera released")

# Global instance
camera_manager = CameraManager()
