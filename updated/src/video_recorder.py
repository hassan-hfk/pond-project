# video_recorder.py - Complete JPEG Sequence with WebM Conversion
import cv2
import os
import time
import threading
import queue
import numpy as np
from collections import deque
from datetime import datetime
import subprocess
import glob
from pathlib import Path

class VideoRecorder:
    """
    Ultra-fast JPEG sequence recorder with automatic WebM conversion.
    
    Main thread: Saves frames as JPEG (5-10ms per frame) - NO FPS DROP
    Background thread: Converts JPEG sequences to WebM (slow, but doesn't affect main loop)
    
    Perfect for real-time detection systems on embedded devices.
    """
    
    def __init__(self, output_dir="recordings", fps=20, pre_secs=4, post_secs=4, 
                 frame_size=(640, 640), jpeg_quality=85, auto_convert=True, 
                 cleanup_jpegs=False):
        """
        Initialize JPEG recorder with WebM conversion.
        
        Args:
            output_dir: Directory to save recordings
            fps: Frames per second
            pre_secs: Seconds of pre-buffer (before trigger)
            post_secs: Seconds after trigger
            frame_size: (width, height) tuple
            jpeg_quality: JPEG quality 0-100 (85 recommended)
            auto_convert: Automatically convert to WebM after recording
            cleanup_jpegs: Delete JPEGs after successful WebM conversion
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.fps = fps
        self.pre_frames = int(pre_secs * fps)
        self.post_frames = int(post_secs * fps)
        self.frame_size = frame_size
        self.jpeg_quality = jpeg_quality
        self.auto_convert = auto_convert
        self.cleanup_jpegs = cleanup_jpegs

        # Pre-buffer for frames before trigger
        self.buffer = deque(maxlen=self.pre_frames)
        
        # Recording state
        self.recording = False
        self.post_counter = 0
        self.current_class = None
        self.current_dir = None
        self.current_webm_filename = None
        self.frame_number = 0
        
        # Thread-safe queue for frames to save
        self.save_queue = queue.Queue(maxsize=200)
        self.saver_thread = None
        self.stop_saver_event = threading.Event()
        
        # Thread-safe queue for JPEG directories to convert
        self.conversion_queue = queue.Queue()
        self.converter_thread = threading.Thread(
            target=self._conversion_worker, 
            daemon=True,
            name="WebM-Converter"
        )
        self.converter_thread.start()
        
        # Thread lock for state changes
        self.lock = threading.Lock()
        
        print(f"[RecorderJPEG] Initialized")
        print(f"  - FPS: {fps}, Pre: {pre_secs}s, Post: {post_secs}s")
        print(f"  - JPEG Quality: {jpeg_quality}")
        print(f"  - Auto-convert to WebM: {auto_convert}")
        print(f"  - Cleanup JPEGs: {cleanup_jpegs}")

    def update(self, frame):
        """
        Add frame to recorder (called every frame in main loop).
        
        This is FAST (~5-10ms) and won't slow down your main loop.
        
        Args:
            frame: OpenCV frame (numpy array)
        """
        # Always add to pre-buffer (for pre-recording)
        self.buffer.append(frame.copy())
        
        # If recording, add to save queue
        with self.lock:
            if self.recording:
                try:
                    # Non-blocking queue add - instant operation
                    self.save_queue.put_nowait((frame.copy(), self.frame_number))
                    self.frame_number += 1
                    self.post_counter -= 1
                    
                    # Check if we've recorded enough post-trigger frames
                    if self.post_counter <= 0:
                        self._stop_recording_internal()
                        
                except queue.Full:
                    print("[RecorderJPEG] WARNING: Queue full, dropping frame")

    def trigger(self, class_name):
        """
        Trigger a new recording (NON-BLOCKING - returns immediately).
        
        Args:
            class_name: Name/class of the detection (e.g., 'person', 'heron')
            
        Returns:
            WebM filename that will be created (e.g., 'person_20251005_123456.webm')
            or None if already recording
        """
        with self.lock:
            if self.recording:
                print(f"[RecorderJPEG] Already recording '{self.current_class}', skipping '{class_name}'")
                return None

            # Create directory for this recording
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            dir_name = f"{class_name}_{ts}"
            recording_dir = os.path.join(self.output_dir, dir_name)
            os.makedirs(recording_dir, exist_ok=True)

            # Save metadata file
            metadata_path = os.path.join(recording_dir, 'metadata.txt')
            with open(metadata_path, 'w') as f:
                f.write(f"fps={self.fps}\n")
                f.write(f"class={class_name}\n")
                f.write(f"timestamp={ts}\n")
                f.write(f"size={self.frame_size[0]}x{self.frame_size[1]}\n")

            # Copy pre-buffer to avoid race condition (fast copy, ~5ms)
            prebuffer_frames = list(self.buffer)

            # Start background saver thread with all necessary data
            self.stop_saver_event.clear()
            self.saver_thread = threading.Thread(
                target=self._saver_worker,
                args=(class_name, prebuffer_frames, recording_dir),  # Pass dir to thread
                daemon=True,
                name=f"JPEG-Saver-{class_name}"
            )
            self.saver_thread.start()

            # Update recording state AFTER starting thread
            self.recording = True
            self.post_counter = self.post_frames
            self.current_class = class_name
            self.current_dir = recording_dir  # Store for update() method
            self.frame_number = self.pre_frames
            
            # The WebM filename that will be created after conversion
            self.current_webm_filename = f"{class_name}_{ts}.webm"

            print(f"[RecorderJPEG] ▶ START recording '{class_name}' → {recording_dir}")
            
            return self.current_webm_filename

    # Remove this method - no longer needed
    # Pre-buffer is now saved in the background thread

    def _saver_worker(self, class_name, prebuffer_frames, recording_dir):
        """
        Background thread that saves frames as JPEG files.
        Saves pre-buffer first, then ongoing frames.
        Runs independently, doesn't block main loop.
        
        Args:
            class_name: Detection class name
            prebuffer_frames: List of pre-buffered frames
            recording_dir: Directory path for this recording (prevents race condition)
        """
        saved = 0
        start_time = time.time()
        
        try:
            # Save pre-buffer frames first
            print(f"[RecorderJPEG] Saving {len(prebuffer_frames)} pre-buffer frames...")
            for i, frame in enumerate(prebuffer_frames):
                if frame is None:
                    print(f"[RecorderJPEG] WARNING: Pre-buffer frame {i} is None, skipping")
                    continue
                filename = os.path.join(recording_dir, f"frame_{i:05d}.jpg")
                success = cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                if not success:
                    print(f"[RecorderJPEG] WARNING: Failed to write {filename}")
            
            saved = len(prebuffer_frames)
            print(f"[RecorderJPEG] Pre-buffer saved ({saved} frames)")
            
            # Now save ongoing frames from queue
            while not self.stop_saver_event.is_set() or not self.save_queue.empty():
                try:
                    if not self.save_queue.empty():
                        frame, frame_num = self.save_queue.get(timeout=0.5)
                        
                        if frame is None:
                            print(f"[RecorderJPEG] WARNING: Frame {frame_num} is None, skipping")
                            self.save_queue.task_done()
                            continue
                        
                        filename = os.path.join(recording_dir, f"frame_{frame_num:05d}.jpg")
                        success = cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                        
                        if success:
                            saved += 1
                        else:
                            print(f"[RecorderJPEG] WARNING: Failed to write frame {frame_num}")
                        
                        self.save_queue.task_done()
                    else:
                        time.sleep(0.01)
                        
                except queue.Empty:
                    pass
                except Exception as e:
                    print(f"[RecorderJPEG] Error saving frame: {e}")
                    import traceback
                    traceback.print_exc()
            
        except Exception as e:
            print(f"[RecorderJPEG] Error in saver worker: {e}")
            import traceback
            traceback.print_exc()
        
        duration = time.time() - start_time
        print(f"[RecorderJPEG] ■ STOP '{class_name}' - {saved} frames in {duration:.2f}s")
        
        # Queue for WebM conversion if auto-convert is enabled
        if self.auto_convert and recording_dir:
            print(f"[RecorderJPEG] Queuing for WebM conversion: {recording_dir}")
            self.conversion_queue.put(recording_dir)
        else:
            print(f"[RecorderJPEG] Auto-convert disabled, JPEGs saved to: {recording_dir}")

    def _conversion_worker(self):
        """
        Background worker that converts JPEG sequences to WebM.
        Runs continuously in a separate thread, processes conversion queue.
        This is SLOW but doesn't affect main loop performance.
        """
        print("[Converter] WebM conversion worker started")
        
        while True:
            try:
                # Wait for a directory to convert
                jpeg_dir = self.conversion_queue.get(timeout=1.0)
                
                if not os.path.exists(jpeg_dir):
                    print(f"[Converter] ERROR: Directory doesn't exist: {jpeg_dir}")
                    self.conversion_queue.task_done()
                    continue
                
                # Check if there are JPEG files
                jpeg_files = glob.glob(os.path.join(jpeg_dir, 'frame_*.jpg'))
                if not jpeg_files:
                    print(f"[Converter] ERROR: No JPEG files found in {jpeg_dir}")
                    self.conversion_queue.task_done()
                    continue
                
                print(f"[Converter] 🔄 Starting conversion: {os.path.basename(jpeg_dir)}")
                print(f"[Converter] Found {len(jpeg_files)} JPEG files")
                start_time = time.time()
                
                # Convert JPEG sequence to WebM
                success, webm_path = self._convert_to_webm(jpeg_dir)
                
                duration = time.time() - start_time
                
                if success:
                    file_size = os.path.getsize(webm_path) / (1024 * 1024)  # MB
                    print(f"[Converter] ✅ Created WebM in {duration:.1f}s: {os.path.basename(webm_path)} ({file_size:.2f} MB)")
                    
                    # Optional: Delete JPEG files after successful conversion
                    if self.cleanup_jpegs:
                        self._cleanup_jpegs(jpeg_dir)
                else:
                    print(f"[Converter] ❌ Failed to convert: {os.path.basename(jpeg_dir)}")
                
                self.conversion_queue.task_done()
                
            except queue.Empty:
                # No conversions pending, continue waiting
                continue
            except Exception as e:
                print(f"[Converter] Error in conversion worker: {e}")
                import traceback
                traceback.print_exc()

    def _convert_to_webm(self, jpeg_dir):
        """
        Convert JPEG sequence to WebM video.
        Tries ffmpeg first (fast), falls back to OpenCV.
        
        Returns:
            (success: bool, output_path: str or None)
        """
        try:
            # Read metadata
            metadata_path = os.path.join(jpeg_dir, 'metadata.txt')
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    for line in f:
                        if '=' in line:
                            key, value = line.strip().split('=', 1)
                            metadata[key] = value
            
            fps = int(metadata.get('fps', self.fps))
            class_name = metadata.get('class', 'unknown')
            timestamp = metadata.get('timestamp', 'unknown')
            
            # Output WebM path (in main recordings directory)
            webm_filename = f"{class_name}_{timestamp}.webm"
            webm_path = os.path.join(self.output_dir, webm_filename)
            
            # Try ffmpeg first (best quality and performance)
            if self._has_ffmpeg():
                success = self._convert_with_ffmpeg(jpeg_dir, webm_path, fps)
                if success:
                    return True, webm_path
                else:
                    print("[Converter] ffmpeg failed, trying OpenCV...")
            else:
                print("[Converter] ffmpeg not found, using OpenCV (slower)")
            
            # Fallback to OpenCV
            success = self._convert_with_opencv(jpeg_dir, webm_path, fps)
            return success, webm_path if success else None
            
        except Exception as e:
            print(f"[Converter] Exception during conversion: {e}")
            import traceback
            traceback.print_exc()
            return False, None

    def _has_ffmpeg(self):
        """Check if ffmpeg is installed and available"""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2
            )
            return result.returncode == 0
        except:
            return False

    def _convert_with_ffmpeg(self, jpeg_dir, output_path, fps):
        """
        Convert using ffmpeg (recommended - fast and high quality).
        
        VP9 codec settings optimized for speed while maintaining quality.
        """
        try:
            # Input pattern for JPEG sequence
            input_pattern = os.path.join(jpeg_dir, 'frame_%05d.jpg')
            
            # ffmpeg command for WebM with VP9 codec
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file
                '-framerate', str(fps),
                '-i', input_pattern,
                '-c:v', 'libvpx-vp9',  # VP9 codec (WebM)
                '-b:v', '1M',  # Target bitrate
                '-crf', '31',  # Quality (23-35, lower=better, 31 is good balance)
                '-deadline', 'realtime',  # Fast encoding mode
                '-cpu-used', '4',  # Speed preset (0-5, higher=faster)
                '-row-mt', '1',  # Enable multi-threading
                '-threads', '4',  # Use 4 threads
                '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
                output_path
            ]
            
            # Run ffmpeg
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,  # 2 minute timeout
                text=True
            )
            
            # Check if conversion was successful
            if result.returncode == 0 and os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                if file_size > 1000:  # At least 1KB
                    return True
                else:
                    print(f"[Converter] Output file too small: {file_size} bytes")
                    return False
            else:
                if result.stderr:
                    print(f"[Converter] ffmpeg stderr: {result.stderr[-500:]}")  # Last 500 chars
                return False
                
        except subprocess.TimeoutExpired:
            print("[Converter] ffmpeg timeout (>120s)")
            return False
        except Exception as e:
            print(f"[Converter] ffmpeg exception: {e}")
            return False

    def _convert_with_opencv(self, jpeg_dir, output_path, fps):
        """
        Convert using OpenCV (fallback if ffmpeg not available).
        Slower than ffmpeg but works without external dependencies.
        """
        try:
            # Get all JPEG files sorted
            jpeg_files = sorted(glob.glob(os.path.join(jpeg_dir, 'frame_*.jpg')))
            
            if not jpeg_files:
                print("[Converter] No JPEG files found")
                return False
            
            print(f"[Converter] Converting {len(jpeg_files)} frames with OpenCV...")
            
            # Read first frame to get dimensions
            first_frame = cv2.imread(jpeg_files[0])
            if first_frame is None:
                print("[Converter] Failed to read first frame")
                return False
            
            h, w = first_frame.shape[:2]
            
            # Create VideoWriter with VP90 codec (WebM)
            fourcc = cv2.VideoWriter_fourcc(*'VP90')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            
            if not writer.isOpened():
                print("[Converter] Failed to open VideoWriter with VP90")
                return False
            
            # Write all frames
            frames_written = 0
            for jpeg_path in jpeg_files:
                frame = cv2.imread(jpeg_path)
                if frame is not None:
                    writer.write(frame)
                    frames_written += 1
            
            writer.release()
            
            print(f"[Converter] Wrote {frames_written} frames")
            
            # Verify output file
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                if file_size > 1000:  # At least 1KB
                    return True
                else:
                    print(f"[Converter] Output file too small: {file_size} bytes")
                    return False
            else:
                print("[Converter] Output file not created")
                return False
                
        except Exception as e:
            print(f"[Converter] OpenCV conversion error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _cleanup_jpegs(self, jpeg_dir):
        """
        Delete JPEG files and directory after successful WebM conversion.
        Only called if cleanup_jpegs=True in constructor.
        """
        try:
            # Delete all JPEG frames
            jpeg_files = glob.glob(os.path.join(jpeg_dir, 'frame_*.jpg'))
            for f in jpeg_files:
                os.remove(f)
            
            # Delete metadata file
            metadata_path = os.path.join(jpeg_dir, 'metadata.txt')
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            
            # Remove directory if empty
            if not os.listdir(jpeg_dir):
                os.rmdir(jpeg_dir)
            
            print(f"[Converter] 🗑️  Cleaned up: {os.path.basename(jpeg_dir)}")
            
        except Exception as e:
            print(f"[Converter] Cleanup error: {e}")

    def _stop_recording_internal(self):
        """
        Internal method to stop recording.
        Must be called with self.lock held.
        """
        if not self.recording:
            return
        
        self.recording = False
        self.stop_saver_event.set()
        
        print(f"[RecorderJPEG] Signaled stop for '{self.current_class}'")
        
        self.current_class = None
        self.current_dir = None
        self.current_webm_filename = None
        self.frame_number = 0

    def stop(self):
        """
        Stop recording (if active) and cleanup.
        Call this when shutting down your application.
        """
        with self.lock:
            if self.recording:
                self._stop_recording_internal()
        
        # Wait for saver thread to finish
        if self.saver_thread and self.saver_thread.is_alive():
            print("[RecorderJPEG] Waiting for saver thread...")
            self.saver_thread.join(timeout=5.0)
        
        # Clear save queue
        while not self.save_queue.empty():
            try:
                self.save_queue.get_nowait()
            except queue.Empty:
                break
        
        print("[RecorderJPEG] Stopped")

    def is_recording(self):
        """Check if currently recording"""
        with self.lock:
            return self.recording

    def get_queue_size(self):
        """Get number of frames waiting to be saved as JPEG"""
        return self.save_queue.qsize()
    
    def get_conversion_queue_size(self):
        """Get number of recordings waiting to be converted to WebM"""
        return self.conversion_queue.qsize()
    
    def get_status(self):
        """
        Get recorder status for monitoring/debugging.
        
        Returns:
            dict with status information
        """
        with self.lock:
            return {
                'recording': self.recording,
                'current_class': self.current_class,
                'save_queue_size': self.save_queue.qsize(),
                'conversion_queue_size': self.conversion_queue.qsize(),
                'post_counter': self.post_counter if self.recording else 0
            }


# Standalone utility function
def convert_jpeg_directory_to_webm(jpeg_dir, output_path=None, fps=20, cleanup=False):
    """
    Standalone function to manually convert a JPEG directory to WebM.
    Useful for batch conversions or if you need to re-convert.
    
    Args:
        jpeg_dir: Path to directory containing frame_*.jpg files
        output_path: Output WebM path (auto-generated if None)
        fps: Frame rate
        cleanup: Delete JPEGs after conversion
        
    Returns:
        (success: bool, output_path: str or None)
    """
    # Create temporary recorder instance
    recorder = VideoRecorder()
    
    # Generate output path if not provided
    if output_path is None:
        dir_name = os.path.basename(jpeg_dir.rstrip('/\\'))
        output_path = os.path.join(os.path.dirname(jpeg_dir), f"{dir_name}.webm")
    
    # Convert
    success, path = recorder._convert_to_webm(jpeg_dir)
    
    # Cleanup if requested and successful
    if success and cleanup:
        recorder._cleanup_jpegs(jpeg_dir)
    
    return success, path


# Example usage and testing
if __name__ == "__main__":
    print("VideoRecorder - Test Mode")
    print("-" * 50)
    
    # Create test recorder
    recorder = VideoRecorder(
        output_dir="test_recordings",
        fps=20,
        pre_secs=2,
        post_secs=2,
        frame_size=(640, 480),
        jpeg_quality=85,
        auto_convert=True,
        cleanup_jpegs=False
    )
    
    # Simulate recording
    print("\nSimulating 3 second recording...")
    recorder.trigger("test")
    
    # Generate test frames
    for i in range(60):  # 3 seconds at 20 FPS
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"Frame {i}", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        recorder.update(frame)
        time.sleep(0.05)  # Simulate 20 FPS
    
    # Wait for conversion
    print("\nWaiting for WebM conversion...")
    time.sleep(5)
    
    status = recorder.get_status()
    print(f"\nStatus: {status}")
    
    print("\nTest complete! Check test_recordings/ directory")