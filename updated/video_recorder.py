# video_recorder.py
import cv2
import os
import time
import threading
from collections import deque
from datetime import datetime

class VideoRecorder:
    def __init__(self, output_dir="recordings", fps=20, pre_secs=4, post_secs=4, frame_size=(640, 640)):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.fps = fps
        self.pre_frames = int(pre_secs * fps)
        self.post_frames = int(post_secs * fps)
        self.frame_size = frame_size

        self.buffer = deque(maxlen=self.pre_frames)
        self.recording = False
        self.post_counter = 0
        self.writer = None
        self.lock = threading.Lock()
        self.current_class = None

    def update(self, frame):
        """Add frame to buffer, and if recording, also write it."""
        with self.lock:
            self.buffer.append(frame.copy())

            if self.recording and self.writer is not None:
                self.writer.write(frame)
                self.post_counter -= 1
                if self.post_counter <= 0:
                    self._stop_writer()

    def trigger(self, class_name):
        """Trigger saving video with pre + post frames."""
        with self.lock:
            if self.recording:  # already recording
                print(f"[VideoRecorder] Already recording for {self.current_class}, skipping new trigger.")
                return None

            # filename
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{class_name}_{ts}.mp4"
            path = os.path.join(self.output_dir, filename)

            # video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(path, fourcc, self.fps, self.frame_size)

            if not self.writer.isOpened():
                print(f"❌ Failed to open VideoWriter for {path}")
                return

            # dump pre-buffer first
            for f in list(self.buffer):
                self.writer.write(f)

            # enable post capture
            self.recording = True
            self.post_counter = self.post_frames
            self.current_class = class_name
            print(f"[VideoRecorder] ▶ START recording for '{class_name}' → {path}")
            return filename

    def _stop_writer(self):
        if self.writer is not None:
            self.writer.release()
            print(f"[VideoRecorder] ■ STOP recording for '{self.current_class}'")
            self.writer = None

        self.recording = False
        self.current_class = None
