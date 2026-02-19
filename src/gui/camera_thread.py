from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
import numpy as np
import cv2
from ..utils import calculate_sharpness, calculate_noise_level
import time
import queue

class CameraThread(QThread):
    # Updated Signal: Emits (QImage, sharpness_score, noise_score, raw_frame_resized)
    frame_captured = Signal(QImage, float, float, np.ndarray)
    
    def __init__(self, camera, target_size=(640, 480)):
        super().__init__()
        self.camera = camera
        self.running = True
        self.paused = False
        self.target_size = target_size
        self.use_high_res = False  # Default to low-res stream for preview
        self._last_capture_time = 0
        self._min_frame_interval = 1.0 / 15.0  # limit to 15 FPS
        self.command_queue = queue.Queue()
        
        # Throttling state
        self._frame_count = 0
        self._last_sharpness = 0.0
        self._last_noise = 0.0

    def set_camera_property(self, prop_id, value):
        """Thread-safe way to request a property change"""
        self.command_queue.put(("set_property", prop_id, value))

    def run(self):
        while self.running:
            # 1. Process Pending Commands (With Coalescing)
            # We drain the entire queue first to find the *latest* value for each property.
            # This prevents executing 100 intermediate "Exposure" updates when dragging the slider.
            pending_updates = {}
            
            while not self.command_queue.empty():
                try:
                    cmd_data = self.command_queue.get_nowait()
                    cmd = cmd_data[0]
                    
                    if cmd == "set_property":
                        # cmd_data = (cmd, prop_id, value)
                        _, prop_id, value = cmd_data
                        # Overwrite with latest value
                        pending_updates[prop_id] = value
                except queue.Empty:
                    break
            
            # Apply only the latest unique updates
            if self.camera and pending_updates:
                for prop_id, value in pending_updates.items():
                    self.camera.set_property(prop_id, value)

            # 2. Capture Frame
            now = time.time()
            if not self.paused and self.camera:
                # Frame throttling
                if now - self._last_capture_time >= self._min_frame_interval:
                    try:
                        # Dual Stream Selection
                        stream = 'main' if self.use_high_res else 'lo'
                        frame = self.camera.get_frame(stream_name=stream)
                    except Exception as e:
                        print(f"Frame capture error (camera likely closed): {e}")
                        frame = None
                    
                    if frame is not None:
                        # Optimization 1: Hardware Resizing via Dual Stream
                        # 'lo' stream is already 640x480 (or configured low res).
                        # 'main' stream is full res.
                        # No need for manual CPU cv2.resize!
                        
                        # Optimization 2: Throttle Sharpness (every 3rd frame)
                        self._frame_count += 1
                        if self._frame_count % 3 == 0:
                            score = calculate_sharpness(frame)
                            noise_score = calculate_noise_level(frame)
                            self._last_sharpness = score
                            self._last_noise = noise_score
                        else:
                            score = self._last_sharpness
                            noise_score = self._last_noise

                        # Optimization 3: Convert to QImage in Thread (offload UI)
                        try:
                            # Convert BGR/Gray to RGB
                            if len(frame.shape) == 3:
                                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            else:
                                rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                            
                            h, w, ch = rgb.shape
                            bytes_per_line = ch * w
                            
                            # Create QImage and COPY to detach from numpy buffer
                            # This is critical so we don't depend on 'rgb' staying alive
                            qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
                            
                            # Emit QImage directl + raw frame for analysis/saving
                            self.frame_captured.emit(qt_image, score, noise_score, frame)
                            self._last_capture_time = now
                        except Exception as e:
                            print(f"Image conversion error: {e}")
            
            # Small sleep to prevent CPU hogging
            self.msleep(5)

    def stop(self):
        self.running = False
        self.wait()
