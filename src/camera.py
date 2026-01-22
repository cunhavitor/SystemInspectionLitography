import cv2
import json
import os

try:
    from picamera2 import Picamera2
    import numpy as np
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("picamera2 not available, falling back to OpenCV")

class Camera:
    def __init__(self, camera_index=0, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.use_picamera2 = PICAMERA2_AVAILABLE
        
        if self.use_picamera2:
            print(f"Using Picamera2 backend. Configured High Res: {width}x{height}")
            try:
                self.picam = Picamera2()
                # Dual Stream Configuration:
                # 'main': High Resolution (for capture/focus)
                # 'lo': Low Resolution (640x480 or similar) for UI Preview (Fast, Low CPU)
                
                # Step 1: Create base config for Main stream
                config = self.picam.create_video_configuration(
                    main={"size": (width, height), "format": "RGB888"}
                )
                
                # Step 2: Manually add Low Res 'lo' stream structure
                # The corrected key for dual stream is 'lores'
                config["lores"] = {
                    "size": (640, 480), 
                    "format": "RGB888",
                    "preserve_ar": True
                }
                
                self.picam.configure(config)
                
                # Set initial controls with auto-exposure enabled
                self.picam.set_controls({
                    "AeEnable": False,  # Enable auto-exposure
                    "AwbEnable": False,  # Enable auto white balance
                    "Brightness": 0.0,
                    "Contrast": 1.0
                })
                
                self.picam.start()
                
                # Give camera time to settle
                import time
                time.sleep(0.5)
                
                print("Picamera2 initialized with Dual Streams (main + lores)")
            except RuntimeError as e:
                if "Device or resource busy" in str(e):
                    raise RuntimeError("Camera is already in use by another process. Please close all camera applications and try again.")
                raise
        else:
            print("Using OpenCV backend (Dual stream simulation not available)")
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                raise ValueError(f"Could not open camera at index {camera_index}")
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)

    def get_frame(self, stream_name='lo'):
        """
        Capture a frame.
        stream_name: 'lo' (default, fast) or 'main' (high res)
        """
        if self.use_picamera2:
            try:
                # Map 'lo' to picamera2's internal 'lores' name
                actual_stream = 'lores' if stream_name == 'lo' else stream_name
                
                # Robustness: Try capture, fallback if stream key doesn't exist
                try:
                    frame = self.picam.capture_array(actual_stream)
                except KeyError:
                    if actual_stream == 'lores':
                        print("Warning: 'lores' stream missing, falling back to 'main'")
                        frame = self.picam.capture_array('main')
                    else:
                        raise
                
                # Ensure frame is in RGB format
                if len(frame.shape) == 2:
                    # Grayscale, convert to RGB
                    import numpy as np
                    frame = np.stack([frame, frame, frame], axis=-1)
                return frame
            except Exception as e:
                print(f"Error capturing from stream '{stream_name}': {e}")
                return None
        else:
            # OpenCV fallback - stream_name ignored, always main stream
            ret, frame = self.cap.read()
            if not ret:
                return None
            return frame

    def release(self):
        if self.use_picamera2 and hasattr(self, 'picam') and self.picam:
            try:
                self.picam.stop()
                self.picam.close() # Explicitly release resources
                print("Picamera2 closed successfully")
            except Exception as e:
                print(f"Error closing Picamera2: {e}")
            finally:
                self.picam = None
        else:
            if self.cap.isOpened():
                self.cap.release()

    def set_property(self, prop_id, value):
        if self.use_picamera2:
            # Map OpenCV properties to picamera2 controls
            controls = {}
            
            if prop_id == cv2.CAP_PROP_EXPOSURE:
                # Exposure Slider:
                # 0 = Auto Exposure ON
                # >0 = Manual Exposure (100us to 33000us range is reasonable for 30fps)
                if value == 0:
                    controls["AeEnable"] = True
                    print("Setting Auto-Exposure: ON")
                else:
                    exposure_us = int(100 + (value / 255.0) * 33000)
                    controls["AeEnable"] = False
                    controls["ExposureTime"] = exposure_us
            elif prop_id == cv2.CAP_PROP_BRIGHTNESS:
                # Brightness: -1.0 to 1.0
                controls["Brightness"] = (value / 255.0) * 2.0 - 1.0
            elif prop_id == cv2.CAP_PROP_CONTRAST:
                # Contrast: 0.0 to 2.0
                controls["Contrast"] = (value / 255.0) * 2.0
            elif prop_id == cv2.CAP_PROP_FOCUS:
                # AnalogueGain: typically 1.0 to 16.0
                controls["AnalogueGain"] = 1.0 + (value / 255.0) * 15.0
            
            if controls:
                try:
                    self.picam.set_controls(controls)
                    print(f"Set picamera2 controls: {controls}")
                    return True
                except Exception as e:
                    print(f"Error setting controls: {e}")
                    return False
        else:
            success = self.cap.set(prop_id, value)
            actual_value = self.cap.get(prop_id)
            print(f"Set property {prop_id} to {value}, success={success}, actual={actual_value}")
            return success
        return False

    def get_property(self, prop_id):
        if self.use_picamera2:
            # Getting properties from picamera2 is more complex
            # For now, return stored values or defaults
            metadata = self.picam.capture_metadata()
            
            if prop_id == cv2.CAP_PROP_EXPOSURE:
                return metadata.get("ExposureTime", 0) / 1000  # μs to ms
            elif prop_id == cv2.CAP_PROP_BRIGHTNESS:
                return metadata.get("Brightness", 0) * 255
            elif prop_id == cv2.CAP_PROP_CONTRAST:
                return metadata.get("Contrast", 1.0) * 255
            elif prop_id == cv2.CAP_PROP_FOCUS:
                return metadata.get("AnalogueGain", 1.0) * 25.5
            return 0.0
        else:
            if self.cap.isOpened():
                return self.cap.get(prop_id)
            return 0.0

    def get_current_parameters(self):
        """Get current camera parameter values"""
        return {
            'exposure': self.get_property(cv2.CAP_PROP_EXPOSURE),
            'brightness': self.get_property(cv2.CAP_PROP_BRIGHTNESS),
            'contrast': self.get_property(cv2.CAP_PROP_CONTRAST),
            'focus': self.get_property(cv2.CAP_PROP_FOCUS)
        }

    def save_parameters(self, filepath='config/camera_params.json'):
        """Save current camera parameters to JSON file"""
        # Ensure absolute path to avoid CWD issues
        if not os.path.isabs(filepath):
            # src/camera.py -> src -> project_root (2 levels up)
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            filepath = os.path.join(base_dir, filepath)

        params = self.get_current_parameters()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=4)
        print(f"Camera parameters saved to {filepath}")
        return True

    def load_parameters(self, filepath='config/camera_params.json'):
        """Load camera parameters from JSON file and apply them"""
        # Ensure absolute path
        if not os.path.isabs(filepath):
            # src/camera.py -> src -> project_root (2 levels up)
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            filepath = os.path.join(base_dir, filepath)

        if not os.path.exists(filepath):
            print(f"No saved parameters found at {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                params = json.load(f)
            
            self.set_property(cv2.CAP_PROP_EXPOSURE, params.get('exposure', 0))
            self.set_property(cv2.CAP_PROP_BRIGHTNESS, params.get('brightness', 0))
            self.set_property(cv2.CAP_PROP_CONTRAST, params.get('contrast', 0))
            self.set_property(cv2.CAP_PROP_FOCUS, params.get('focus', 0))
            
            print(f"Camera parameters loaded from {filepath}")
            return params
        except Exception as e:
            print(f"Error loading parameters: {e}")
            return False
    def warmup(self, num_frames=30, callback=None):
        """Warmup camera by discarding initial frames to let AE/AWB settle"""
        print(f"Warming up camera ({num_frames} frames)...")
        for i in range(num_frames):
            self.get_frame()
            if callback:
                callback(i + 1, num_frames)
        print("Camera warmup complete.")
