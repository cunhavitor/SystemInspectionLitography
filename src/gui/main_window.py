import cv2
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QTabWidget, QStatusBar,
                             QSlider, QGroupBox, QFormLayout, QProgressBar)
from .user_management import UserManagementDialog
from PySide6.QtCore import QTimer, Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from ..camera import Camera
from ..utils import save_image, calculate_sharpness
import numpy as np

class CameraThread(QThread):
    frame_captured = Signal(np.ndarray, float)
    
    def __init__(self, camera, target_size=(640, 480)):
        super().__init__()
        self.camera = camera
        self.running = True
        self.paused = False
        self.target_size = target_size
        self._last_capture_time = 0
        self._min_frame_interval = 1.0 / 24.0  # Limit to 24 FPS for UI smoothness
        
    def run(self):
        import time
        while self.running:
            now = time.time()
            if not self.paused and self.camera:
                # Frame throttling
                if now - self._last_capture_time >= self._min_frame_interval:
                    frame = self.camera.get_frame()
                    if frame is not None:
                        # Resize here to save UI thread work if needed
                        # Currently just passing through, but throttling helps responsiveness
                        
                        # Calculate sharpness in this thread to save UI thread CPU
                        score = calculate_sharpness(frame)
                        self.frame_captured.emit(frame, score)
                        self._last_capture_time = now
            
            # Small sleep to prevent CPU hogging
            self.msleep(5) # 5ms sleep is plenty

    def stop(self):
        self.running = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self, user_manager, config):
        super().__init__()
        self.user_manager = user_manager
        self.config = config
        self.camera = None
        self.camera_thread = None
        
        self.setWindowTitle(f"Inspection Vision - User: {user_manager.current_user.username}")
        self.resize(800, 600)
        
        self._setup_ui()
        self._start_camera()
        
        # Maximize window
        self.showMaximized()

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Header
        header_layout = QHBoxLayout()
        self.user_label = QLabel(f"Logged in as: {self.user_manager.current_user.username} ({self.user_manager.current_user.role})")
        if self.user_manager.current_user.role == "admin":
            manage_users_btn = QPushButton("Manage Users")
            manage_users_btn.clicked.connect(self.open_user_management)
            header_layout.addWidget(manage_users_btn)

        logout_btn = QPushButton("Logout")
        logout_btn.clicked.connect(self.logout)
        header_layout.addWidget(self.user_label)
        header_layout.addStretch()
        header_layout.addWidget(logout_btn)
        layout.addLayout(header_layout)
        
        # Tabs
        self.tabs = QTabWidget()
        
        self.inspection_tab = self._create_inspection_tab()
        self.tabs.addTab(self.inspection_tab, "Inspection Mode")
        
        self.dataset_tab = self._create_dataset_tab()
        self.tabs.addTab(self.dataset_tab, "Dataset Mode")
        
        if self.user_manager.current_user.role in ["admin", "tecnico"]:
            self.adjustment_tab = self._create_adjustment_tab()
            self.tabs.addTab(self.adjustment_tab, "Adjustment Mode")

        layout.addWidget(self.tabs)
        
        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def _create_inspection_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.inspection_video_label = QLabel("Camera Feed")
        self.inspection_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.inspection_video_label.setMinimumSize(640, 480)
        layout.addWidget(self.inspection_video_label)
        
        # Add inspection specific controls here
        btn_layout = QHBoxLayout()
        inspect_btn = QPushButton("Start Inspection") # Placeholder functionality
        btn_layout.addWidget(inspect_btn)
        layout.addLayout(btn_layout)
        
        return widget

    def _create_dataset_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.dataset_video_label = QLabel("Camera Feed")
        self.dataset_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dataset_video_label.setMinimumSize(640, 480)
        layout.addWidget(self.dataset_video_label)
        
        capture_btn = QPushButton("Capture Image")
        capture_btn.clicked.connect(self.capture_image)
        layout.addWidget(capture_btn)
        
        return widget

    def _create_adjustment_tab(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # Left: Video Feed
        self.adjustment_video_label = QLabel("Camera Feed")
        self.adjustment_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.adjustment_video_label.setMinimumSize(640, 480)
        layout.addWidget(self.adjustment_video_label, stretch=2)
        
        # Right: Controls
        controls_panel = QWidget()
        controls_layout = QVBoxLayout(controls_panel)
        
        # Sharpness Meter
        sharpness_group = QGroupBox("Focus Assist (Sharpness)")
        sharpness_layout = QVBoxLayout()
        self.sharpness_bar = QProgressBar()
        self.sharpness_bar.setRange(0, 1000) # Arbitrary range, will adjust
        self.sharpness_bar.setTextVisible(True)
        self.sharpness_value_label = QLabel("Score: 0.0")
        sharpness_layout.addWidget(self.sharpness_bar)
        sharpness_layout.addWidget(self.sharpness_value_label)
        sharpness_group.setLayout(sharpness_layout)
        controls_layout.addWidget(sharpness_group)
        
        # Camera Parameters
        params_group = QGroupBox("Camera Parameters")
        params_layout = QFormLayout()
        
        # Helper to create sliders
        def create_slider(name, prop_id, min_val, max_val, default=128):
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(min_val, max_val)
            # Try to get current value
            if self.camera:
                 # If using picamera2, getting properties back is tricky, better to start with defaults
                 # or tracked values. For now, use the default passed in.
                 current = default
            else:
                 current = default
            
            slider.setValue(int(current))
            slider.valueChanged.connect(lambda v: self.update_camera_param(prop_id, v))
            params_layout.addRow(name, slider)
            return slider

        # Defaults: Exposure=0 (Auto/Low), Brightness=128 (0.0), Contrast=128 (1.0), Focus=128 (Gain ~8x)
        self.exposure_slider = create_slider("Exposure", cv2.CAP_PROP_EXPOSURE, 0, 255, 0)
        self.brightness_slider = create_slider("Brightness", cv2.CAP_PROP_BRIGHTNESS, 0, 255, 128)
        self.contrast_slider = create_slider("Contrast", cv2.CAP_PROP_CONTRAST, 0, 255, 128)
        self.focus_slider = create_slider("Focus", cv2.CAP_PROP_FOCUS, 0, 255, 20) # Low gain default
        
        params_group.setLayout(params_layout)
        controls_layout.addWidget(params_group)
        
        # Save/Load Buttons
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Save Parameters")
        save_btn.clicked.connect(self.save_camera_params)
        btn_layout.addWidget(save_btn)
        
        load_btn = QPushButton("Load Parameters")
        load_btn.clicked.connect(self.load_camera_params)
        btn_layout.addWidget(load_btn)
        controls_layout.addLayout(btn_layout)
        
        controls_layout.addStretch()
        layout.addWidget(controls_panel, stretch=1)
        
        return widget

    def update_camera_param(self, prop_id, value):
        if self.camera:
            self.camera.set_property(prop_id, value)

    def _start_camera(self):
        try:
            self.camera = Camera(
                camera_index=self.config['camera']['index'],
                width=self.config['camera']['width'],
                height=self.config['camera']['height'],
                fps=self.config['camera']['fps']
            )
            
            # Start camera thread
            self.camera_thread = CameraThread(self.camera)
            self.camera_thread.frame_captured.connect(self.update_frame)
            self.camera_thread.start()
            
            # Auto-load saved camera parameters
            self.camera.load_parameters()
        except Exception as e:
            self.status_bar.showMessage(f"Error initializing camera: {str(e)}")

    def update_frame(self, frame, sharpness_score):
        if frame is not None:
                # Handle malformed frame shape - reshape if needed
                if len(frame.shape) == 2 and frame.shape[0] == 1:
                    # Frame is flattened (1, N) - need to reshape to (height, width, channels)
                    width = self.config['camera']['width']
                    height = self.config['camera']['height']
                    total_pixels = width * height
                    
                    # Check if it's grayscale or color based on data size
                    if frame.shape[1] == total_pixels:
                        # Grayscale
                        frame = frame.reshape(height, width)
                    elif frame.shape[1] == total_pixels * 3:
                        # Color
                        frame = frame.reshape(height, width, 3)
                    else:
                        print(f"WARNING: Unexpected frame size: {frame.shape}")
                        return
                
                # Convert to Qt format
                if len(frame.shape) == 2:
                    # Grayscale - convert to RGB for display
                    rgb_image = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                elif frame.shape[2] == 3:
                    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    print(f"WARNING: Unexpected frame channels: {frame.shape}")
                    return
                    
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                
                # Update the label in the current tab
                current_widget = self.tabs.currentWidget()
                
                if current_widget == self.inspection_tab:
                    self.inspection_video_label.setPixmap(pixmap.scaled(
                        self.inspection_video_label.size(), 
                        Qt.AspectRatioMode.KeepAspectRatio
                    ))
                elif current_widget == self.dataset_tab:
                    self.dataset_video_label.setPixmap(pixmap.scaled(
                        self.dataset_video_label.size(), 
                        Qt.AspectRatioMode.KeepAspectRatio
                    ))
                elif hasattr(self, 'adjustment_tab') and current_widget == self.adjustment_tab:
                    if hasattr(self, 'adjustment_video_label'):
                        self.adjustment_video_label.setPixmap(pixmap.scaled(
                            self.adjustment_video_label.size(), 
                            Qt.AspectRatioMode.KeepAspectRatio
                        ))
                        # Update Sharpness (using pre-calculated score from thread)
                        self.sharpness_bar.setValue(min(int(sharpness_score), 1000))
                        self.sharpness_value_label.setText(f"Score: {sharpness_score:.2f}")

                self.last_frame = frame

    def capture_image(self):
        if hasattr(self, 'last_frame') and self.last_frame is not None:
            path = self.config['paths']['data_raw']
            saved_path = save_image(self.last_frame, path, prefix="dataset")
            self.status_bar.showMessage(f"Image saved to {saved_path}", 3000)

    def logout(self):
        self._stop_camera()
        self.user_manager.logout()
        self.close()

    def closeEvent(self, event):
        print("Closing window, releasing camera...")
        self._stop_camera()
        event.accept()

    def _stop_camera(self):
        if self.camera_thread:
            print("Stopping camera thread...")
            self.camera_thread.stop()
            self.camera_thread = None
            
        if self.camera:
            print("Releasing camera...")
            self.camera.release()
            self.camera = None

    def open_user_management(self):
        dialog = UserManagementDialog(self.user_manager, self)
        dialog.exec()

    def save_camera_params(self):
        if self.camera:
            self.camera.save_parameters()
            self.status_bar.showMessage("Camera parameters saved", 3000)

    def load_camera_params(self):
        if self.camera:
            params = self.camera.load_parameters()
            if params:
                # Update slider values
                if hasattr(self, 'exposure_slider'):
                    self.exposure_slider.setValue(int(params.get('exposure', 0)))
                if hasattr(self, 'brightness_slider'):
                    self.brightness_slider.setValue(int(params.get('brightness', 0)))
                if hasattr(self, 'contrast_slider'):
                    self.contrast_slider.setValue(int(params.get('contrast', 0)))
                if hasattr(self, 'focus_slider'):
                    self.focus_slider.setValue(int(params.get('focus', 0)))
                self.status_bar.showMessage("Camera parameters loaded", 3000)
