from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QLabel, QPushButton, QSlider, QGroupBox, QFormLayout, 
                               QProgressBar, QCheckBox, QStatusBar, QTabWidget, QDoubleSpinBox,
                               QStackedWidget, QSpinBox, QFileDialog)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
import cv2
import numpy as np
import os
import json

from .camera_thread import CameraThread
from ..camera import Camera
from .video_widget import VideoWidget
from ..can_process_img.detect_corner import CornerDetector
from ..can_process_img.crop_cans import CanCropper
from ..can_process_img.rectified_sheet import SheetRectifier
from ..can_process_img.resize_can import CanResizer
from ..can_process_img.align_can import CanAligner

class AdjustmentWindow(QMainWindow):
    def __init__(self, config, dashboard=None):
        super().__init__()
        self.config = config
        self.dashboard = dashboard
        self.camera = None
        self.camera_thread = None
        self.is_high_res = False # Default to Low Res (Performance)
        self.max_focus_score = 1.0 # Dynamic peak tracking
        self.current_frozen_frame = None # Store the captured high-res frame
        
        # Pipeline Components
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        self.detector = CornerDetector()
        self.detector.load_params(os.path.join(base_dir, 'config', 'corner_params.json'))
        
        self.cropper = CanCropper()
        self.cropper.load_params(os.path.join(base_dir, 'config', 'crop_params.json'))
        
        self.rectifier = SheetRectifier()
        self.resizer = CanResizer(size=(300, 300))
        # Note: Aligner needs a reference image, we can init with a placeholder or handle later
        # We will initialize aligner lazily or with a dummy path and update it in the UI
        self.aligner = None 
        
        # Wizard State to pass data between steps
        self.wiz_state = {
            'corners': None,
            'rectified': None,
            'cans': [],
            'resized': [],
            'aligned': []
        }
        
        self.setWindowTitle("Camera Adjustment Mode")
        self.resize(1200, 800)
        
        # Debouncing
        self.pending_ui_updates = {}
        self.slider_labels = {}
        self.update_timer = QTimer()
        self.update_timer.setInterval(300) # 300ms throttling (3.3Hz) to prevent camera command backlog
        self.update_timer.timeout.connect(self._process_pending_updates)
        self.update_timer.start()
        
        self._setup_ui()
        self._start_camera()
        self.showMaximized()
        
    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget) # Vertical for Header + Content
        
        # Header
        header = QHBoxLayout()
        btn_back = QPushButton("<- Dashboard")
        btn_back.clicked.connect(self.go_back)
        header.addWidget(btn_back)
        lbl_title = QLabel("⚙️ Calibration & Tuning")
        lbl_title.setStyleSheet("font-size: 18px; font-weight: bold; margin-left: 20px;")
        header.addWidget(lbl_title)
        header.addStretch()
        layout.addLayout(header)
        
        # Main Content
        content_layout = QHBoxLayout()
        layout.addLayout(content_layout)
        
        # --- Left Side: Camera Feed ---
        video_panel = QGroupBox("Live Feed / Capture")
        video_layout = QVBoxLayout()
        video_layout.setContentsMargins(0,0,0,0) # Maximize space
        
        # Use our new VideoWidget
        self.video_display = VideoWidget()
        video_layout.addWidget(self.video_display)
        
        video_panel.setLayout(video_layout)
        content_layout.addWidget(video_panel, stretch=2)
        
        # --- Right Side: Wizard Controls ---
        self.wizard = QStackedWidget()
        
        # --- PAGE 0: Camera Setup (Live/Focus) ---
        page_cam = QWidget()
        layout_cam = QVBoxLayout(page_cam)
        
        layout_cam.addWidget(QLabel("<h2>1. Camera Setup</h2>"))
        
        # Focus/Sharpness Controls (Existing)
        self.chk_focus_mode = QCheckBox("🔍 Focus Mode (High Res)")
        self.chk_focus_mode.setToolTip("Enables full resolution for precise focusing.")
        self.chk_focus_mode.stateChanged.connect(self.toggle_resolution)
        layout_cam.addWidget(self.chk_focus_mode)

        sharpness_layout = QHBoxLayout()
        sharpness_layout.addWidget(QLabel("Focus Score:"))
        self.btn_reset_peak = QPushButton("Reset Peak")
        self.btn_reset_peak.clicked.connect(self.reset_peak)
        sharpness_layout.addWidget(self.btn_reset_peak)
        layout_cam.addLayout(sharpness_layout)
        
        self.sharpness_bar = QProgressBar()
        self.sharpness_bar.setRange(0, 100)
        self.sharpness_bar.setTextVisible(True)
        self.sharpness_bar.setFormat("%p%")
        self.sharpness_value_label = QLabel("Score: 0.0 (Peak: 0.0)")
        layout_cam.addWidget(self.sharpness_bar)
        layout_cam.addWidget(self.sharpness_value_label)
        
        # Noise Meter
        layout_cam.addWidget(QLabel("Digital Noise Level (Lower is Better):"))
        self.noise_bar = QProgressBar()
        self.noise_bar.setRange(0, 20) # Typical range 0-10. >20 is very noisy.
        self.noise_bar.setTextVisible(True)
        self.noise_bar.setFormat("%v") # Show value
        self.noise_value_label = QLabel("Noise: 0.0")
        layout_cam.addWidget(self.noise_bar)
        layout_cam.addWidget(self.noise_value_label)
        
        # Camera Params Sliders (Using helper)
        params_group = QGroupBox("Camera Parameters")
        params_layout = QFormLayout()
        
        def create_slider(name, prop_id, min_val, max_val, default=128):
            container = QHBoxLayout()
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            
            def to_ui(val): return int((val / 255.0) * 100)
            def to_backend(val): return int((val / 100.0) * 255)
            
            val_label = QLabel(f"{to_ui(default)}%")
            val_label.setFixedWidth(50)
            self.slider_labels[prop_id] = val_label
            
            current = default
            slider.setValue(to_ui(current))
            val_label.setText(f"{to_ui(current)}%")
            
            slider.valueChanged.connect(lambda v: val_label.setText(f"{v}%"))
            slider.valueChanged.connect(lambda v: self.update_camera_param(prop_id, to_backend(v)))
            slider.sliderReleased.connect(self._process_pending_updates)
            
            container.addWidget(slider)
            container.addWidget(val_label)
            params_layout.addRow(name, container)
            return slider

        self.exposure_slider = create_slider("Exposure", cv2.CAP_PROP_EXPOSURE, 0, 255, 0)
        self.brightness_slider = create_slider("Brightness", cv2.CAP_PROP_BRIGHTNESS, 0, 255, 128)
        self.contrast_slider = create_slider("Contrast", cv2.CAP_PROP_CONTRAST, 0, 255, 128)
        self.focus_slider = create_slider("Focus", cv2.CAP_PROP_FOCUS, 0, 255, 20) 
        params_group.setLayout(params_layout)
        layout_cam.addWidget(params_group)
        
        btn_save_cam = QPushButton("Save Camera Params")
        btn_save_cam.clicked.connect(self.save_camera_params)
        layout_cam.addWidget(btn_save_cam)
        
        # Freeze Button
        self.btn_toggle_view = QPushButton("Show Capture (Freeze)")
        self.btn_toggle_view.setCheckable(True)
        self.btn_toggle_view.setStyleSheet("background-color: #f0f0f0; color: #1e1e1e; padding: 8px;")
        self.btn_toggle_view.clicked.connect(self.toggle_view_mode)
        layout_cam.addWidget(self.btn_toggle_view)
        
        layout_cam.addStretch()
        
        # Next Button
        btn_next_1 = QPushButton("Start Calibration (Detect Corners) ➤")
        btn_next_1.setStyleSheet("font-size: 14px; font-weight: bold; background-color: #4CAF50; color: white; padding: 10px;")
        # Need to be in Freeze mode to proceed
        btn_next_1.clicked.connect(lambda: self.go_to_step(1))
        layout_cam.addWidget(btn_next_1)
        
        self.wizard.addWidget(page_cam)
        
        # --- PAGE 1: Corner Detection ---
        page_corners = QWidget()
        layout_corners = QVBoxLayout(page_corners)
        layout_corners.addWidget(QLabel("<h2>2. Tune Corner Detection</h2>"))
        
        # Test Button
        btn_test_detect = QPushButton("▶ Test Detection (Clean View)")
        btn_test_detect.clicked.connect(lambda: self.update_detection_preview(show_rois=False))
        layout_corners.addWidget(btn_test_detect)
        
        det_group = QGroupBox("Corner Detection Params")
        det_form = QFormLayout()
        
        def create_det_slider(name, attr_name, r_min, r_max):
            container = QHBoxLayout()
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(r_min, r_max)
            val_label = QLabel(str(getattr(self.detector, attr_name)))
            val_label.setFixedWidth(50)
            slider.setValue(getattr(self.detector, attr_name))
            
            def on_change(val):
                val_label.setText(str(val))
                setattr(self.detector, attr_name, val)
                self.update_detection_preview(show_rois=True)
            
            slider.valueChanged.connect(on_change)
            container.addWidget(slider)
            container.addWidget(val_label)
            det_form.addRow(name, container)
            
        create_det_slider("ROI Size", "roi_size", 50, 600)
        create_det_slider("Margin Top", "margin_top", 0, 800)
        create_det_slider("Margin Bottom", "margin_bottom", 0, 800)
        create_det_slider("Margin Left", "margin_left", 0, 800)
        create_det_slider("Margin Right", "margin_right", 0, 800)
        create_det_slider("Skew TL->BL (X)", "dist_TL_BL", -500, 500)
        create_det_slider("Skew TR->BR (X)", "dist_TR_BR", -500, 500)
        det_group.setLayout(det_form)
        layout_corners.addWidget(det_group)
        
        btn_save_det = QPushButton("Save Detection Params")
        btn_save_det.clicked.connect(self.save_corner_params)
        layout_corners.addWidget(btn_save_det)
        layout_corners.addStretch()
        
        nav_1 = QHBoxLayout()
        btn_prev_1 = QPushButton("Back")
        btn_prev_1.clicked.connect(lambda: self.go_to_step(0))
        btn_next_corners = QPushButton("Next: Rectify Sheet ➤")
        btn_next_corners.setStyleSheet("font-weight: bold; background-color: #2196F3; color: white;")
        btn_next_corners.clicked.connect(lambda: self.go_to_step(2))
        nav_1.addWidget(btn_prev_1)
        nav_1.addWidget(btn_next_corners)
        layout_corners.addLayout(nav_1)
        
        self.wizard.addWidget(page_corners)
        
        # --- PAGE 2: Rectification ---
        page_rect = QWidget()
        layout_rect = QVBoxLayout(page_rect)
        layout_rect.addWidget(QLabel("<h2>3. Verify Rectification</h2>"))
        layout_rect.addWidget(QLabel("Review the warped image. Adjust Skew if vertical edges are not straight."))
        
        # Rectification Params
        rect_group = QGroupBox("Rectification Params (mm)")
        rect_form = QFormLayout()
        
        # Skew Spinner
        spin_skew = QDoubleSpinBox()
        spin_skew.setRange(0, 200)
        spin_skew.setSingleStep(0.1)
        spin_skew.setValue(self.rectifier.sheet_skew_mm)
        
        def on_skew_change(val):
            self.rectifier.sheet_skew_mm = val
            self.rectifier.update_pixel_values()
            self.run_rectification_step()
            
        spin_skew.valueChanged.connect(on_skew_change)
        rect_form.addRow("Sheet Skew (X Offset):", spin_skew)
        
        rect_group.setLayout(rect_form)
        layout_rect.addWidget(rect_group)
        
        btn_apply_rect = QPushButton("▶ Apply Rectification (Warp)")
        btn_apply_rect.clicked.connect(self.run_rectification_step)
        layout_rect.addWidget(btn_apply_rect)
        
        btn_save_rect = QPushButton("Save Rect Params")
        btn_save_rect.clicked.connect(self.save_rect_params)
        layout_rect.addWidget(btn_save_rect)
        
        layout_rect.addStretch()
        
        nav_2 = QHBoxLayout()
        btn_prev_2 = QPushButton("Back")
        btn_prev_2.clicked.connect(lambda: self.go_to_step(1))
        btn_next_rect = QPushButton("Next: Crop Grid ➤")
        btn_next_rect.setStyleSheet("font-weight: bold; background-color: #2196F3; color: white;")
        btn_next_rect.clicked.connect(lambda: self.go_to_step(3))
        nav_2.addWidget(btn_prev_2)
        nav_2.addWidget(btn_next_rect)
        layout_rect.addLayout(nav_2)
        
        self.wizard.addWidget(page_rect)
        
        # --- PAGE 3: Crop Cans ---
        page_crop = QWidget()
        layout_crop = QVBoxLayout(page_crop)
        layout_crop.addWidget(QLabel("<h2>4. Tune Crop Grid (mm)</h2>"))
        
        btn_preview_crop = QPushButton("▶ Preview Grid on Rectified Image")
        btn_preview_crop.clicked.connect(self.update_crop_preview)
        layout_crop.addWidget(btn_preview_crop)
        
        crop_group = QGroupBox("Grid Params")
        crop_form_layout = QFormLayout()
        
        def create_crop_spinbox(name, attr_name, r_min=0.0, r_max=500.0, step=0.1):
            spin = QDoubleSpinBox()
            spin.setRange(r_min, r_max)
            spin.setSingleStep(step)
            spin.setDecimals(2)
            spin.setValue(getattr(self.cropper, attr_name))
            
            def on_change(val):
                setattr(self.cropper, attr_name, val)
                self.cropper.update_pixel_values()
                self.update_crop_preview()
            
            spin.valueChanged.connect(on_change)
            crop_form_layout.addRow(name, spin)
            
        create_crop_spinbox("First Can X", "first_can_center_x_mm")
        create_crop_spinbox("First Can Y", "first_can_center_y_mm")
        create_crop_spinbox("Step X", "step_x_mm")
        create_crop_spinbox("Step Y", "step_y_mm")
        create_crop_spinbox("Tolerance", "tolerance_box_mm", 0, 50)
        crop_group.setLayout(crop_form_layout)
        layout_crop.addWidget(crop_group)
        
        btn_save_crop = QPushButton("Save Crop Params")
        btn_save_crop.clicked.connect(self.save_crop_params)
        layout_crop.addWidget(btn_save_crop)
        layout_crop.addStretch()
        
        nav_3 = QHBoxLayout()
        btn_prev_3 = QPushButton("Back")
        btn_prev_3.clicked.connect(lambda: self.go_to_step(2))
        btn_next_crop = QPushButton("Next: Resize & Align ➤")
        btn_next_crop.setStyleSheet("font-weight: bold; background-color: #2196F3; color: white;")
        btn_next_crop.clicked.connect(lambda: self.go_to_step(4)) # Going straight to Resize/Align page
        nav_3.addWidget(btn_prev_3)
        nav_3.addWidget(btn_next_crop)
        layout_crop.addLayout(nav_3)
        
        self.wizard.addWidget(page_crop)

        # --- PAGE 4: Resize ---
        page_resize = QWidget()
        layout_resize = QVBoxLayout(page_resize)
        layout_resize.addWidget(QLabel("<h2>5. Resize Cans</h2>"))
        layout_resize.addWidget(QLabel("Set target resolution for AI (Square recommended)."))

        resize_group = QGroupBox("Target Size (px)")
        resize_form = QFormLayout()
        
        self.spin_width = QSpinBox()
        self.spin_width.setRange(32, 1024)
        self.spin_width.setValue(300)
        self.spin_height = QSpinBox()
        self.spin_height.setRange(32, 1024)
        self.spin_height.setValue(300)
        
        resize_form.addRow("Width:", self.spin_width)
        resize_form.addRow("Height:", self.spin_height)
        resize_group.setLayout(resize_form)
        layout_resize.addWidget(resize_group)
        
        btn_show_can = QPushButton("▶ Show Sample Resized Can")
        btn_show_can.clicked.connect(self.run_resize_step)
        layout_resize.addWidget(btn_show_can)
        
        layout_resize.addStretch()
        
        nav_4 = QHBoxLayout()
        btn_prev_4 = QPushButton("Back")
        btn_prev_4.clicked.connect(lambda: self.go_to_step(3))
        btn_next_resize = QPushButton("Next: Alignment ➤")
        btn_next_resize.setStyleSheet("font-weight: bold; background-color: #2196F3; color: white;")
        btn_next_resize.clicked.connect(lambda: self.go_to_step(5))
        nav_4.addWidget(btn_prev_4)
        nav_4.addWidget(btn_next_resize)
        layout_resize.addLayout(nav_4)
        
        self.wizard.addWidget(page_resize)
        
        # --- PAGE 5: Alignment ---
        page_align = QWidget()
        layout_align = QVBoxLayout(page_align)
        layout_align.addWidget(QLabel("<h2>6. Alignment</h2>"))
        layout_align.addWidget(QLabel("Align cans to reference image (orb matching)."))
        
        # Reference Image Loader
        self.lbl_ref_status = QLabel("Reference: Not Loaded")
        btn_load_ref = QPushButton("Load Reference Image...")
        btn_load_ref.clicked.connect(self.load_reference_image)
        layout_align.addWidget(self.lbl_ref_status)
        layout_align.addWidget(btn_load_ref)
        
        btn_test_align = QPushButton("▶ Test Alignment (Sample)")
        btn_test_align.clicked.connect(self.run_alignment_step)
        layout_align.addWidget(btn_test_align)
        
        layout_align.addStretch()
        
        nav_5 = QHBoxLayout()
        btn_prev_5 = QPushButton("Back")
        btn_prev_5.clicked.connect(lambda: self.go_to_step(4))
        btn_finish = QPushButton("✅ Finish & Save All")
        btn_finish.setStyleSheet("font-weight: bold; background-color: #4CAF50; color: white;")
        btn_finish.clicked.connect(self.finish_wizard)
        nav_5.addWidget(btn_prev_5)
        nav_5.addWidget(btn_finish)
        layout_align.addLayout(nav_5)
        
        self.wizard.addWidget(page_align)

        content_layout.addWidget(self.wizard, stretch=1)
        
        # --- Bottom Panel: Camera Info ---
        info_panel = QGroupBox("Camera Details")
        
        # --- Bottom Panel: Camera Info ---
        info_panel = QGroupBox("Camera Details")
        info_layout = QHBoxLayout()
        
        self.lbl_res = QLabel(f"Resolution: {self.config['camera']['width']}x{self.config['camera']['height']}")
        self.lbl_fps = QLabel(f"Target FPS: {self.config['camera']['fps']}")
        self.lbl_backend = QLabel("Backend: PiCamera2 (Native)")
        
        info_layout.addWidget(self.lbl_res)
        info_layout.addWidget(self.lbl_fps)
        info_layout.addWidget(self.lbl_backend)
        info_layout.addStretch()
        
        info_panel.setLayout(info_layout)
        # Add to main vertical layout
        layout.addWidget(info_panel)
        
        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def update_camera_param(self, prop_id, value):
        # Buffer the update
        self.pending_ui_updates[prop_id] = value

    def _process_pending_updates(self):
        # Send buffered commands to the thread
        if self.camera_thread and self.pending_ui_updates:
            for prop_id, value in list(self.pending_ui_updates.items()):
                self.camera_thread.set_camera_property(prop_id, value)
            self.pending_ui_updates.clear()

    def save_camera_params(self):
        """Save slider values directly to JSON"""
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        filepath = os.path.join(base_dir, 'config', 'camera_params.json')
        
        # Helper to convert UI (0-100) to backend (0-255)
        def to_backend(ui_val): return int((ui_val / 100.0) * 255)
        
        # Read current slider positions
        params = {
            'exposure': to_backend(self.exposure_slider.value()) if hasattr(self, 'exposure_slider') else 0,
            'brightness': to_backend(self.brightness_slider.value()) if hasattr(self, 'brightness_slider') else 128,
            'contrast': to_backend(self.contrast_slider.value()) if hasattr(self, 'contrast_slider') else 128,
            'focus': to_backend(self.focus_slider.value()) if hasattr(self, 'focus_slider') else 20
        }
        
        # Save to file
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=4)
        
        print(f"✓ Saved params to {filepath}: {params}")
        self.status_bar.showMessage("Parameters Saved Successfully!", 3000)

    def load_camera_params(self):
        """Load params from file or set defaults, then apply to Thread"""
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        filepath = os.path.join(base_dir, 'config', 'camera_params.json')
        
        print(f"DEBUG: load_camera_params called, checking {filepath}")
        
        params = {}
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    params = json.load(f)
                print(f"✓ Loaded params from file: {params}")
            except Exception as e:
                print(f"✗ Error reading params file: {e}")
        else:
            print(f"✗ Params file not found, using defaults")
        
        # Default fallback values (match slider creation defaults)
        exposure = int(params.get('exposure', 0))        # 0 = Auto
        brightness = int(params.get('brightness', 128))  # 128 = Center (FIXED: was 0)
        contrast = int(params.get('contrast', 128))      # 128 = Neutral (1.0 after mapping)
        focus = int(params.get('focus', 20))             # 20 = default
        
        print(f"DEBUG: Final values to apply - E:{exposure}, B:{brightness}, C:{contrast}, F:{focus}")
        
        # Helper helpers
        def to_ui(val): return int((val / 255.0) * 100)
        
        # Update Sliders (Block signals to prevent double-setting)
        if hasattr(self, 'exposure_slider'):
            self.exposure_slider.blockSignals(True)
            self.exposure_slider.setValue(to_ui(exposure))
            self.exposure_slider.blockSignals(False)
            if cv2.CAP_PROP_EXPOSURE in self.slider_labels:
                self.slider_labels[cv2.CAP_PROP_EXPOSURE].setText(f"{to_ui(exposure)}%")

        if hasattr(self, 'brightness_slider'):
            self.brightness_slider.blockSignals(True)
            self.brightness_slider.setValue(to_ui(brightness))
            self.brightness_slider.blockSignals(False)
            if cv2.CAP_PROP_BRIGHTNESS in self.slider_labels:
                self.slider_labels[cv2.CAP_PROP_BRIGHTNESS].setText(f"{to_ui(brightness)}%")

        if hasattr(self, 'contrast_slider'):
            self.contrast_slider.blockSignals(True)
            self.contrast_slider.setValue(to_ui(contrast))
            self.contrast_slider.blockSignals(False)
            if cv2.CAP_PROP_CONTRAST in self.slider_labels:
                self.slider_labels[cv2.CAP_PROP_CONTRAST].setText(f"{to_ui(contrast)}%")

        if hasattr(self, 'focus_slider'):
            self.focus_slider.blockSignals(True)
            self.focus_slider.setValue(to_ui(focus))
            self.focus_slider.blockSignals(False)
            if cv2.CAP_PROP_FOCUS in self.slider_labels:
                self.slider_labels[cv2.CAP_PROP_FOCUS].setText(f"{to_ui(focus)}%")

        # APPLY TO CAMERA THREAD
        if self.camera_thread:
            self.camera_thread.set_camera_property(cv2.CAP_PROP_EXPOSURE, exposure)
            self.camera_thread.set_camera_property(cv2.CAP_PROP_BRIGHTNESS, brightness)
            self.camera_thread.set_camera_property(cv2.CAP_PROP_CONTRAST, contrast)
            self.camera_thread.set_camera_property(cv2.CAP_PROP_FOCUS, focus)
            print("DEBUG: Sent loaded/default params to CameraThread")

        msg = "Parameters Loaded" if params else "Default Parameters Applied"
        self.status_bar.showMessage(msg, 3000)

    def toggle_resolution(self, state):
        self.is_high_res = (state == Qt.CheckState.Checked.value)
        if self.camera_thread:
            # Instant Switch! No restart needed.
            # 'lo' stream -> 'main' stream (or vice versa) handled by thread/hardware
            self.camera_thread.use_high_res = self.is_high_res
            print(f"Switched stream. High Res: {self.is_high_res}")

    def _start_camera(self):
        try:
            # Ensure no stale UI updates overwrite loaded parameters
            self.pending_ui_updates.clear()
            
            # Always initialize with full resolution config
            # Picamera2 dual stream will handle the low-res 'lo' stream for us automatically.
            width = self.config.get('camera', {}).get('width', 1920)
            height = self.config.get('camera', {}).get('height', 1080)
            
            print(f"Starting camera in Dual Stream Mode. Main: {width}x{height}, Preview: 640x480")

            self.camera = Camera(
                camera_index=self.config.get('camera', {}).get('index', 0),
                width=width,
                height=height,
                fps=self.config.get('camera', {}).get('fps', 30)
            )
            
            # Thread handles fetching from 'lo' (default) or 'main' stream
            self.camera_thread = CameraThread(self.camera)
            # Set initial state
            self.camera_thread.use_high_res = self.is_high_res
            
            self.camera_thread.frame_captured.connect(self.update_frame)
            self.camera_thread.start()
            
            # Delay loading params to allow camera/thread to fully initialize
            QTimer.singleShot(500, self.load_camera_params)
            
            # Update Info Panel
            if hasattr(self, 'lbl_res'):
                self.lbl_res.setText(f"Resolution: {width}x{height} (Dual Stream)")
                
        except Exception as e:
            print(f"Error initializing camera: {e}")

    def go_to_step(self, step):
        """Handle Wizard Navigation and Data Passing"""
        
        if step == 1: # Moving to Detection
            if not self.btn_toggle_view.isChecked():
                self.status_bar.showMessage("⚠️ Please Freeze the image first!", 3000)
                return
            if self.current_frozen_frame is None:
                self.status_bar.showMessage("⚠️ No frame captured!", 3000)
                return
            self.update_detection_preview()

        elif step == 2: # Moving to Rectification
            # Ensure detection ran and we have corners
            corners = self.detector.detect(self.current_frozen_frame)
            if corners is None or len(corners) != 4:
                self.status_bar.showMessage("⚠️ Detection failed! Please adjust parameters.", 3000)
                return
            self.wiz_state['corners'] = corners
            # Auto-run rectification preview
            self.run_rectification_step()

        elif step == 3: # Moving to Crop
            if self.wiz_state['rectified'] is None:
                self.status_bar.showMessage("⚠️ Rectification data missing!", 3000)
                return
            self.update_crop_preview()
            
        elif step == 4: # Moving to Resize
            # Ensure crop step has data, actually Run the crop
            if self.wiz_state['rectified'] is None:
                self.status_bar.showMessage("⚠️ Rectification missing!", 3000)
                return
            
            # Extract cans now
            try:
                self.wiz_state['cans'] = self.cropper.crop_cans(self.wiz_state['rectified'])
                self.status_bar.showMessage(f"Extracted {len(self.wiz_state['cans'])} cans", 2000)
                self.run_resize_step() # Show first one
            except Exception as e:
                self.status_bar.showMessage(f"Crop Error: {e}", 4000)
                return

        elif step == 5: # Moving to Alignment
            # Ensure resize is configured
            if not self.wiz_state['cans']:
                self.status_bar.showMessage("⚠️ No cans extracted!", 3000)
                return
            
            # Run resize on the first can to allow alignment test
            self.run_resize_step() 
            
            # Auto-load Default Reference if not set
            if self.aligner is None:
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                def_ref_path = os.path.join(base_dir, 'models', 'can_reference', 'aligned_can_reference.png')
                if os.path.exists(def_ref_path):
                    try:
                        self.aligner = CanAligner(def_ref_path, target_size=self.resizer.size)
                        self.lbl_ref_status.setText(f"Reference: {os.path.basename(def_ref_path)} (Auto)")
                        print(f"Auto-loaded reference: {def_ref_path}")
                        if self.aligner.ref_img is not None:
                             self.display_image(self.aligner.ref_img)
                    except Exception as e:
                        print(f"Failed to auto-load reference: {e}")

            # (Resize happens on demand in preview usually, but let's cache it)
            if self.wiz_state['resized']:
                 self.run_alignment_step()

        self.wizard.setCurrentIndex(step)

    def run_rectification_step(self):
        """Warp the image using detected corners"""
        print("DEBUG: Executing run_rectification_step")
        if self.current_frozen_frame is not None and self.wiz_state['corners'] is not None:
            try:
                print(f"DEBUG: Frame shape: {self.current_frozen_frame.shape}")
                print(f"DEBUG: Corners: {self.wiz_state['corners']}")
                
                self.wiz_state['rectified'] = self.rectifier.get_warped(
                    self.current_frozen_frame, 
                    self.wiz_state['corners']
                )
                
                if self.wiz_state['rectified'] is not None:
                    h, w = self.wiz_state['rectified'].shape[:2]
                    print(f"DEBUG: Rectified Image Shape: {w}x{h}")
                    self.display_image(self.wiz_state['rectified'])
                    self.status_bar.showMessage(f"Rectification Applied ({w}x{h})", 2000)
                else:
                    self.status_bar.showMessage("Error: Rectifier returned None", 3000)
                    print("DEBUG: Rectifier returned None")
                    
            except Exception as e:
                self.status_bar.showMessage(f"Rectification Error: {e}")
                print(f"DEBUG: Exception in rectification: {e}")
                import traceback
                traceback.print_exc()
        else:
            msg = "Missing Data: "
            if self.current_frozen_frame is None: msg += "No Frame. "
            if self.wiz_state['corners'] is None: msg += "No Corners."
            self.status_bar.showMessage(msg, 3000)
            print(f"DEBUG: {msg}")

    def update_crop_preview(self):
        """Draw grid preview on the RECTIFIED image"""
        # Ensure we have the rectified image from previous step
        if self.wiz_state['rectified'] is None:
            # If jumping straight here (dev), try to gen it
            if self.current_frozen_frame is not None:
                corners = self.detector.detect(self.current_frozen_frame)
                if corners is not None:
                    self.wiz_state['rectified'] = self.rectifier.get_warped(self.current_frozen_frame, corners)
            
        if self.wiz_state['rectified'] is not None:
            preview_frame = self.wiz_state['rectified'].copy()
            preview_frame = self.cropper.draw_grid_preview(preview_frame)
            self.display_image(preview_frame)

    def display_image(self, frame):
        """Helper to display BGR numpy image"""
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)
        self.video_display.set_frame(pixmap)

    def toggle_view_mode(self):
        """Toggle between Live Preview and High-Res Static Capture (Freeze)"""
        is_freeze_mode = self.btn_toggle_view.isChecked()
        
        if is_freeze_mode:
            # Change to Capture Mode (Freeze)
            self.btn_toggle_view.setText("Return to Live View")
            if self.camera_thread:
                self.camera_thread.paused = True
                QTimer.singleShot(50, self._capture_and_show_high_res)
        else:
            # Return to Live Mode
            self.current_frozen_frame = None 
            self.btn_toggle_view.setText("Show Capture (Freeze)")
            
            # Reset Wizard to Step 0 if unfreezing
            self.wizard.setCurrentIndex(0)
            
            if self.camera_thread:
                self.camera_thread.paused = False
                self.status_bar.showMessage("Resumed Live Preview", 2000)

    def _capture_and_show_high_res(self):
        if self.camera:
            try:
                frame = self.camera.get_frame(stream_name='main')
                if frame is not None:
                    self.current_frozen_frame = frame 
                    self.display_image(frame)
                    
                    self.status_bar.showMessage(f"Captured: {frame.shape[1]}x{frame.shape[0]} (Static Mode)")
                else:
                    self.status_bar.showMessage("Error: Could not capture frame")
            except Exception as e:
                self.status_bar.showMessage(f"Capture Error: {e}")
                
    def save_rect_params(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(base_dir, 'config', 'rect_params.json')
        if self.rectifier.save_params(path):
            self.status_bar.showMessage("Rectification Parameters Saved!", 3000)

    def save_crop_params(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(base_dir, 'config', 'crop_params.json')
        if self.cropper.save_params(path):
            self.status_bar.showMessage("Crop Parameters Saved!", 3000)

    def save_corner_params(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(base_dir, 'config', 'corner_params.json')
        if self.detector.save_params(path):
            self.status_bar.showMessage("Corner Parameters Saved!", 3000)

    def update_detection_preview(self, show_rois=True):
        if self.current_frozen_frame is not None:
            preview_frame = self.current_frozen_frame.copy()
            if show_rois:
                preview_frame = self.detector.visualize_search_areas(preview_frame)
            
            corners = self.detector.detect(preview_frame)
            preview_frame = self.detector.draw_preview(preview_frame, corners)
            self.display_image(preview_frame)

    def showEvent(self, event):
        """Called when window is shown. Ensure latest parameters are loaded."""
        super().showEvent(event)
        # Add a small delay to ensure UI is ready
        QTimer.singleShot(500, self.load_camera_params)

    def reset_peak(self):
        self.max_focus_score = 1.0
        self.sharpness_bar.setValue(0)
        self.sharpness_value_label.setText("Score: 0.0 (Reset)")

    def update_frame(self, qt_image, sharpness_score, noise_score, raw_frame):
        """Called by CameraThread when a new frame is available"""
        # qt_image is now a QImage passed directly from the thread
        if qt_image is not None and not qt_image.isNull():
            pixmap = QPixmap.fromImage(qt_image)
            
            # Use custom widget set_frame
            self.video_display.set_frame(pixmap)
            
            # Update Sharpness Logic
            # 1. Update Peak
            if sharpness_score > self.max_focus_score:
                self.max_focus_score = sharpness_score
            
            # 2. Calculate Percentage
            if self.max_focus_score > 0:
                percent = int((sharpness_score / self.max_focus_score) * 100)
            else:
                percent = 0
            
            # 3. Update Bar
            self.sharpness_bar.setValue(min(percent, 100))
            
            # 4. Visual Feedback (Green if > 90%)
            if percent >= 90:
                # Green chunk
                self.sharpness_bar.setStyleSheet("""
                    QProgressBar::chunk { background-color: #4caf50; } 
                    QProgressBar { border: 1px solid #4caf50; }
                """)
            else:
                # Default Teal chunk
                self.sharpness_bar.setStyleSheet("""
                    QProgressBar::chunk { background-color: #00bcd4; }
                    QProgressBar { border: 1px solid #4d4d4d; }
                """)

            # 5. Update Text
            self.sharpness_value_label.setText(
                f"Score: {sharpness_score:.1f} / Peak: {self.max_focus_score:.1f}"
            )
            
            # 6. Update Noise Meter
            self.noise_bar.setValue(min(int(noise_score), 20))
            self.noise_value_label.setText(f"Noise: {noise_score:.2f}")
            
            # Color Coding (Low is Good)
            if noise_score < 3.0:
                # Green (Excellent)
                self.noise_bar.setStyleSheet("QProgressBar::chunk { background-color: #4caf50; }")
            elif noise_score < 8.0:
                # Yellow (Acceptable)
                self.noise_bar.setStyleSheet("QProgressBar::chunk { background-color: #ffeb3b; }")
            else:
                # Red (High Noise)
                self.noise_bar.setStyleSheet("QProgressBar::chunk { background-color: #f44336; }")

    def run_resize_step(self):
        """Update Resizer params and show preview of first can"""
        w = self.spin_width.value()
        h = self.spin_height.value()
        self.resizer.size = (w, h)
        
        if self.wiz_state['cans']:
            # Pick the middle can (likely to be good) or first
            sample_can = self.wiz_state['cans'][0] 
            resized = self.resizer.process(sample_can['image'])
            
            self.wiz_state['resized'] = [resized] # Store for alignment
            self.display_image(resized)
            self.status_bar.showMessage(f"Resized to {w}x{h}", 2000)

    def load_reference_image(self):
        """Open file dialog to load reference"""
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        initial_dir = os.path.join(base_dir, 'models', 'can_reference')
        if not os.path.exists(initial_dir):
            initial_dir = ''
            
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Reference Image', initial_dir, "Box Images (*.png *.jpg *.bmp)")
        
        if fname:
            try:
                self.aligner = CanAligner(fname, target_size=self.resizer.size)
                self.lbl_ref_status.setText(f"Reference: {os.path.basename(fname)}")
                
                # Show reference briefly
                if self.aligner.ref_img is not None:
                     self.display_image(self.aligner.ref_img)
                     
                self.status_bar.showMessage("Reference Loaded", 2000)
            except Exception as e:
                self.status_bar.showMessage(f"Error loading ref: {e}", 4000)

    def run_alignment_step(self):
        """Test alignment on the sample resized can"""
        if self.aligner is None:
            self.status_bar.showMessage("⚠️ Load Reference Image First!", 3000)
            return

        # Ensure we have a resized sample
        if not self.wiz_state['resized']:
            self.run_resize_step()
            
        if self.wiz_state['resized']:
             sample = self.wiz_state['resized'][0]
             
             # Update target size of aligner if changed
             self.aligner.target_size = self.resizer.size
             
             aligned = self.aligner.align(sample)
             if aligned is not None:
                 self.display_image(aligned)
                 self.status_bar.showMessage("Alignment Applied", 2000)
             else:
                 self.status_bar.showMessage("Alignment Failed (Check ORB features)", 3000)

    def finish_wizard(self):
        """Save everything and close"""
        self.save_corner_params()
        self.save_rect_params()
        self.save_crop_params()
        self.status_bar.showMessage("✅ All Calibration Parameters Saved!", 4000)

    def go_back(self):
        self.close()
        if self.dashboard:
            self.dashboard.show()

    def closeEvent(self, event):
        if self.camera_thread:
            self.camera_thread.stop()
        if self.camera:
            self.camera.release()
        event.accept()
