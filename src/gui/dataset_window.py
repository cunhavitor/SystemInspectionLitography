from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QLabel, QPushButton, QStatusBar, QSpinBox, QProgressBar, QGroupBox, QFormLayout, QTextEdit, QFileDialog, QComboBox, QDialog, QLineEdit, QDialogButtonBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from ..camera import Camera
from ..sku_manager import SKUManager
from .dataset_window_dialogs import AddNewSKUDialog
from .camera_thread import CameraThread
from ..utils import save_image
from ..can_process_img.detect_corner import CornerDetector
from ..can_process_img.rectified_sheet import SheetRectifier
from ..can_process_img.crop_cans import CanCropper
from ..can_process_img.resize_can import CanResizer
from ..can_process_img.align_can import CanAligner
from ..can_process_img.nomrmalize_can import prepare_for_autoencoder
from ..can_process_img.dataset_validation import is_image_good_for_dataset

# Optional report generation (requires matplotlib, seaborn)
try:
    from ..reports.dataset_report import generate_dataset_report, DatasetReportGenerator
    REPORTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠ Report generation unavailable: {e}")
    REPORTS_AVAILABLE = False
    generate_dataset_report = None

import cv2
import numpy as np
import os
import time
from datetime import datetime

class DatasetWindow(QMainWindow):
    def __init__(self, config, dashboard=None):
        super().__init__()
        self.config = config
        self.config = config
        self.dashboard = dashboard
        
        # Init SKU Manager
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.sku_manager = SKUManager(os.path.join(base_dir, 'config', 'skus.json'))
        
        self.camera = None
        self.camera_thread = None
        self.last_frame = None
        
        # Processing pipeline
        self.detector = None
        self.rectifier = None
        self.cropper = None
        self.resizer = None
        self.aligner = None
        self.reference_image = None
        
        # Capture state
        self.is_capturing = False
        self.captured_count = 0
        self.total_cans_processed = 0
        self.global_can_counter = 0  # Global counter for can numbering
        self.total_good_cans = 0  # Track good cans
        self.total_bad_cans = 0   # Track bad cans
        self.session_start_time = None  # Track session duration
        
        self.setWindowTitle("Dataset Collection")
        self.resize(1000, 700)
        
        self._setup_ui()
        self._init_processors()
        self._start_camera()
        self.showMaximized()
        
    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Header
        header = QHBoxLayout()
        btn_back = QPushButton("← Dashboard")
        btn_back.clicked.connect(self.go_back)
        header.addWidget(btn_back)
        header.addStretch()
        
        title = QLabel("Dataset Capture & Processing")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        header.addWidget(title)
        header.addStretch()
        
        # SKU Selection
        sku_layout = QHBoxLayout()
        sku_layout.addWidget(QLabel("SKU:"))
        
        self.sku_combo = QComboBox()
        self.sku_combo.setMinimumWidth(200)
        self.sku_combo.setStyleSheet("background-color: #333; color: white; padding: 5px;")
        
        # Populate SKUs
        sku_names = self.sku_manager.get_sku_names()
        self.sku_combo.addItems(sku_names)
        
        # Set current selection
        current_sku_data = self.sku_manager.get_active_sku()
        if current_sku_data:
            self.sku_combo.setCurrentText(current_sku_data.get("name", ""))
            
        self.sku_combo.currentTextChanged.connect(self.on_sku_changed)
        sku_layout.addWidget(self.sku_combo)
        
        btn_add_sku = QPushButton("➕ New SKU")
        btn_add_sku.setStyleSheet("background-color: #2196F3; color: white; padding: 5px 10px; border-radius: 3px;")
        btn_add_sku.clicked.connect(self.add_new_sku)
        sku_layout.addWidget(btn_add_sku)
        
        header.addLayout(sku_layout)
        # header.addStretch() # Removed one stretch to balance
        main_layout.addLayout(header)
        
        # Main content: Camera on Left, Logs on Right
        content_layout = QHBoxLayout()
        
        # LEFT SIDE - Camera Feed
        left_panel = QVBoxLayout()
        
        camera_group = QGroupBox("Camera Feed")
        camera_layout = QVBoxLayout()
        
        self.video_label = QLabel("Initializing Camera...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.setScaledContents(True)
        self.video_label.setMinimumSize(640, 480)
        camera_layout.addWidget(self.video_label)
        
        camera_group.setLayout(camera_layout)
        left_panel.addWidget(camera_group)
        
        content_layout.addLayout(left_panel, stretch=2)
        
        # RIGHT SIDE - Processing Status & Statistics
        right_panel = QVBoxLayout()
        
        # Processing Status Log
        status_group = QGroupBox("Processing Log")
        status_layout = QVBoxLayout()
        
        self.lbl_process_status = QTextEdit()
        self.lbl_process_status.setReadOnly(True)
        self.lbl_process_status.setStyleSheet(
            "font-size: 12px; "
            "padding: 10px; "
            "background-color: #2b2b2b; "
            "color: #00ff00; "
            "border-radius: 5px; "
            "font-family: monospace;"
        )
        self.lbl_process_status.setMinimumHeight(300)
        self.lbl_process_status.append("Ready - Click 'Capture & Process' to begin")
        status_layout.addWidget(self.lbl_process_status)
        
        status_group.setLayout(status_layout)
        right_panel.addWidget(status_group, stretch=1)
        
        # Session Statistics
        stats_group = QGroupBox("Session Statistics")
        stats_layout = QVBoxLayout()
        
        self.lbl_status = QLabel(
            "Captures: 0\n"
            "Total cans: 0\n"
            "Quality: 0 good / 0 bad (0%)\n"
            "Session time: 00:00:00\n"
            "Dataset: Not created yet"
        )
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("font-size: 11px; padding: 5px; line-height: 1.4;")
        stats_layout.addWidget(self.lbl_status)
        
        stats_group.setLayout(stats_layout)
        right_panel.addWidget(stats_group)
        
        # Progress Bar
        progress_group = QGroupBox("Processing Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% - %v/%m cans")
        self.progress_bar.setStyleSheet(
            "QProgressBar {"
            "   border: 2px solid #555;"
            "   border-radius: 5px;"
            "   text-align: center;"
            "   background-color: #2b2b2b;"
            "}"
            "QProgressBar::chunk {"
            "   background-color: #4CAF50;"
            "   border-radius: 3px;"
            "}"
        )
        progress_layout.addWidget(self.progress_bar)
        
        progress_group.setLayout(progress_layout)
        right_panel.addWidget(progress_group)
        
        content_layout.addLayout(right_panel, stretch=1)
        
        main_layout.addLayout(content_layout, stretch=1)
        
        # BOTTOM - Action Buttons
        btn_layout = QHBoxLayout()
        
        self.btn_capture = QPushButton("📸 Capture & Process")
        self.btn_capture.setStyleSheet(
            "font-size: 16px; "
            "font-weight: bold; "
            "background-color: #4CAF50; "
            "color: white; "
            "padding: 15px; "
            "border-radius: 5px;"
        )
        self.btn_capture.clicked.connect(self.capture_and_process)
        btn_layout.addWidget(self.btn_capture)
        
        self.btn_export = QPushButton("📦 Export Dataset")
        self.btn_export.setStyleSheet(
            "font-size: 16px; "
            "font-weight: bold; "
            "background-color: #2196F3; "
            "color: white; "
            "padding: 15px; "
            "border-radius: 5px;"
        )
        self.btn_export.clicked.connect(self.export_dataset)
        btn_layout.addWidget(self.btn_export)
        
        self.btn_report = QPushButton("📊 Generate Report")
        self.btn_report.setStyleSheet(
            "font-size: 16px; "
            "font-weight: bold; "
            "background-color: #9C27B0; "
            "color: white; "
            "padding: 15px; "
            "border-radius: 5px;"
        )
        self.btn_report.clicked.connect(self.generate_report)
        btn_layout.addWidget(self.btn_report)

        self.btn_clean = QPushButton("🧹 Clean Outliers")
        self.btn_clean.setStyleSheet(
            "font-size: 16px; "
            "font-weight: bold; "
            "background-color: #FF9800; "
            "color: white; "
            "padding: 15px; "
            "border-radius: 5px;"
        )
        self.btn_clean.clicked.connect(self.clean_outliers)
        btn_layout.addWidget(self.btn_clean)
        
        main_layout.addLayout(btn_layout)
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def _init_processors(self):
        """Initialize all processing pipeline components"""
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            # Initialize processors
            self.detector = CornerDetector()
            self.detector.load_params(os.path.join(base_dir, 'config', 'corner_params.json'))
            
            self.rectifier = SheetRectifier()
            self.rectifier.load_params(os.path.join(base_dir, 'config', 'rect_params.json'))
            
            self.cropper = CanCropper()
            self.cropper.load_params(os.path.join(base_dir, 'config', 'crop_params.json'))
            
            self.resizer = CanResizer(size=(448, 448))
            
            # Load reference image for alignment from active SKU
            sku_data = self.sku_manager.get_active_sku()
            ref_path = sku_data.get("reference_path", "")
            
            # Fallback if empty or not found (try default location)
            if not ref_path or not os.path.exists(ref_path):
                 ref_path = os.path.join(base_dir, 'models', 'can_reference', 'aligned_can_reference448.png')
            
            if os.path.exists(ref_path):
                self.aligner = CanAligner(ref_path, target_size=(448, 448))
                self.reference_image = cv2.imread(ref_path)
                print(f"✓ Loaded reference image: {ref_path}")
                print(f"  Reference shape: {self.reference_image.shape if self.reference_image is not None else 'Failed to load'}")
                self.log(f"✓ Reference image loaded for validation")
            else:
                print(f"⚠ Reference image not found: {ref_path}")
                self.aligner = None
                self.reference_image = None
                self.log(f"⚠ No reference image - validation will be skipped")
                
            self.status_bar.showMessage("Pipeline processors initialized", 3000)
            
        except Exception as e:
            self.status_bar.showMessage(f"Error initializing processors: {e}", 5000)
            print(f"Processor init error: {e}")

    def _start_camera(self):
        try:
            self.camera = Camera(
                camera_index=self.config['camera']['index'],
                width=self.config['camera']['width'],
                height=self.config['camera']['height'],
                fps=self.config['camera']['fps']
            )
            
            self.camera_thread = CameraThread(self.camera)
            self.camera_thread.frame_captured.connect(self.update_frame)
            self.camera_thread.start()
            self.camera.load_parameters()
        except Exception as e:
            self.status_bar.showMessage(f"Error: {str(e)}")

    def update_frame(self, qt_image, sharpness, raw_frame):
        if qt_image is not None and not qt_image.isNull():
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))
        
        if raw_frame is not None:
            self.last_frame = raw_frame

    def log(self, message):
        """Append a message to the processing log and auto-scroll to bottom"""
        from PySide6.QtWidgets import QApplication
        self.lbl_process_status.append(message)
        # Auto-scroll to bottom
        scrollbar = self.lbl_process_status.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        # Force UI update to show the log immediately
        QApplication.processEvents()

    def capture_and_process(self):
        """Capture and process a single image with detailed status feedback"""
        if self.is_capturing:
            return
        
        self.is_capturing = True
        self.btn_capture.setEnabled(False)
        
        # Create dataset folder if doesn't exist
        if not hasattr(self, 'dataset_folder') or not os.path.exists(self.dataset_folder):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.dataset_folder = os.path.join(base_dir, 'data', 'dataset', f'batch_{timestamp}')
            os.makedirs(self.dataset_folder, exist_ok=True)
            self.status_bar.showMessage(f"Dataset folder: {self.dataset_folder}", 5000)
            # Start session timer
            self.session_start_time = datetime.now()
        
        self.captured_count += 1
        
        try:
            self.log(f"\n{'='*50}")
            self.log(f"🔹 Capture #{self.captured_count} - Pausing camera preview...")
            
            # Pause preview
            if self.camera_thread:
                self.camera_thread.paused = True
                time.sleep(0.05)
            
            self.log("📸 Capturing high-resolution frame...")
            
            # Capture high-res frame
            frame = self.camera.get_frame(stream_name='main')
            
            if frame is None:
                raise Exception("Failed to capture frame from camera")
            
            # Process through pipeline with detailed status
            processed_cans, good_count, bad_count = self._process_frame(frame, self.captured_count)
            
            # Update total cans counter and quality counts
            self.total_cans_processed += len(processed_cans)
            self.total_good_cans += good_count
            self.total_bad_cans += bad_count
            
            # Calculate session duration
            if self.session_start_time:
                duration = datetime.now() - self.session_start_time
                hours, remainder = divmod(int(duration.total_seconds()), 3600)
                minutes, seconds = divmod(remainder, 60)
                session_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            else:
                session_time = "00:00:00"
            
            # Calculate success rate
            if self.total_cans_processed > 0:
                success_rate = int((self.total_good_cans / self.total_cans_processed) * 100)
            else:
                success_rate = 0
            
            # Calculate average cans per capture
            avg_cans = self.total_cans_processed / self.captured_count if self.captured_count > 0 else 0
            
            # Update statistics
            self.lbl_status.setText(
                f"Captures: {self.captured_count} (avg {avg_cans:.1f} cans/capture)\n"
                f"Total cans: {self.total_cans_processed}\n"
                f"Quality: {self.total_good_cans} good / {self.total_bad_cans} bad ({success_rate}%)\n"
                f"Session time: {session_time}\n"
                f"Dataset: {os.path.basename(self.dataset_folder)}"
            )
            self.log(f"✅ Capture #{self.captured_count} complete!")
            self.log(f"   Processed {len(processed_cans)} cans successfully.")
            self.log(f"   Ready for next capture.")
            
        except Exception as e:
            self.log(f"❌ Error on capture #{self.captured_count}:")
            self.log(f"   {str(e)}")
            print(f"Capture {self.captured_count} error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Resume preview
            if self.camera_thread:
                self.camera_thread.paused = False
            
            self.is_capturing = False
            self.btn_capture.setEnabled(True)

    def _process_frame(self, frame, capture_id):
        """Process a single frame through the full pipeline"""
        processed_cans = []
        
        # Step 1: Detect corners
        self.log("🔍 Step 1/6: Detecting corners...")
        
        corners = self.detector.detect(frame)
        if corners is None or len(corners) != 4:
            raise Exception("Corner detection failed - could not find 4 corners")
        
        # Step 2: Rectify sheet
        self.log("📐 Step 2/6: Rectifying sheet...")
        
        rectified = self.rectifier.get_warped(frame, corners)
        if rectified is None:
            raise Exception("Rectification failed")
        
        # Step 3: Crop cans
        self.log("✂️ Step 3/6: Cropping cans...")
        
        cans = self.cropper.crop_cans(rectified)
        if not cans:
            raise Exception("No cans detected in rectified sheet")
        
        self.log(f"✂️ Step 3/6: Found {len(cans)} cans")
        
        # Step 4-6: Resize, Align, Normalize each can
        # Create train and debug folders if they don't exist
        train_folder = os.path.join(self.dataset_folder, 'train')
        debug_folder = os.path.join(self.dataset_folder, 'debug')
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(debug_folder, exist_ok=True)
        
        good_count = 0
        bad_count = 0
        total_cans = len(cans)
        
        # Initialize progress bar
        self.progress_bar.setMaximum(total_cans)
        self.progress_bar.setValue(0)
        
        for can_idx, item in enumerate(cans):
            try:
                # Extract image from dict
                can_img = item['image']
                
                # Only log for first can to avoid spam
                if can_idx == 0:
                    self.log(f"🔧 Step 4-6/6: Processing {total_cans} cans (Resize → Align → Normalize)...")
                
                # Resize
                resized = self.resizer.process(can_img)
                
                # Align (if aligner available)
                if self.aligner:
                    aligned = self.aligner.align(resized)
                else:
                    aligned = resized
                
                # Normalize
                normalized = prepare_for_autoencoder(aligned, target_size=(448, 448))
                
                # Increment global can counter
                self.global_can_counter += 1
                can_number = self.global_can_counter
                
                # Validate (if reference available)
                if self.reference_image is not None:
                    is_good, reason = is_image_good_for_dataset(
                        normalized, 
                        self.reference_image,
                        min_sharpness=50,  # Lower threshold for dataset collection
                        check_alignment=True  # Enable alignment check with resized reference
                    )
                    if is_good:
                        # Good image -> save to train folder
                        folder = train_folder
                        status = "good"
                        good_count += 1
                        # Log success with validation details
                        self.log(f"   Can {can_number:03d} → train: {reason}")
                    else:
                        # Bad image -> save to debug folder
                        folder = debug_folder
                        status = "bad"
                        bad_count += 1
                        # Log validation issues to console for debugging
                        self.log(f"   Can {can_number:03d} → debug: {reason}")
                else:
                    # No reference -> save to debug as unknown
                    folder = debug_folder
                    status = "unknown"
                    bad_count += 1
                    self.log(f"   Can {can_number:03d} → debug: No reference image")
                
                # Save with global can number
                filename = f'can{can_number:03d}.png'
                filepath = os.path.join(folder, filename)
                cv2.imwrite(filepath, normalized)
                processed_cans.append(filepath)
                
                # DEBUG: Log brightness for first 5 cans to verify consistency
                if can_idx < 5:
                    brightness = np.mean(normalized)
                    print(f"[DATASET] Can {can_number:03d} - Brightness: {brightness:.2f}, Folder: {folder.split('/')[-1]}")
                
                # Update progress bar
                self.progress_bar.setValue(can_idx + 1)
                
                # Only log last can to show completion
                if can_idx == total_cans - 1:
                    self.log(f"💾 Step 4-6/6: Saved {total_cans} cans")
                    # Log validation summary
                    if self.reference_image is not None:
                        self.log(f"✓ Validation: {good_count} → train/, {bad_count} → debug/")
                    else:
                        self.log(f"⚠ Validation: Skipped (no reference) - All → debug/")

                
            except Exception as e:
                print(f"Error processing can {can_idx}: {e}")
                bad_count += 1
                # Still update progress even on error
                self.progress_bar.setValue(can_idx + 1)
        
        # Reset progress bar
        self.progress_bar.setValue(0)

        
        return processed_cans, good_count, bad_count

    def export_dataset(self):
        """Export the current dataset folder as a compressed archive"""
        if not hasattr(self, 'dataset_folder') or not os.path.exists(self.dataset_folder):
            self.log("\n" + "="*50)
            self.log("⚠️ No dataset to export yet!")
            self.log("   Please capture some images first.")
            self.status_bar.showMessage("No dataset folder found", 3000)
            return
        
        try:
            import shutil
            
            # Get folder name for the zip file
            folder_name = os.path.basename(self.dataset_folder)
            
            # Create folder selection dialog with better styling
            dialog = QFileDialog(self)
            dialog.setFileMode(QFileDialog.Directory)
            dialog.setOption(QFileDialog.ShowDirsOnly, True)
            dialog.setWindowTitle("Choose Export Destination Folder")
            dialog.setDirectory(os.path.expanduser("~"))
            
            # Apply custom stylesheet for better contrast
            dialog.setStyleSheet("""
                QFileDialog {
                    background-color: #ffffff;
                    color: #000000;
                }
                QTreeView, QListView {
                    background-color: #ffffff;
                    color: #000000;
                    selection-background-color: #0078d7;
                    selection-color: #ffffff;
                }
                QLineEdit {
                    background-color: #ffffff;
                    color: #000000;
                    border: 1px solid #cccccc;
                    padding: 5px;
                }
                QPushButton {
                    background-color: #f0f0f0;
                    color: #000000;
                    border: 1px solid #cccccc;
                    padding: 5px 15px;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    background-color: #e0e0e0;
                }
                QLabel {
                    color: #000000;
                }
            """)
            
            # Show dialog and get result
            if dialog.exec():
                export_dir = dialog.selectedFiles()[0]
            else:
                export_dir = None
            
            # User cancelled the dialog
            if not export_dir:
                self.log("\n" + "="*50)
                self.log("⚠️ Export cancelled by user")
                return
            
            # Create full path for the zip file
            save_path = os.path.join(export_dir, f"{folder_name}.zip")
            
            self.log("\n" + "="*50)
            self.log("📦 Exporting dataset...")
            self.log(f"   Destination: {save_path}")
            
            # Remove existing zip if it exists
            if os.path.exists(save_path):
                os.remove(save_path)
            
            # Create the zip archive (without extension for shutil.make_archive)
            zip_path_without_ext = save_path[:-4]
            shutil.make_archive(
                zip_path_without_ext,
                'zip',
                self.dataset_folder
            )
            
            # Get file size
            file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
            
            self.log(f"✅ Dataset exported successfully!")
            self.log(f"   Archive: {os.path.basename(save_path)}")
            self.log(f"   Size: {file_size_mb:.2f} MB")
            self.log(f"   Location: {os.path.dirname(save_path)}")
            self.status_bar.showMessage(f"Dataset exported to: {save_path}", 10000)
            
        except Exception as e:
            self.log(f"❌ Export failed:")
            self.log(f"   {str(e)}")
            self.status_bar.showMessage(f"Export error: {str(e)}", 5000)
            print(f"Export error: {e}")
            import traceback
            traceback.print_exc()

    def generate_report(self):
        """Generate a comprehensive PDF report for the dataset"""
        # Check if report generation is available
        if not REPORTS_AVAILABLE:
            self.log("\n" + "="*50)
            self.log("⚠️ Report generation unavailable!")
            self.log("   Missing dependencies: matplotlib, seaborn, pandas")
            self.log("   Install with: pip install matplotlib seaborn pandas")
            self.status_bar.showMessage("Report dependencies not installed", 5000)
            return
        
        if not hasattr(self, 'dataset_folder') or not os.path.exists(self.dataset_folder):
            self.log("\n" + "="*50)
            self.log("⚠️ No dataset to generate report for!")
            self.log("   Please capture some images first.")
            self.status_bar.showMessage("No dataset folder found", 3000)
            return
        
        try:
            # Get folder for saving report
            folder_name = os.path.basename(self.dataset_folder)
            
            # Create folder selection dialog
            dialog = QFileDialog(self)
            dialog.setFileMode(QFileDialog.Directory)
            dialog.setOption(QFileDialog.ShowDirsOnly, True)
            dialog.setWindowTitle("Choose Report Destination Folder")
            dialog.setDirectory(os.path.expanduser("~"))
            
            # Apply custom stylesheet for better contrast
            dialog.setStyleSheet("""
                QFileDialog {
                    background-color: #ffffff;
                    color: #000000;
                }
                QTreeView, QListView {
                    background-color: #ffffff;
                    color: #000000;
                    selection-background-color: #0078d7;
                    selection-color: #ffffff;
                }
                QLineEdit {
                    background-color: #ffffff;
                    color: #000000;
                    border: 1px solid #cccccc;
                    padding: 5px;
                }
                QPushButton {
                    background-color: #f0f0f0;
                    color: #000000;
                    border: 1px solid #cccccc;
                    padding: 5px 15px;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    background-color: #e0e0e0;
                }
                QLabel {
                    color: #000000;
                }
            """)
            
            # Show dialog and get result
            if dialog.exec():
                export_dir = dialog.selectedFiles()[0]
            else:
                export_dir = None
            
            # User cancelled
            if not export_dir:
                self.log("\n" + "="*50)
                self.log("⚠️ Report generation cancelled by user")
                return
            
            # Generate timestamped report filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(export_dir, f"dataset_report_{timestamp}.pdf")
            
            self.log("\n" + "="*50)
            self.log("📊 Generating dataset report...")
            self.log(f"   Analyzing dataset: {folder_name}")
            
            # Disable button during generation
            self.btn_report.setEnabled(False)
            self.btn_report.setText("⏳ Generating...")
            
            # Generate report (this may take a while)
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents()  # Keep UI responsive
            
            report_path = generate_dataset_report(self.dataset_folder, report_path)
            
            self.log(f"✅ Report generated successfully!")
            self.log(f"   File: {os.path.basename(report_path)}")
            self.log(f"   Location: {export_dir}")
            self.log(f"   Dataset: {folder_name}")
            self.status_bar.showMessage(f"Report saved: {report_path}", 10000)
            
        except Exception as e:
            self.log(f"❌ Report generation failed:")
            self.log(f"   {str(e)}")
            self.status_bar.showMessage(f"Report error: {str(e)}", 5000)
            print(f"Report generation error: {e}")
            import traceback
            traceback.print_exc()
        
            self.btn_report.setText("📊 Generate Report")

    def clean_outliers(self):
        """Clean potential outliers from the dataset"""
        if not REPORTS_AVAILABLE:
            self.log("\n" + "="*50)
            self.log("⚠️ Cleaning requires analysis dependencies!")
            self.log("   Missing: matplotlib, seaborn, pandas")
            return

        if not hasattr(self, 'dataset_folder') or not os.path.exists(self.dataset_folder):
            self.status_bar.showMessage("No dataset to clean", 3000)
            return

        try:
            self.log("\n" + "="*50)
            self.log("🧹 Analyzing dataset for outliers...")
            self.btn_clean.setEnabled(False)
            self.btn_clean.setText("Analysing...")
            
            from PySide6.QtWidgets import QApplication, QMessageBox
            QApplication.processEvents()

            # Initialize generator to use its analysis methods
            generator = DatasetReportGenerator(self.dataset_folder)
            generator.analyze_dataset()
            
            if not generator.outliers:
                self.log("✅ No significant outliers found.")
                self.log("   Dataset stats are consistent.")
                QMessageBox.information(self, "Clean Dataset", "No outliers detected.\nTraining set looks consistent!")
                return
            
            # Found outliers
            count = len(generator.outliers)
            self.log(f"⚠️ Found {count} potential outliers.")
            
            # Ask for confirmation
            msg = f"Found {count} images that deviate significantly from the mean.\n\n"
            msg += "These may pollute the PatchCore memory bank.\n"
            msg += "Do you want to move them to the 'debug' folder?"
            
            reply = QMessageBox.question(self, "Found Outliers", msg, 
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                self.log(f"📦 Moving {count} outliers to 'debug' folder...")
                self.btn_clean.setText("Moving files...")
                QApplication.processEvents()
                
                moved = generator.move_outliers_to_debug()
                
                self.log(f"✅ Moved {moved} images to debug.")
                self.log("   Training set is now cleaner.")
                
                # Update stats display
                # We subtract moved files from good count and add to bad count
                # Note: this is an approximation as we don't know if they were originally 'good'
                # But typically they are in 'train' so they were counted as good
                self.total_good_cans -= moved
                self.total_bad_cans += moved
                
                # Update text
                if self.total_cans_processed > 0:
                    success_rate = int((self.total_good_cans / self.total_cans_processed) * 100)
                else:
                    success_rate = 0
                
                self.lbl_status.setText(
                    f"Captures: {self.captured_count} (avg {avg_cans:.1f} cans/capture)\n"
                    f"Total cans: {self.total_cans_processed}\n"
                    f"Quality: {self.total_good_cans} good / {self.total_bad_cans} bad ({success_rate}%)\n"
                    f"Session time: {session_time}\n"
                    f"Dataset: {os.path.basename(self.dataset_folder)}"
                )
            

                
            else:
                self.log("ℹ️ Clean operation cancelled.")
                
        except Exception as e:
            self.log(f"❌ Error cleaning outliers: {e}")
            print(f"Clean error: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            self.btn_clean.setEnabled(True)
            self.btn_clean.setText("🧹 Clean Outliers")


    def go_back(self):
        self.close()
        if self.dashboard:
            self.dashboard.show()

    def closeEvent(self, event):
        self.is_capturing = False
        if self.camera_thread:
            self.camera_thread.stop()
        if self.camera:
            self.camera.release()
        event.accept()

    def add_new_sku(self):
        """Open dialog to add a new SKU configuration"""
        dialog = AddNewSKUDialog(self)
        if dialog.exec():
            data = dialog.get_data()
            name = data['name']
            model_path = data['model_path']
            ref_path = data['reference_path']
            
            if not name or not ref_path:
                self.status_bar.showMessage("Error: Name and Reference Image are required", 5000)
                return
            
            # Add to manager
            self.sku_manager.add_sku(name, model_path, ref_path)
            
            # Update combo box
            self.sku_combo.addItem(name)
            
            # Select the new SKU
            self.sku_combo.setCurrentText(name)
            
            self.log("\n" + "="*50)
            self.log(f"✅ New SKU added: {name}")
            self.log(f"   Ref: {os.path.basename(ref_path)}")
            if model_path:
                self.log(f"   Model: {os.path.basename(model_path)}")

    def on_sku_changed(self, text):
        """Handle SKU selection change"""
        if not text: return
        
        # Update active SKU in manager
        if self.sku_manager.set_active_sku_by_name(text):
            self.log("\n" + "="*50)
            self.log(f"🔄 Switched SKU to: {text}")
            
            # Reload processors using new configuration
            self._init_processors()
        else:
            self.log(f"❌ Failed to switch SKU: {text}")
