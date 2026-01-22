import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QLabel, QPushButton, QStatusBar, QFrame, QGroupBox,
                               QListWidget, QListWidgetItem, QDoubleSpinBox, QSplitter, 
                               QGridLayout, QDialog, QProgressBar, QTreeView, QFileSystemModel, QHeaderView)
from PySide6.QtCore import Qt, QTimer, QSize, QThread, Signal, QDir
from PySide6.QtGui import QImage, QPixmap, QFont, QColor
from ..camera import Camera
from .camera_thread import CameraThread
from ..can_process_img.align_can import CanAligner
from ..inference.patchcore_inference_v2 import PatchCoreInferencer
from ..can_process_img.detect_corner import CornerDetector
from ..can_process_img.rectified_sheet import SheetRectifier
import json
import os
from datetime import datetime
from ..can_process_img.crop_cans import CanCropper
from ..can_process_img.resize_can import CanResizer
from ..can_process_img.nomrmalize_can import prepare_for_autoencoder
import cv2
import numpy as np
from datetime import datetime
import os

class InspectionWorker(QThread):
    finished = Signal(object, dict, list, list, object) # qt_image, stats, logs, results_data, clean_image
    error = Signal(str)
    progress = Signal(str)
    # Changed: Added list of scores to the signal
    progressive_update = Signal(object, int, int, list)  # (partial_image, current_can, total_cans, current_scores)
    
    def __init__(self, frame, detector, rectifier, cropper, aligner, inferencer, resizer, normalizer):
        super().__init__()
        self.frame = frame
        self.detector = detector
        self.rectifier = rectifier
        self.cropper = cropper
        self.aligner = aligner
        self.inferencer = inferencer
        self.resizer = resizer
        self.normalizer = normalizer
        
    def run(self):
        try:
            # Create defects directory if it doesn't exist
            # Create defects directory if it doesn't exist
            # Generate timestamp for this inspection
            from datetime import datetime
            start_time_dt = datetime.now()
            year_dir = start_time_dt.strftime("%Y")
            month_dir = start_time_dt.strftime("%m")
            
            defects_base = 'defects'
            defects_dir = os.path.join(defects_base, year_dir, month_dir)
            os.makedirs(defects_dir, exist_ok=True)
            timestamp = start_time_dt.strftime("%Y%m%d_%H%M%S")
            start_time_str = start_time_dt.strftime("%H:%M:%S")
            
            check_img = self.frame.copy()
            
            # --- Step 1: Detect Corners ---
            self.progress.emit("Detecting Corners...")
            corners = self.detector.detect(check_img)
            
            # --- Step 2: Rectify Sheet ---
            self.progress.emit("Rectifying Sheet...")
            rectified = self.rectifier.get_warped(check_img, corners)
            if rectified is None:
                self.error.emit("Rectification Failed")
                return

            # --- Step 3: Crop Cans ---
            self.progress.emit("Cropping Cans...")
            cans = self.cropper.crop_cans(rectified)
            if not cans:
                self.error.emit("No cans detected")
                return
            
            self.progress.emit(f"Inspecting {len(cans)} cans...")
                
            # Copy rectified for annotation
            annotated_sheet = rectified.copy()
            
            # Track Batch Statistics
            batch_ok = 0
            batch_ng = 0
            batch_quality = 0
            batch_defect = 0
            logs = []
            results_data = []
            defects_saved = 0
            
            # --- Step 4: Process Each Can ---
            import time
            
            for i, item in enumerate(cans):
                can_id = item['id']
                can_img = item['image']
                bbox = item['bbox']
                
                can_start_time = time.time()
                
                # A. Resize
                resized_can = self.resizer.process(can_img)
                
                # B. Align (REQUIRED for accurate scores)
                if self.aligner:
                    aligned_can = self.aligner.align(resized_can)
                else:
                    aligned_can = resized_can
                
                # C. Normalize with CLAHE
                normalized_can = self.normalizer(aligned_can, target_size=(448, 448))
                
                # DEBUG: Log/Save if it's one of the first 3 cans OR specifically can 1
                if i < 3 or can_id == 1:
                    cv2.imwrite(f'debug_pi_can{can_id:02d}_1_raw_crop.png', can_img)
                    cv2.imwrite(f'debug_pi_can{can_id:02d}_2_resized.png', resized_can)
                    cv2.imwrite(f'debug_pi_can{can_id:02d}_3_aligned.png', aligned_can)
                    cv2.imwrite(f'debug_pi_can{can_id:02d}_4_normalized.png', normalized_can)
                    
                    print(f"\n=== CAN {can_id} PREPROCESSING DEBUG ===")
                    print(f"1. Raw crop    - Shape: {can_img.shape}, Mean: {np.mean(can_img):.2f}")
                    print(f"2. Resized     - Shape: {resized_can.shape}, Mean: {np.mean(resized_can):.2f}")
                    print(f"3. Aligned     - Shape: {aligned_can.shape}, Mean: {np.mean(aligned_can):.2f}")
                    print(f"4. Normalized  - Shape: {normalized_can.shape}, Mean: {np.mean(normalized_can):.2f}")
                    print(f"💾 Saved debug output for Can {can_id}\n")
                
                # Quality Check: Brightness threshold (too bright = bad crop/missing can)
                brightness = np.mean(normalized_can)
                brightness_threshold = 180  # Increased to 180 for 448x448 model
                # print(f"DEBUG: Can {can_id} Brightness: {brightness:.2f}") # Debug brightness
                
                heatmap = None # Initialize heatmap
                
                if brightness > brightness_threshold:
                    # Too bright - likely bad crop or missing can
                    batch_quality += 1
                    batch_ng += 1
                    color = (255, 0, 0)  # Blue bbox for quality failure (BGR format)
                    status_text = "NOK-QUALITY"
                    score = 999.0  # Special score for quality failures
                    is_normal = False
                    can_duration = (time.time() - can_start_time) * 1000
                    
                    # Save quality failure image
                    defect_filename = f"{defects_dir}/NOK_QUALITY_{timestamp}_can{can_id:02d}_bright{brightness:.1f}.png"
                    cv2.imwrite(defect_filename, normalized_can)
                    defects_saved += 1
                else:
                    # D. Inference (only if quality check passes)
                    if self.inferencer is None:
                        self.error.emit("PatchCore model not loaded!")
                        return
                        
                    # Use aligned_can (Inferencer V2 handles CLAHE+Resize+Norm internaly)
                    score, is_normal, heatmap = self.inferencer.predict(aligned_can)
                    print(f"DEBUG: Can #{can_id} Score: {score:.8f}")
                    
                    can_duration = (time.time() - can_start_time) * 1000
                    
                    # Update Stats
                    if is_normal:
                        batch_ok += 1
                        color = (0, 255, 0)
                        status_text = "OK"
                    else:
                        batch_defect += 1
                        batch_ng += 1
                        color = (0, 0, 255)
                        status_text = "NG"
                        
                        # Save NOK image
                        defect_filename = f"{defects_dir}/NOK_{timestamp}_can{can_id:02d}_score{score:.2f}.png"
                        cv2.imwrite(defect_filename, normalized_can)
                        defects_saved += 1
                
                # Draw Result with timing
                x1, y1, x2, y2 = bbox
                cv2.rectangle(annotated_sheet, (x1, y1), (x2, y2), color, 2)
                
                label = f"#{can_id} {status_text} {score:.2f} ({can_duration:.0f}ms)"
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(annotated_sheet, (x1, y1 - 20), (x1 + text_w, y1), color, -1)
                text_color = (0, 0, 0) if is_normal else (255, 255, 255)
                cv2.putText(annotated_sheet, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
                
                logs.append((can_id, score, is_normal, can_duration))
                
                results_data.append({
                    'can_id': can_id,
                    'bbox': bbox,
                    'score': score,
                    'image': normalized_can,
                    'heatmap': heatmap
                })
                
                # Emit progressive update for EVERY can
                rgb_partial = cv2.cvtColor(annotated_sheet, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_partial.shape
                bytes_per_line = ch * w
                qt_partial = QImage(rgb_partial.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
                
                current_scores = [log[1] for log in logs]
                self.progressive_update.emit(qt_partial, i + 1, len(cans), current_scores)
            
            if defects_saved > 0:
                print(f"💾 Saved {defects_saved} defect images to '{defects_dir}/'")
            
            stats = {
                'total': len(cans),
                'ok': batch_ok,
                'ng': batch_ng,
                'defect': batch_defect,
                'quality': batch_quality,
                'start_time': start_time_str
            }
            
            # Convert to RGB for UI
            rgb_sheet = cv2.cvtColor(annotated_sheet, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_sheet.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_sheet.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
            
            self.finished.emit(qt_image, stats, logs, results_data, rectified)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

class LogsDialog(QDialog):
    def __init__(self, log_list_widget, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Registos de Inspeção")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        self.setStyleSheet("background-color: #0D1117; color: #E6EDF3;")
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Histórico de Inspeções")
        title.setStyleSheet("font-weight: bold; font-size: 16px; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # List Widget
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet("""
            QListWidget {
                background-color: #0d1117;
                border: 1px solid #30363D;
                border-radius: 6px;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #21262d;
            }
        """)
        layout.addWidget(self.list_widget)
        
        # Clone items
        for i in range(log_list_widget.count()):
            src_item = log_list_widget.item(i)
            new_item = QListWidgetItem(src_item.text())
            new_item.setFont(src_item.font())
            new_item.setForeground(src_item.foreground())
            new_item.setBackground(src_item.background())
            self.list_widget.addItem(new_item)
            
        # Close Button
        btn_close = QPushButton("Fechar")
        btn_close.setCursor(Qt.PointingHandCursor)
        btn_close.setStyleSheet("""
            QPushButton {
                background-color: #238636;
                color: white;
                padding: 8px 16px;
                border: none;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #2EA043; }
        """)
        btn_close.clicked.connect(self.accept)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)

class DefectsGalleryDialog(QDialog):
    def __init__(self, defects_dir="defects", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Galeria de Defeitos")
        self.resize(1100, 700)
        
        # Ensure absolute path for QFileSystemModel
        self.defects_dir = os.path.abspath(defects_dir)
        if not os.path.exists(self.defects_dir):
            os.makedirs(self.defects_dir)
            
        self.setStyleSheet("""
            QDialog { background-color: #0D1117; color: #E6EDF3; }
            QTreeView {
                background-color: #161B22; 
                border: 1px solid #30363D;
                border-radius: 6px;
                color: #E6EDF3;
                font-family: 'Segoe UI', sans-serif;
            }
            QTreeView::item { 
                padding: 4px; 
                border-bottom: 0px; 
            }
            QTreeView::item:selected { 
                background-color: #1F6FEB; 
                color: white; 
            }
            QTreeView::item:hover { 
                background-color: #21262D; 
            }
            QHeaderView::section {
                background-color: #161B22;
                color: #8B949E;
                padding: 4px;
                border: 1px solid #30363D;
            }
            QLabel { color: #E6EDF3; }
        """)
        
        layout = QHBoxLayout(self)
        
        # Left: File Tree
        left_panel = QVBoxLayout()
        left_panel.addWidget(QLabel("Navegação (Ano / Mês):"))
        
        self.file_model = QFileSystemModel()
        self.file_model.setRootPath(self.defects_dir)
        self.file_model.setFilter(QDir.NoDotAndDotDot | QDir.AllDirs | QDir.Files)
        self.file_model.setNameFilters(["*.png", "*.jpg", "*.jpeg"])
        self.file_model.setNameFilterDisables(False) # Hide files that don't match
        
        self.tree_view = QTreeView()
        self.tree_view.setModel(self.file_model)
        self.tree_view.setRootIndex(self.file_model.index(self.defects_dir))
        self.tree_view.setFixedWidth(400)
        
        # Hide unnecessary columns (Size, Type, Date) - keep Name
        self.tree_view.setColumnHidden(1, True)
        self.tree_view.setColumnHidden(2, True)
        self.tree_view.setColumnHidden(3, True)
        self.tree_view.header().setVisible(False)
        
        # Connect Selection
        self.tree_view.selectionModel().selectionChanged.connect(self.on_selection_changed)
        
        left_panel.addWidget(self.tree_view)
        layout.addLayout(left_panel)
        
        # Right: Image Preview
        right_panel = QVBoxLayout()
        self.image_label = QLabel("Selecione uma imagem")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #010409; border: 1px solid #30363D; border-radius: 6px;")
        
        # FIX: Prevent image from forcing window resize loop
        from PySide6.QtWidgets import QSizePolicy
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        
        right_panel.addWidget(self.image_label)
        layout.addLayout(right_panel)
        
    def on_selection_changed(self, selected, deselected):
        indexes = self.tree_view.selectionModel().selectedIndexes()
        if indexes:
            # Column 0 is the Name column
            index = indexes[0]
            file_path = self.file_model.filePath(index)
            
            if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.show_image(file_path)
            else:
                self.image_label.setText("Selecione uma imagem")
                self.image_label.setPixmap(QPixmap()) # Clear

    def show_image(self, path):
        if path and os.path.exists(path):
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                w = self.image_label.width()
                h = self.image_label.height()
                scaled = pixmap.scaled(QSize(w, h), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(scaled)
            else:
                self.image_label.setText("Erro ao carregar imagem.")
        else:
            self.image_label.setText("Ficheiro não encontrado.")
            
    def resizeEvent(self, event):
        # Refresh current image on resize
        indexes = self.tree_view.selectionModel().selectedIndexes()
        if indexes:
            index = indexes[0]
            file_path = self.file_model.filePath(index)
            if os.path.isfile(file_path):
                 self.show_image(file_path)
        super().resizeEvent(event)

class CanDetailDialog(QDialog):
    def __init__(self, can_id, score, status, image, heatmap, threshold, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Detalhes da Lata #{can_id}")
        self.setMinimumWidth(800)
        self.setStyleSheet("background-color: #0D1117; color: #E6EDF3;")
        
        layout = QHBoxLayout(self)
        
        # 1. Original Image
        img_layout = QVBoxLayout()
        lbl_img_title = QLabel("Imagem Processada")
        lbl_img_title.setAlignment(Qt.AlignCenter)
        lbl_img_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 5px;")
        
        self.lbl_image = QLabel()
        self.lbl_image.setFixedSize(300, 300)
        self.lbl_image.setStyleSheet("border: 1px solid #30363D; background-color: black;")
        self.lbl_image.setScaledContents(True)
        
        if image is not None:
             # Identify if RGB or BGR. normalized_can is usually BGR from cv2
             rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
             h, w, ch = rgb_img.shape
             qimg = QImage(rgb_img.data, w, h, ch * w, QImage.Format_RGB888)
             self.lbl_image.setPixmap(QPixmap.fromImage(qimg))
        else:
            self.lbl_image.setText("Sem Imagem")
            
        img_layout.addWidget(lbl_img_title)
        img_layout.addWidget(self.lbl_image)
        layout.addLayout(img_layout)
        
        # 2. Heatmap
        hm_layout = QVBoxLayout()
        lbl_hm_title = QLabel("Mapa de Calor (Defeitos)")
        lbl_hm_title.setAlignment(Qt.AlignCenter)
        lbl_hm_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 5px;")
        
        self.lbl_heatmap = QLabel()
        self.lbl_heatmap.setFixedSize(300, 300)
        self.lbl_heatmap.setStyleSheet("border: 1px solid #30363D; background-color: black;")
        self.lbl_heatmap.setScaledContents(True)
        
        if heatmap is not None:
            # Heatmap is likely float 0-1 or similar. Need to normalize and colorize.
            # Assuming heatmap is pre-processed or raw float map.
            try:
                # Normalize to 0-255
                hm_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
                hm_uint8 = hm_norm.astype(np.uint8)
                # Apply colormap (Jet or similar)
                hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
                
                rgb_hm = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_hm.shape
                qhm = QImage(rgb_hm.data, w, h, ch * w, QImage.Format_RGB888)
                self.lbl_heatmap.setPixmap(QPixmap.fromImage(qhm))
            except Exception as e:
                print(f"Error displaying heatmap: {e}")
                self.lbl_heatmap.setText("Erro Heatmap")
        else:
            self.lbl_heatmap.setText("N/D")
            
        hm_layout.addWidget(lbl_hm_title)
        hm_layout.addWidget(self.lbl_heatmap)
        layout.addLayout(hm_layout)
        
        # 3. Data Panel
        data_layout = QVBoxLayout()
        data_layout.setAlignment(Qt.AlignTop)
        
        def add_data_row(label, value, color="#E6EDF3"):
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setStyleSheet("color: #8B949E; font-weight: bold;")
            val = QLabel(str(value))
            val.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 16px;")
            row.addWidget(lbl)
            row.addWidget(val)
            row.addStretch()
            data_layout.addLayout(row)
            
        add_data_row("ID da Lata:", f"#{can_id}")
        
        score_color = "#2EA043" if score <= threshold else "#DA3633" # Green or Red
        add_data_row("Pontuação:", f"{score:.4f}", score_color)
        
        add_data_row("Limiar Atual:", f"{threshold:.1f}")
        add_data_row("Estado:", status, score_color)
        
        layout.addLayout(data_layout)

class InspectionWindow(QMainWindow):
    def __init__(self, config, user_manager, dashboard=None):
        super().__init__()
        self.config = config
        self.user_manager = user_manager
        self.dashboard = dashboard
        self.camera = None
        self.camera_thread = None
        
        # Inspection State
        self.total_count = 0
        self.ok_count = 0
        self.ng_count = 0
        self.defect_count = 0
        self.quality_count = 0
        self.current_frame = None
        self.last_clean_image = None # For live threshold updates
        self.last_results_data = []  # For live threshold updates
        self.is_running = False # Start/Stop state
        
        # Inspection Trigger Logic
        self.pending_inspection = False
        self.inspection_timeout_timer = QTimer(self)
        self.inspection_timeout_timer.setSingleShot(True)
        self.inspection_timeout_timer.timeout.connect(self.on_inspection_timeout)
        
        # Load Aligner
        self.aligner = None
        ref_path = 'models/can_reference/aligned_can_reference448.png'
        if os.path.exists(ref_path):
            try:
                # FIX: Set target_size to 448x448 to match PatchCore model
                self.aligner = CanAligner(ref_path, target_size=(448, 448))
                print(f"Aligner loaded with reference: {ref_path}")
            except Exception as e:
                print(f"Failed to load aligner: {e}")
        else:
            print(f"Reference image not found at {ref_path}")



        # Load Inferencer (PatchCore)
        self.inferencer = None
        try:
            self.inferencer = PatchCoreInferencer()
            print(f"✓ PatchCore Inferencer loaded. Threshold: {self.inferencer.threshold}")
        except Exception as e:
            print(f"Failed to load PatchCore: {e}")
            from PySide6.QtWidgets import QMessageBox
            
        # Load Sheet Processors
        self.detector = CornerDetector()
        self.detector.load_params()
        
        self.rectifier = SheetRectifier()
        self.rectifier.load_params()
        
        self.cropper = CanCropper()
        self.cropper.load_params()
        
        # Load Can Resizer - 448x448 for alignment input
        self.resizer = CanResizer(size=(448, 448))
        print("Can Resizer initialized (448x448 → CanAligner → 448x448)")
            # We can't show message box here easily as init might be early, 
            # but we will handle it in trigger
        
        self.setWindowTitle("Inspection Dashboard")
        self.setFocusPolicy(Qt.StrongFocus) # Ensure window can catch key events
        self.resize(1280, 720)
        
        self._setup_ui()
        # Delay camera start to allow UI to show (and status bar to be visible during warmup)
        QTimer.singleShot(500, self._start_camera)
        self.showMaximized()
        
    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Apply Premium Dark Theme
        central_widget.setStyleSheet("""
            QWidget {
                background-color: transparent; 
                color: #E6EDF3;
                font-family: 'Segoe UI', sans-serif;
            }
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1C232B, stop:1 #0F141A);
            }
            /* Cards / Panels */
            QGroupBox {
                background-color: #1A2129;
                border: 1px solid #2E3A46;
                border-top: 1px solid #3E4C59; /* Highlight discreto no topo */
                border-radius: 8px;
                margin-top: 24px; /* Space for title */
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                padding: 0 5px;
                color: #58A6FF; /* Technical Blue Text */
                background-color: transparent;
            }
            /* Buttons */
            QPushButton {
                background-color: #212830;
                color: #E6EDF3;
                border: 1px solid #30363D;
                padding: 6px 12px;
                border-radius: 6px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #262C36;
                border-color: #8B949E;
            }
            QPushButton:pressed {
                background-color: #161B22;
                border-color: #2E3A46;
            }
            /* Lists & Inputs */
            QListWidget {
                background-color: #0D1117;
                border: 1px solid #30363D;
                border-radius: 6px;
                outline: none;
            }
            QDoubleSpinBox {
                background-color: #0D1117;
                border: 1px solid #30363D;
                border-radius: 6px;
                padding: 4px;
                color: #E6EDF3;
                selection-background-color: #1F6FEB;
            }
            QLabel {
                color: #E6EDF3;
            }
            /* Splitter Handle */
            QSplitter::handle {
                background-color: transparent;
                height: 1px;
                width: 12px; 
            }
        """)
        
        # Set main window background explicitly for gradient to work
        self.setStyleSheet("QMainWindow { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1C232B, stop:1 #0F141A); }")

        # Header
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 10)
        
        btn_back = QPushButton("← Painel")
        btn_back.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                color: #8B949E;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.1);
                color: #E6EDF3;
            }
        """)
        btn_back.clicked.connect(self.go_back)
        header.addWidget(btn_back)
        
        title = QLabel("INSPEÇÃO")
        title.setStyleSheet("font-size: 20px; font-weight: 700; color: #E6EDF3; letter-spacing: 1px;")
        title.setAlignment(Qt.AlignCenter)
        header.addWidget(title, stretch=1)
        
        # Date/Time
        self.lbl_time = QLabel()
        self.lbl_time.setStyleSheet("font-size: 14px; color: #8B949E; font-family: monospace;")
        header.addWidget(self.lbl_time)
        main_layout.addLayout(header)
        
        # Timer for clock
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_clock)
        self.timer.start(1000)
        self.update_clock()

        from PySide6.QtWidgets import QSplitter, QSizePolicy

        # Middle Section using QSplitter for responsiveness
        middle_splitter = QSplitter(Qt.Horizontal)
        
        # --- LEFT PANEL (Controls & KPIs) ---
        left_widget = QWidget()
        left_widget.setMinimumWidth(150) # Allow shrinking but keep usable
        # left_widget.setMaximumWidth(300) # Optional: don't let it get too huge
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # 1. Start/Stop Controls
        controls_layout = QHBoxLayout()
        
        self.btn_start = QPushButton("INICIAR")
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #238636;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
                border: 1px solid #2EA043;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #2EA043; }
            QPushButton:disabled { background-color: #1F2428; color: #484F58; border: 1px solid #30363D; }
        """)
        self.btn_start.clicked.connect(self.start_inspection_mode)
        
        self.btn_stop = QPushButton("PARAR")
        self.btn_stop.setCursor(Qt.PointingHandCursor)
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #DA3633;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
                border: 1px solid #F85149;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #F85149; }
            QPushButton:disabled { background-color: #1F2428; color: #484F58; border: 1px solid #30363D; }
        """)
        self.btn_stop.clicked.connect(self.stop_inspection_mode)
        self.btn_stop.setEnabled(False)
        
        controls_layout.addWidget(self.btn_start)
        controls_layout.addWidget(self.btn_stop)
        left_layout.addLayout(controls_layout)
        

        
        # KPI Cards (Vertical Stack)
        kpi_layout = QVBoxLayout()
        kpi_layout.setSpacing(12)
        # Using requested palette:
        self.kpi_total = self._create_kpi_card("TOTAL", "0", "#21262D") # Dark Neutral
        self.kpi_ok = self._create_kpi_card("OK", "0", "#1C3323")    # Dark Green Tint #2EA043 accent
        # Just tint the background slightly, use border for accent
        
        # Let's customize create_kpi_card instead to be smarter
        self.kpi_ng = self._create_kpi_card("NG", "0", "#3E2020")    # Dark Red Tint
        self.kpi_yield = self._create_kpi_card("Perc. OK", "0%", "#382810") # Dark Orange Tint
        
        kpi_layout.addWidget(self.kpi_total)
        kpi_layout.addWidget(self.kpi_ok)
        kpi_layout.addWidget(self.kpi_ng)
        kpi_layout.addWidget(self.kpi_yield)
        left_layout.addLayout(kpi_layout)
        
        # Donut Chart for Breakdown
        donut_group = QGroupBox("Distribuição")
        donut_layout = QVBoxLayout()
        donut_layout.setContentsMargins(0, 5, 0, 5)
        
        self.fig_donut = Figure(figsize=(3, 3), dpi=100)
        self.fig_donut.subplots_adjust(left=0, right=1, top=1, bottom=0) # Maximize space
        self.fig_donut.patch.set_alpha(0) # Transparent background
        
        self.canvas_donut = FigureCanvas(self.fig_donut)
        self.canvas_donut.setStyleSheet("background-color: transparent;")
        
        self.ax_donut = self.fig_donut.add_subplot(111)
        self.ax_donut.patch.set_alpha(0) # Transparent axes
        self.ax_donut.axis('off') # Hide axes
        
        donut_layout.addWidget(self.canvas_donut)
        donut_group.setLayout(donut_layout)
        
        left_layout.addWidget(donut_group)
        
        # Defects Gallery Button
        self.btn_defects = QPushButton("📂 Ver Defeitos")
        self.btn_defects.setCursor(Qt.PointingHandCursor)
        self.btn_defects.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                color: #C9D1D9;
                padding: 8px;
                border-radius: 6px;
                margin-top: 10px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.1);
                color: white;
            }
        """)
        self.btn_defects.clicked.connect(self.open_defects_gallery)
        left_layout.addWidget(self.btn_defects)
        
        
        left_layout.addStretch() # Push everything up
        
        # --- CENTER PANEL (Camera) ---
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        
        camera_group = QGroupBox("Vista em Tempo Real / Resultado")
        cam_inner_layout = QVBoxLayout()
        
        self.video_label = QLabel("AGUARDA INÍCIO")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white; border: 2px solid #444;")
        self.video_label.setScaledContents(True)
        # FIX: Set SizePolicy to Ignored to prevent image from forcing window expansion
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        cam_inner_layout.addWidget(self.video_label)
        
        # Warmup Progress Bar (Hidden by default)
        self.warmup_progress = QProgressBar()
        self.warmup_progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #30363D;
                border-radius: 5px;
                text-align: center;
                background-color: #0D1117;
                color: #E6EDF3;
                font-weight: bold;
                height: 30px;
            }
            QProgressBar::chunk {
                background-color: #238636;
                width: 20px;
            }
        """)
        self.warmup_progress.setValue(0)
        self.warmup_progress.setFormat("A AQUECER: %p%")
        self.warmup_progress.hide()
        cam_inner_layout.addWidget(self.warmup_progress)
        
        camera_group.setLayout(cam_inner_layout)
        center_layout.addWidget(camera_group)
        
        # --- RIGHT PANEL (Stats & Status) ---
        right_widget = QWidget()
        right_widget.setMinimumWidth(150)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # 1. Statistics Group
        stats_group = QGroupBox("Estatísticas")
        stats_layout = QGridLayout()
        stats_layout.setVerticalSpacing(8)
        stats_layout.setColumnStretch(1, 1) # Values on right take remaining space

        def add_stat_row(row, label_text, initial_value):
            lbl = QLabel(label_text)
            lbl.setStyleSheet("color: #8B949E; font-size: 13px;")
            val = QLabel(initial_value)
            val.setStyleSheet("color: #E6EDF3; font-size: 13px; font-weight: bold;")
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            stats_layout.addWidget(lbl, row, 0)
            stats_layout.addWidget(val, row, 1)
            return val

        self.lbl_stats_total = add_stat_row(0, "Total Latas:", "0")
        self.lbl_stats_start = add_stat_row(1, "Início:", "--/-- --:--")
        self.lbl_stats_end   = add_stat_row(2, "Fim:", "--/-- --:--")
        self.lbl_stats_avg_time = add_stat_row(3, "Média Tempo:", "--ms")
        self.lbl_stats_good_pct = add_stat_row(4, "Bons:", "0%")
        self.lbl_stats_bad_pct  = add_stat_row(5, "Maus:", "0%")
        
        stats_group.setLayout(stats_layout)
        right_layout.addWidget(stats_group)
        
        # 2. Graph Group (Analysis)
        graph_group = QGroupBox("Análise")
        graph_layout = QVBoxLayout()
        
        # Threshold Control (Moved from Left Panel)
        thresh_layout = QHBoxLayout()
        thresh_layout.setContentsMargins(0, 0, 0, 5)
        
        thresh_label = QLabel("Limiar Máx:")
        thresh_label.setStyleSheet("color: #8B949E; font-size: 12px; margin-right: 5px;")
        
        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setRange(0.0, 200.0)
        self.threshold_spinbox.setSingleStep(0.5)
        self.threshold_spinbox.setDecimals(1)
        
        # Load persisted threshold
        saved_threshold = self.load_config()
        self.threshold_spinbox.setValue(saved_threshold)
        if self.inferencer:
            self.inferencer.threshold = saved_threshold
            
        self.threshold_spinbox.valueChanged.connect(self.update_threshold)
        
        # Access Control: Disable threshold adjustment for Operators
        if self.user_manager.current_user.role not in ['admin', 'tecnico']:
            self.threshold_spinbox.setEnabled(False)
            self.threshold_spinbox.setToolTip("Apenas Master/Técnico podem ajustar o limiar.")
        
        thresh_layout.addWidget(thresh_label)
        thresh_layout.addWidget(self.threshold_spinbox)
        thresh_layout.addStretch()
        graph_layout.addLayout(thresh_layout)
        
        # Initialize Matplotlib Figure
        self.figure = Figure(figsize=(5, 3), dpi=100)
        self.figure.patch.set_facecolor('#0D1117') # Dark background
        
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: #0D1117; border-radius: 4px;")
        
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#0D1117')
        
        # Initial empty plot styling
        self.ax.spines['bottom'].set_color('none')
        self.ax.spines['top'].set_color('none') 
        self.ax.spines['right'].set_color('none')
        self.ax.spines['left'].set_color('none')
        self.ax.grid(False)
        
        self.ax.tick_params(axis='x', colors='#8B949E', length=0)
        self.ax.tick_params(axis='y', colors='#8B949E', length=0)
        self.ax.yaxis.label.set_color('#8B949E')
        self.ax.xaxis.label.set_color('#8B949E')
        
        # Initial X Axis setup 1-50
        self.ax.set_xlim(0.5, 50.5)
        
        self.ax.set_title("Pontuações", color='#E6EDF3', fontsize=10)
        
        graph_layout.addWidget(self.canvas)
        graph_group.setLayout(graph_layout)
        right_layout.addWidget(graph_group, stretch=1) # Graph takes available space
        
        # 3. System Status Group
        status_group = QGroupBox("Estado do Sistema")
        status_layout = QVBoxLayout()
        
        self.lbl_status_camera = QLabel("📷 Câmara: OK")
        self.lbl_status_camera.setStyleSheet("color: #2EA043; font-weight: bold;") # Green
        self.lbl_status_model = QLabel("🧠 Modelo IA: OK")
        self.lbl_status_model.setStyleSheet("color: #2EA043; font-weight: bold;")
        self.lbl_status_plcc = QLabel("🔌 PLC: N/D")
        self.lbl_status_plcc.setStyleSheet("color: #8B949E;") # Grey
        
        status_layout.addWidget(self.lbl_status_camera)
        status_layout.addWidget(self.lbl_status_model)
        status_layout.addWidget(self.lbl_status_plcc)
        status_group.setLayout(status_layout)
        right_layout.addWidget(status_group)
        
        # 4. Log Button
        self.btn_logs = QPushButton("📄 Ver Registos")
        self.btn_logs.setCursor(Qt.PointingHandCursor)
        # Inherits global style, maybe add specific emphasis if needed
        self.btn_logs.setStyleSheet("border: 1px solid #30363D; color: #8B949E;") 
        self.btn_logs.clicked.connect(self.open_logs_dialog)
        right_layout.addWidget(self.btn_logs)
        
        # Access Control: Hide Logs and Defects for Operators
        # Only Master and Tecnico can see these sensitive debugging/historical tools
        if self.user_manager.current_user.role not in ['admin', 'tecnico']:
            self.btn_logs.hide()
            self.btn_defects.hide()

        # Hidden Log List (Keep object alive for logic compatibility)
        self.log_list = QListWidget() 
        self.log_list.setVisible(False) 
        
        # Add Widgets to Splitter
        middle_splitter.addWidget(left_widget)
        middle_splitter.addWidget(center_widget)
        middle_splitter.addWidget(right_widget)
        
        # Set initial sizes: Widen Right Panel (Statistics/Graph), Decrease Center
        # [Left, Center, Right] -> [220, 660, 400]
        middle_splitter.setSizes([220, 660, 400])
        
        # Add Splitter to Main Layout
        main_layout.addWidget(middle_splitter, stretch=1)
        
        # --- BOTTOM PANEL (Status/Footer) - Fixed Height ---
        bottom_widget = QWidget()
        bottom_widget.setFixedHeight(60) # Reduced height
        bottom_widget.setStyleSheet("background-color: rgba(22, 27, 34, 0.8); border-top: 1px solid #30363D;")
        bottom_layout = QHBoxLayout(bottom_widget)
        
        # Placeholder content for bottom panel
        self.lbl_system_status = QLabel("● SISTEMA PRONTO")
        self.lbl_system_status.setStyleSheet("font-size: 14px; font-weight: bold; color: #2EA043; letter-spacing: 2px;")
        bottom_layout.addWidget(self.lbl_system_status)
        
        bottom_layout.addStretch()
        
        lbl_version = QLabel("v1.0.0")
        lbl_version.setStyleSheet("color: #484F58;")
        bottom_layout.addWidget(lbl_version)
        
        main_layout.addWidget(bottom_widget)
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def _create_kpi_card(self, title, value, bg_color):
        frame = QFrame()
        
        # Determine border color based on bg_color approximation or use a default
        # Simple mapping for now
        border_color = "#30363D"
        text_color = "#E6EDF3"
        value_color = "#FFFFFF"
        
        if "2EA043" in bg_color or "1C3323" in bg_color: # Green
             border_color = "rgba(46, 160, 67, 0.4)"
             value_color = "#2EA043"
        elif "DA3633" in bg_color or "3E2020" in bg_color: # Red
             border_color = "rgba(218, 54, 51, 0.4)"
             value_color = "#DA3633"
        elif "F0883E" in bg_color or "382810" in bg_color: # Orange
             border_color = "rgba(240, 136, 62, 0.4)"
             value_color = "#F0883E"
             
        frame.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 8px;
            }}
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(10, 10, 10, 10)
        
        lbl_title = QLabel(title)
        lbl_title.setAlignment(Qt.AlignLeft)
        lbl_title.setStyleSheet("font-size: 11px; font-weight: 600; color: #8B949E; text-transform: uppercase; border: none; background: transparent;")
        
        lbl_value = QLabel(value)
        lbl_value.setAlignment(Qt.AlignLeft)
        lbl_value.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {value_color}; border: none; background: transparent;")
        lbl_value.setObjectName("value")
        
        layout.addWidget(lbl_title)
        layout.addWidget(lbl_value)
        frame.setLayout(layout)
        return frame

    def update_kpi(self, card, value):
        # Find the value label inside the card layout
        layout = card.layout()
        if layout:
            for i in range(layout.count()):
                widget = layout.itemAt(i).widget()
                if widget and widget.objectName() == "value":
                    widget.setText(str(value))
                    break

    def update_clock(self):
        now = datetime.now()
        self.lbl_time.setText(now.strftime("%Y-%m-%d %H:%M:%S"))

    def _start_camera(self):
        try:
            self.camera = Camera(
                camera_index=self.config['camera']['index'],
                width=self.config['camera']['width'],
                height=self.config['camera']['height'],
                fps=self.config['camera']['fps']
            )
            
            # Use lower res for smooth preview, but we might want high res for inspection
            # If we want high res capture, we should set use_high_res=True
            self.camera_thread = CameraThread(self.camera, target_size=(640, 480))
            self.camera_thread.use_high_res = True # FORCE HIGH RES for Inspection
            self.camera_thread.frame_captured.connect(self.update_frame)
            # MOVED start() to AFTER warmup to prevent resource contention
            # self.camera_thread.start() 
            self.camera.load_parameters()
            
            # Warmup to settle Auto-Exposure
            self.status_bar.showMessage("A aquecer câmara...", 0)
            
            # Show Progress Bar
            self.warmup_progress.show()
            self.warmup_progress.setValue(0)
            
            from PySide6.QtWidgets import QApplication
            
            def update_progress(current, total):
                pct = int((current / total) * 100)
                self.warmup_progress.setValue(pct)
                self.warmup_progress.setFormat(f"A AQUECER: {pct}%")
                QApplication.processEvents() # Keep UI alive
                
            self.camera.warmup(30, callback=update_progress)
            
            # NOW start the thread safely
            self.camera_thread.start()
            
            self.warmup_progress.hide()
            self.status_bar.showMessage("Câmara Pronta", 2000)
        except Exception as e:
            self.video_label.setText(f"Camera Error: {str(e)}")
            self.status_bar.showMessage(f"Error: {str(e)}")

    def update_frame(self, qt_image, sharpness, raw_frame):
        # Update internal frame for capture, but DO NOT update UI (Live View Disabled)
        if self.camera_thread and not self.camera_thread.paused:
            # print("DEBUG: update_frame received") # Too spammy, rely on thread log
            self.current_frame = raw_frame
            
            # Check if we are waiting for a frame to inspect
            # Check if we are waiting for a frame to inspect
            if self.pending_inspection:
                self.pending_inspection = False
                self.inspection_timeout_timer.stop()
                self._start_inspection()

            # Optional: Show a static "Ready" placeholder if not already set? 
            # For now, we just don't show the live stream.




    def start_inspection_mode(self):
        self.is_running = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status_bar.showMessage("SISTEMA EM EXECUÇÃO - Pressione 'T' para Inspecionar", 0)
        
        # Set Start Timestamp
        now = datetime.now()
        self.lbl_stats_start.setText(now.strftime('%d/%m/%Y %H:%M'))
        self.lbl_stats_end.setText("--/-- --:--")
        
        # Force focus back to main window so KeyPressEvent works immediately
        self.setFocus()
        
    def stop_inspection_mode(self):
        self.is_running = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_bar.showMessage("SISTEMA PARADO", 0)
        
        # Set Stop Timestamp
        now = datetime.now()
        self.lbl_stats_end.setText(now.strftime('%d/%m/%Y %H:%M'))

    def keyPressEvent(self, event):
        # Trigger inspection with 'T' key only if running
        if event.key() == Qt.Key_T:
            if self.is_running:
                self.trigger_inspection()
            else:
                self.status_bar.showMessage("AVISO: Sistema PARADO. Pressione INICIAR primeiro.", 2000)
        else:
            super().keyPressEvent(event)

    def trigger_inspection(self):
        """
        Capture current frame and trigger inspection for ALL cans.
        """
        if not self.is_running:
            self.status_bar.showMessage("Ignorado: Sistema PARADO.", 1000)
            return

        if self.camera_thread and self.camera_thread.running:
            # Check if inferencer is loaded
            if self.inferencer is None:
                self.status_bar.showMessage("ERROR: PatchCore model not loaded! Cannot inspect.", 5000)
                return
                
            # Briefly unpause to get fresh frame (camera keeps updating current_frame in background)
            # Briefly unpause to get fresh frame (camera keeps updating current_frame in background)
            self.camera_thread.paused = False
            
            # Set flag to capture NEXT available frame in update_frame
            self.pending_inspection = True
            
            # Start safety timeout (2 seconds)
            self.inspection_timeout_timer.start(2000) 
            self.status_bar.showMessage("A aguardar frame...", 2000)
            
    def on_inspection_timeout(self):
        """Called if no frame arrives within timeout period"""
        self.pending_inspection = False
        self.status_bar.showMessage("Erro: Timeout ao aguardar frame da câmara.", 4000)
        # Ensure we don't leave camera running if we wanted to stop it? 
        # Actually logic is we want it running to get a frame. 
        # But if it failed, maybe we should repause if that was the intent, 
        # but here we just leave it as is or user can try again.
            
    def _start_inspection(self):
        """Helper to start inspection with captured frame"""
        if self.current_frame is None:
            self.status_bar.showMessage("Sem sinal de câmara!", 2000)
            return
        
        # Capture the current frame for inspection
        inspection_frame = self.current_frame.copy()
        
        # Pause camera to freeze display during and after processing
        self.camera_thread.paused = True

        # Start Worker with captured frame
        self.status_bar.showMessage("A Inspecionar...", 1000)
        
        self.worker = InspectionWorker(
            inspection_frame, 
            self.detector, 
            self.rectifier, 
            self.cropper, 
            self.aligner, 
            self.inferencer,
            self.resizer,  # NEW
            prepare_for_autoencoder  # NEW: pass function directly
        )
        self.worker.progress.connect(lambda msg: self.status_bar.showMessage(msg))
        self.worker.error.connect(self.on_inspection_error)
        self.worker.finished.connect(self.on_inspection_finished)
        self.worker.progressive_update.connect(self.on_progressive_update)  # NEW
        self.worker.start()

    def on_progressive_update(self, qt_image, current, total, scores):
        """Update display progressively as cans are inspected"""
        self.update_image_display(qt_image)
        self.status_bar.showMessage(f"A Inspecionar... {current}/{total} latas", 500)
        self.update_graph(scores, self.threshold_spinbox.value())

    def on_inspection_error(self, msg):
        self.status_bar.showMessage(f"Error: {msg}", 4000)
        # Unpause on error so user can try again
        if self.camera_thread:
            self.camera_thread.paused = False

    def on_inspection_finished(self, qt_image, stats, logs, results_data, clean_image):
        self.status_bar.showMessage("Inspeção Concluída", 2000)
        
        # Store data for live visual updates
        self.last_clean_image = clean_image
        self.last_results_data = results_data
        
        # 1. Update Display
        self.update_image_display(qt_image)
        
        # 2. Update Stats
        total_batch = stats['total']
        ok_batch = stats['ok']
        ng_batch = stats['ng']
        defect_batch = stats.get('defect', 0)
        quality_batch = stats.get('quality', 0)
        
        self.total_count += total_batch
        self.ok_count += ok_batch
        self.ng_count += ng_batch
        self.defect_count += defect_batch
        self.quality_count += quality_batch
        
        self.update_kpis()
        self.update_donut()
        
        # 3. Update Detailed Statistics Panel
        if total_batch > 0:
            good_pct = (ok_batch / total_batch) * 100
            bad_pct = (ng_batch / total_batch) * 100
        else:
            good_pct = 0
            bad_pct = 0
            
        # Timing stats
        timings = [log[3] for log in logs] # log[3] is duration
        scores = [log[1] for log in logs] # log[1] is score
        
        self.update_graph(scores, self.threshold_spinbox.value())
        
        if timings:
            avg_time = sum(timings) / len(timings)
        else:
            avg_time = 0
            
        # Update Labels
        self.lbl_stats_total.setText(f"{total_batch}")
        
        # Use start time from worker stats if available, else current
        # start_time_str = stats.get('start_time', '--:--:--')
        # self.lbl_stats_start.setText(f"Start: {start_time_str}")  # Disabled: User wants Session Start time, not Batch Start time 
        
        self.lbl_stats_avg_time.setText(f"{avg_time:.1f}ms")
        self.lbl_stats_good_pct.setText(f"{good_pct:.1f}%")
        self.lbl_stats_bad_pct.setText(f"{bad_pct:.1f}%")
        
        # 4. Add Logs with timing
        timing_msg = ""
        if timings:
            min_time = min(timings)
            max_time = max(timings)
            timing_msg = f"Timing: avg {avg_time:.0f}ms, min {min_time:.0f}ms, max {max_time:.0f}ms"
            
        for log_entry in logs:
            can_id, score, is_normal, duration = log_entry
            self.add_log_entry(can_id, score, is_normal)
        
        self.add_log_entry(0, 0, True, is_summary=True, 
                          message=f"Batch: {ok_batch} OK, {ng_batch} NG | {timing_msg}")
        
        # 4. Freeze display - do NOT auto-resume camera
        # User wants to keep inspection result on screen
        self.status_bar.showMessage(f"Inspection Complete. {timing_msg}", 3000)

    def refresh_visuals(self, threshold):
        """Redraw bounding boxes and update graph based on new threshold"""
        
        # 1. Update Graph Logic (Visuals)
        if self.last_results_data:
            scores = [item['score'] for item in self.last_results_data]
            self.update_graph(scores, threshold)
            
        # 2. Update Image Logic (Visuals)
        if self.last_clean_image is not None and self.last_results_data:
            # Create a copy to draw on
            annotated_sheet = self.last_clean_image.copy()
            
            for item in self.last_results_data:
                bbox = item['bbox']
                score = item['score']
                can_id = item['can_id']
                
                # Re-evaluate status
                is_normal = score <= threshold
                
                if is_normal:
                    color = (0, 255, 0)
                    status_text = "OK"
                else:
                    color = (0, 0, 255)
                    status_text = "NG"
                
                # Draw Box
                x1, y1, x2, y2 = bbox
                cv2.rectangle(annotated_sheet, (x1, y1), (x2, y2), color, 2)
                
                # Draw Label
                label = f"#{can_id} {status_text} {score:.2f}"
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(annotated_sheet, (x1, y1 - 20), (x1 + text_w, y1), color, -1)
                text_color = (0, 0, 0) if is_normal else (255, 255, 255)
                cv2.putText(annotated_sheet, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

            # Convert to QImage and Display
            rgb_sheet = cv2.cvtColor(annotated_sheet, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_sheet.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_sheet.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
            self.update_image_display(qt_image)

    def resume_camera(self):
        """Resume camera thread after inspection"""
        if self.camera_thread:
            self.camera_thread.paused = False


    def update_image_display(self, qt_image):
        """Helper to update the camera label with a QImage"""
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    def update_kpis(self):
        """Update KPI cards"""
        yield_pct = (self.ok_count / self.total_count * 100) if self.total_count > 0 else 0
        
        self.update_kpi(self.kpi_total, self.total_count)
        self.update_kpi(self.kpi_ok, self.ok_count)
        self.update_kpi(self.kpi_ng, self.ng_count)
        self.update_kpi(self.kpi_yield, f"{yield_pct:.1f}%")

    def add_log_entry(self, can_id, score, is_normal, is_summary=False, message=None):
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if is_summary:
            status = "BATCH"
            color = "#2196F3" # Blue
            text = message or "Batch Complete"
            label_text = f"[{timestamp}] {text}"
        else:
            status = "PASS" if is_normal else "FAIL"
            color = "#4CAF50" if is_normal else "#F44336"
            label_text = f"[{timestamp}] #{can_id}: {status} ({score:.2f})"
        
        item = QListWidgetItem(label_text)
        item.setForeground(Qt.white)
        item.setBackground(QColor(color))
        
        font = QFont()
        font.setBold(True)
        item.setFont(font)
        
        self.log_list.insertItem(0, item)

    def go_back(self):
        # Hide window to keep inspection running in background
        self.hide()
        if self.dashboard:
            self.dashboard.show()
        elif self.parent():
            self.parent().show()
    
    def update_graph(self, scores, threshold):
        self.ax.clear()
        
        # X coordinates (1 to 48)
        num_cans = len(scores)
        x_pos = range(1, num_cans + 1)
        
        # Colors based on threshold
        colors = ['#2EA043' if s <= threshold else '#DA3633' for s in scores]
        
        # Increased bar width to 0.9 (default is 0.8)
        bars = self.ax.bar(x_pos, scores, color=colors, alpha=0.7, width=0.9)
        
        # Threshold Line
        self.ax.axhline(y=threshold, color='#DA3633', linestyle='--', linewidth=1.5, label=f'Max: {threshold}')
        
        # Styling refresh - Remove all border lines ("lines in graph")
        self.ax.set_facecolor('#0D1117')
        self.ax.spines['bottom'].set_color('none')
        self.ax.spines['top'].set_color('none') 
        self.ax.spines['right'].set_color('none')
        self.ax.spines['left'].set_color('none')
        self.ax.grid(False) # Ensure no grid lines
        
        # Ticks styling - Remove tick lines (length=0)
        self.ax.tick_params(axis='x', colors='#8B949E', length=0)
        self.ax.tick_params(axis='y', colors='#8B949E', length=0)
        
        # X Axis setup 1-50 explicitly
        self.ax.set_xlim(0.5, 50.5)
        
        # Labels
        self.ax.set_title(f"Score Distribution (Max: {max(scores) if scores else 0:.1f})", color='#E6EDF3', fontsize=9)
        self.canvas.draw()

    def open_logs_dialog(self):
        """Open the logs dialog"""
        dlg = LogsDialog(self.log_list, self)
        dlg.exec()

    def update_donut(self):
        """Update the circular chart (Pie Chart) with current stats"""
        try:
            self.ax_donut.clear()
            
            raw_sizes = [self.ok_count, self.defect_count, self.quality_count]
            raw_labels = ['OK', 'Nok', 'Luz']
            
            # Palette: (Background, TextColor)
            # Palette: (Background, TextColor)
            # BG: Bright Accent Color
            # Text: Dark Color (Requested: Dark Green/Red text)
            palette = [
                ('#2EA043', '#1C3323'), # OK: Bright Green BG / Dark Green Text
                ('#DA3633', '#3E2020'), # Nok: Bright Red BG / Dark Red Text
                ('#2196F3', '#0C2D6B')  # Luz: Bright Blue BG / Dark Blue Text
            ]
            
            final_sizes = []
            final_colors = []
            final_text_colors = []
            final_labels = []
            
            for i, s in enumerate(raw_sizes):
                if s > 0:
                    final_sizes.append(s)
                    final_colors.append(palette[i][0])
                    final_text_colors.append(palette[i][1])
                    final_labels.append(raw_labels[i])
            
            if not final_sizes:
                self.ax_donut.axis('off')
                self.canvas_donut.draw()
                return
            
            # Use 'explode' to separate slices for better visibility
            explode = [0.03] * len(final_sizes)
            
            wedges, texts, autotexts = self.ax_donut.pie(
                final_sizes, 
                labels=final_labels,
                colors=final_colors, 
                autopct='%1.1f%%',
                startangle=90,
                explode=explode,
                shadow=False, 
                wedgeprops=dict(linewidth=0), # No border
                textprops=dict(fontsize=9, weight="bold"), # Color set below
                labeldistance=0.4,
                pctdistance=0.75
            )
            
            # Apply individual text colors
            for t, at, c in zip(texts, autotexts, final_text_colors):
                t.set_color(c)
                at.set_color(c)
                t.set_horizontalalignment('center')
                at.set_horizontalalignment('center')
            
            self.ax_donut.axis('equal') # Ensure circular shape
            self.canvas_donut.draw()
        except Exception as e:
            print(f"Error updating donut: {e}")

    def return_to_menu(self):
        """Hide this window and show the main dashboard (Background processing)"""
        self.hide()
        if self.parent():
            self.parent().show()

    def open_defects_gallery(self):
        """Open the defects gallery dialog"""
        dlg = DefectsGalleryDialog(defects_dir='defects', parent=self)
        dlg.exec()

    def update_threshold(self, value):
        """Update the PatchCore threshold when spinbox value changes"""
        if self.inferencer:
            self.inferencer.threshold = value
        
        # Live visual feedback
        self.refresh_visuals(value)
            
        self.save_config()

    def load_config(self):
        try:
            with open('config/detection_params.json', 'r') as f:
                data = json.load(f)
                return float(data.get('threshold', 2.0))
        except (FileNotFoundError, json.JSONDecodeError):
            return 2.0

    def save_config(self):
        try:
            os.makedirs('config', exist_ok=True)
            with open('config/detection_params.json', 'w') as f:
                json.dump({'threshold': self.threshold_spinbox.value()}, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")

    def mousePressEvent(self, event):
        # 1. Check if we have results and image to map to
        if not self.last_results_data or self.last_clean_image is None:
            super().mousePressEvent(event)
            return

        # 2. Map coordinates: Window -> Video Label
        lbl_pos = self.video_label.mapFrom(self, event.pos())
        
        # Check if inside label
        if not self.video_label.rect().contains(lbl_pos):
            super().mousePressEvent(event)
            return
            
        # 3. Calculate Image Coordinates logic
        lbl_w = self.video_label.width()
        lbl_h = self.video_label.height()
        
        img_h, img_w = self.last_clean_image.shape[:2]
        
        # Calculate scale (AspectRatioMode.KeepAspectRatio)
        scale_w = lbl_w / img_w
        scale_h = lbl_h / img_h
        scale = min(scale_w, scale_h)
        
        # Displayed dimensions
        disp_w = int(img_w * scale)
        disp_h = int(img_h * scale)
        
        # Offset (centering)
        off_x = (lbl_w - disp_w) // 2
        off_y = (lbl_h - disp_h) // 2
        
        # Click Position relative to label
        click_x = lbl_pos.x()
        click_y = lbl_pos.y()
        
        # Check if inside displayed image
        if (click_x < off_x or click_x > off_x + disp_w or 
            click_y < off_y or click_y > off_y + disp_h):
            return # Clicked on black bars
            
        # Map to original image coords
        orig_x = int((click_x - off_x) / scale)
        orig_y = int((click_y - off_y) / scale)
        
        # 4. Find Can
        for item in self.last_results_data:
            x1, y1, x2, y2 = item['bbox']
            if x1 <= orig_x <= x2 and y1 <= orig_y <= y2:
                # Found it!
                self.show_can_details(item)
                return
                
        super().mousePressEvent(event)

    def show_can_details(self, item):
        score = item['score']
        threshold = self.threshold_spinbox.value()
        status = "OK" if score <= threshold else "NG"
        
        dlg = CanDetailDialog(
            can_id=item['can_id'],
            score=score,
            status=status,
            image=item.get('image'),
            heatmap=item.get('heatmap'),
            threshold=threshold,
            parent=self
        )
        dlg.exec()
        
    def closeEvent(self, event):
        from PySide6.QtWidgets import QMessageBox
        if self.is_running:
            reply = QMessageBox.question(self, 'Alerta de Saída',
                "⚠️ A inspeção está em andamento!\n\nDeseja realmente sair e PARAR a inspeção?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.No:
                event.ignore()
                return

        # Explicitly stop resources if closing
        self.timer.stop()
        if self.camera_thread:
            self.camera_thread.stop()
        if self.camera:
            self.camera.release()
        
        # Ensure Dashboard is shown when this window closes
        if self.parent():
            self.parent().show()
            
        event.accept()
