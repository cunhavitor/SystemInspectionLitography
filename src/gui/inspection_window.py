import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import json
import os
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
# from ..inference.rd4ad_inference import RD4ADInferencer # Removed
# from ..inspection import SELECTED_MODEL  # Removed
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
from datetime import datetime
import os
from ..plc_control import PLCManager

class InspectionWorker(QThread):
    finished = Signal(object, dict, list, list, object, float) # qt_image, stats, logs, results_data, clean_image, completion_time
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
            
            defects_base = os.path.join('data', 'defects')
            defects_dir = os.path.join(defects_base, year_dir, month_dir)
            
            # Ensure data directory exists
            if not os.path.exists(defects_dir):
                os.makedirs(defects_dir)
            if not os.path.exists('data'):
                os.makedirs('data')
                
            os.makedirs(defects_dir, exist_ok=True)
            timestamp = start_time_dt.strftime("%Y%m%d_%H%M%S")
            start_time_str = start_time_dt.strftime("%H:%M:%S")
            
            # Capture Total Pipeline Start Time
            import time
            t_trigger = time.time()
            timings = {}
            
            check_img = self.frame.copy()
            
            # --- Step 1: Detect Corners ---
            self.progress.emit("Detecting Corners...")
            t0 = time.time()
            corners = self.detector.detect(check_img)
            timings['detect_corners'] = (time.time() - t0) * 1000
            
            # --- Step 2: Rectify Sheet ---
            self.progress.emit("Rectifying Sheet...")
            t0 = time.time()
            rectified, pixels_per_mm = self.rectifier.get_warped(check_img, corners)
            timings['rectify'] = (time.time() - t0) * 1000
            if rectified is None:
                self.error.emit("Rectification Failed")
                return

            # --- Step 3: Crop Cans ---
            self.progress.emit("Cropping Cans...")
            t0 = time.time()
            cans = self.cropper.crop_cans(rectified, pixels_per_mm=pixels_per_mm)
            timings['crop'] = (time.time() - t0) * 1000
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
            t_cans_start = time.time()
            
            # Accumulators for average
            acc_align = 0
            acc_infer = 0
            acc_clahe = 0
            acc_resize = 0
            acc_norm = 0
            acc_norm = 0
            acc_ov = 0
            
            t_first_can_finish = None
            
            print(f"DEBUG: Starting loop for {len(cans)} cans.")
            
            for i, item in enumerate(cans):
                can_id = item['id']
                can_img = item['image']
                bbox = item['bbox']
                
                can_start_time = time.time()
                
                # A. Resize
                resized_can = self.resizer.process(can_img)
                
                # B. Align (REQUIRED for accurate scores)
                t_align_0 = time.time()
                if self.aligner:
                    aligned_can = self.aligner.align(resized_can)
                else:
                    aligned_can = resized_can
                acc_align += (time.time() - t_align_0) * 1000
                
                # C. Normalize with CLAHE
                normalized_can = self.normalizer(aligned_can, target_size=(448, 448))
                
                # DEBUG: Log/Save if it's one of the first 3 cans OR specifically can 1
                if i < 3 or can_id == 1:
                    # cv2.imwrite(f'debug_pi_can{can_id:02d}_1_raw_crop.png', can_img)
                    # cv2.imwrite(f'debug_pi_can{can_id:02d}_2_resized.png', resized_can)
                    # cv2.imwrite(f'debug_pi_can{can_id:02d}_3_aligned.png', aligned_can)
                    # cv2.imwrite(f'debug_pi_can{can_id:02d}_4_normalized.png', normalized_can)
                    
                    print(f"\n=== CAN {can_id} PREPROCESSING DEBUG ===")
                    #print(f"1. Raw crop    - Shape: {can_img.shape}, Mean: {np.mean(can_img):.2f}")
                    #print(f"2. Resized     - Shape: {resized_can.shape}, Mean: {np.mean(resized_can):.2f}")
                    #print(f"3. Aligned     - Shape: {aligned_can.shape}, Mean: {np.mean(aligned_can):.2f}")
                    #print(f"4. Normalized  - Shape: {normalized_can.shape}, Mean: {np.mean(normalized_can):.2f}")
                    #print(f"💾 Saved debug output for Can {can_id}\n")
                
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
                    score, is_normal, viz, heatmap, infer_timings = self.inferencer.predict(aligned_can)
                    #print(f"DEBUG: Can #{can_id} Score: {score:.8f}")
                    
                    acc_infer += infer_timings['total_infer']
                    acc_clahe += infer_timings['clahe']
                    acc_resize += infer_timings['resize']
                    acc_norm += infer_timings['norm']
                    acc_ov += infer_timings['openvino']
                    
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
                
                label = f"#{can_id} {status_text}\n{score:.2f}"
                font_scale = 1.2
                thickness = 2
                
                # Split lines: handle both \n and \r
                lines = label.replace('\r', '').split('\n')
                
                # Calculate max width and total height
                max_width = 0
                total_height = 0
                line_heights = []
                
                for line in lines:
                   (txt_w, txt_h), baseline = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                   max_width = max(max_width, txt_w)
                   # Add padding between lines
                   h_with_padding = txt_h + baseline + 10
                   line_heights.append(h_with_padding)
                   total_height += h_with_padding
                
                # Draw background box (inside the bbox, aligned to top)
                box_buffer = 10
                box_top_y = y1
                box_bottom_y = y1 + total_height + box_buffer
                
                cv2.rectangle(annotated_sheet, (x1, box_top_y), (x1 + max_width + 10, box_bottom_y), color, -1)
                
                text_color = (0, 0, 0) if is_normal else (255, 255, 255)
                
                # Draw each line
                current_y = box_top_y + box_buffer
                for line_idx, line in enumerate(lines):
                    # getTextSize return height is from baseline to top. putText y is baseline.
                    (txt_w, txt_h), baseline = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    current_y += txt_h # move down by text height
                    
                    cv2.putText(annotated_sheet, line, (x1 + 5, current_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
                    
                    current_y += baseline + 10 # move down for next line
                
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
                
                # Check for first can timing
                if i == 0:
                     t_first_can_finish = time.time()
                
                self.progressive_update.emit(qt_partial, i + 1, len(cans), current_scores)
            
            if defects_saved > 0:
                print(f"💾 Saved {defects_saved} defect images to '{defects_dir}/'")
            
            stats = {
                'total': len(cans),
                'ok': batch_ok,
                'ng': batch_ng,
                'defect': batch_defect,
                'quality': batch_quality,
                'start_time': start_time_str,
                't_first_can': t_first_can_finish # Pass back to main thread
            }
            
            # Convert to RGB for UI
            rgb_sheet = cv2.cvtColor(annotated_sheet, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_sheet.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_sheet.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
            
            # Capture completion time for accurate total time
            completion_time = time.time()
            
            # --- PRINT PERFORMANCE REPORT ---
            num_cans = len(cans)
            timings['total_cans'] = (time.time() - t_cans_start) * 1000
            timings['total_pipeline'] = (completion_time - t_trigger) * 1000
            
            print("\n" + "="*50)
            print(f" PERFORMANCE REPORT ({num_cans} cans) [GUI MODE]")
            print("="*50)
            print(f" TOTAL TIME:         {timings['total_pipeline']:.1f} ms  ({timings['total_pipeline']/1000:.2f} s)")
            print("-" * 50)
            print(f" 1. Detect Corners:  {timings['detect_corners']:.1f} ms")
            print(f" 2. Rectify Sheet:   {timings['rectify']:.1f} ms")
            print(f" 3. Crop Cans:       {timings['crop']:.1f} ms")
            print("-" * 50)
            print(f" 4. Processing Loop: {timings['total_cans']:.1f} ms")
            if num_cans > 0:
                print(f"    - Avg Per Can:   {timings['total_cans']/num_cans:.1f} ms")
                print(f"    - Avg Align:     {acc_align/num_cans:.1f} ms")
                print(f"    - Avg Infer:     {acc_infer/num_cans:.1f} ms")
                print(f"        * CLAHE:     {acc_clahe/num_cans:.1f} ms")
                print(f"        * Resize:    {acc_resize/num_cans:.1f} ms")
                print(f"        * Norm:      {acc_norm/num_cans:.1f} ms")
                print(f"        * OpenVINO:  {acc_ov/num_cans:.1f} ms")
            print("="*50 + "\n")
            
            self.finished.emit(qt_image, stats, logs, results_data, rectified, completion_time)
            
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
    def __init__(self, defects_dir=None, parent=None):
        super().__init__(parent)
        if defects_dir is None:
            defects_dir = os.path.join('data', 'defects')
            
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

class NewOPDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Nova Ordem de Produção (OP)")
        self.setMinimumWidth(400)
        self.setStyleSheet("background-color: #0D1117; color: #E6EDF3;")
        
        layout = QVBoxLayout(self)
        
        # Form Layout
        from PySide6.QtWidgets import QFormLayout, QLineEdit, QComboBox, QSpinBox
        form_layout = QFormLayout()
        
        # 1. SKU / Modelo
        self.combo_sku = QComboBox()
        self.combo_sku.setStyleSheet("background-color: #21262D; border: 1px solid #30363D; padding: 5px; color: white;")
        self.combo_sku.addItem("Bom Petisco Oleo - rr125")
        self.combo_sku.addItem("Bom Petisco Azeite - rr125")
        # Add more items here if needed in future
        form_layout.addRow("SKU / Modelo:", self.combo_sku)
        
        # 2. Ordem de Produção
        self.txt_op = QLineEdit()
        self.txt_op.setPlaceholderText("Ex: OP-2024-001")
        self.txt_op.setStyleSheet("background-color: #010409; border: 1px solid #30363D; padding: 5px; color: white;")
        form_layout.addRow("Ordem Produção:", self.txt_op)
        
        # 3. Quantidade
        self.spin_qty = QSpinBox()
        self.spin_qty.setRange(1, 999999)
        self.spin_qty.setValue(1000)
        self.spin_qty.setStyleSheet("background-color: #010409; border: 1px solid #30363D; padding: 5px; color: white;")
        form_layout.addRow("Quantidade Prevista:", self.spin_qty)
        
        layout.addLayout(form_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_cancel = QPushButton("Cancelar")
        btn_cancel.setStyleSheet("background-color: #DA3633; color: white; border: none; padding: 8px; border-radius: 4px;")
        btn_cancel.clicked.connect(self.reject)
        
        btn_confirm = QPushButton("Criar OP")
        btn_confirm.setStyleSheet("background-color: #238636; color: white; border: none; padding: 8px; border-radius: 4px; font-weight: bold;")
        btn_confirm.clicked.connect(self.accept)
        
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_confirm)
        layout.addLayout(btn_layout)
        
    def get_data(self):
        return {
            "sku": self.combo_sku.currentText(),
            "op": self.txt_op.text().strip(),
            "qty": self.spin_qty.value()
        }

class ReportsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Relatórios de Produção")
        self.resize(800, 600)
        self.setStyleSheet("background-color: #0D1117; color: #E6EDF3;")
        
        layout = QHBoxLayout(self)
        
        # Left: File List
        list_layout = QVBoxLayout()
        lbl_files = QLabel("Arquivos Disponíveis")
        lbl_files.setStyleSheet("font-weight: bold; color: #58A6FF;")
        list_layout.addWidget(lbl_files)
        
        self.file_list = QListWidget()
        self.file_list.setStyleSheet("""
            QListWidget {
                background-color: #161B22;
                border: 1px solid #30363D;
                border-radius: 6px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #21262D;
            }
            QListWidget::item:selected {
                background-color: #1F6FEB;
                color: white;
            }
        """)
        self.file_list.itemClicked.connect(self.load_report_content)
        list_layout.addWidget(self.file_list)
        
        # Refresh Button
        btn_refresh = QPushButton("Atualizar Lista")
        btn_refresh.setStyleSheet("background-color: #238636; color: white; padding: 6px; border-radius: 4px;")
        btn_refresh.clicked.connect(self.load_file_list)
        list_layout.addWidget(btn_refresh)
        
        layout.addLayout(list_layout, 1)
        
        # Right: Content Viewer
        content_layout = QVBoxLayout()
        lbl_content = QLabel("Conteúdo do Relatório")
        lbl_content.setStyleSheet("font-weight: bold; color: #58A6FF;")
        content_layout.addWidget(lbl_content)
        
        from PySide6.QtWidgets import QTextEdit
        self.text_viewer = QTextEdit()
        self.text_viewer.setReadOnly(True)
        self.text_viewer.setStyleSheet("""
            QTextEdit {
                background-color: #0D1117;
                border: 1px solid #30363D;
                border-radius: 6px;
                padding: 10px;
                font-family: monospace;
                font-size: 13px;
                color: #C9D1D9;
            }
        """)
        content_layout.addWidget(self.text_viewer)
        
        # Button Layout (Close + Print)
        btn_layout = QHBoxLayout()
        
        # Print Button
        btn_print = QPushButton("Imprimir")
        btn_print.setStyleSheet("background-color: #1F6FEB; color: white; padding: 8px; border-radius: 4px; font-weight: bold;")
        btn_print.clicked.connect(self.print_report)
        btn_layout.addWidget(btn_print)
        
        btn_layout.addStretch()
        
        # Close Button
        btn_close = QPushButton("Fechar")
        btn_close.setStyleSheet("background-color: #30363D; color: white; padding: 8px; border-radius: 4px;")
        btn_close.clicked.connect(self.accept)
        btn_layout.addWidget(btn_close)
        
        content_layout.addLayout(btn_layout)
        
        layout.addLayout(content_layout, 2)
        
        # Initial Load
        self.load_file_list()
        
    def load_file_list(self):
        self.file_list.clear()
        reports_dir = "reports"
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
            
        files = [f for f in os.listdir(reports_dir) if f.endswith(".txt")]
        # Sort by newest first (assuming naming convention has date, otherwise sort by mtime)
        files.sort(reverse=True) 
        
        for f in files:
            self.file_list.addItem(f)
            
    def load_report_content(self, item):
        filename = item.text()
        path = os.path.join("reports", filename)
        try:
            with open(path, "r") as f:
                content = f.read()
            self.text_viewer.setText(content)
        except Exception as e:
            self.text_viewer.setText(f"Erro ao ler arquivo: {e}")

    def print_report(self):
        """Print the currently displayed report"""
        from PySide6.QtPrintSupport import QPrinter, QPrintDialog
        
        printer = QPrinter(QPrinter.ScreenResolution)
        dialog = QPrintDialog(printer, self)
        
        if dialog.exec() == QPrintDialog.Accepted:
            self.text_viewer.print_(printer)
        lbl_files.setStyleSheet("font-weight: bold; color: #58A6FF;")
        list_layout.addWidget(lbl_files)
        
        self.file_list = QListWidget()
        self.file_list.setStyleSheet("""
            QListWidget {
                background-color: #161B22;
                border: 1px solid #30363D;
                border-radius: 6px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #21262D;
            }
            QListWidget::item:selected {
                background-color: #1F6FEB;
                color: white;
            }
        """)
        self.file_list.itemClicked.connect(self.load_report_content)
        list_layout.addWidget(self.file_list)
        
        # Refresh Button
        btn_refresh = QPushButton("Atualizar Lista")
        btn_refresh.setStyleSheet("background-color: #238636; color: white; padding: 6px; border-radius: 4px;")
        btn_refresh.clicked.connect(self.load_file_list)
        list_layout.addWidget(btn_refresh)
        
        layout.addLayout(list_layout, 1)
        
        # Right: Content Viewer
        content_layout = QVBoxLayout()
        lbl_content = QLabel("Conteúdo do Relatório")
        lbl_content.setStyleSheet("font-weight: bold; color: #58A6FF;")
        content_layout.addWidget(lbl_content)
        
        from PySide6.QtWidgets import QTextEdit
        self.text_viewer = QTextEdit()
        self.text_viewer.setReadOnly(True)
        self.text_viewer.setStyleSheet("""
            QTextEdit {
                background-color: #0D1117;
                border: 1px solid #30363D;
                border-radius: 6px;
                padding: 10px;
                font-family: monospace;
                font-size: 13px;
                color: #C9D1D9;
            }
        """)
        content_layout.addWidget(self.text_viewer)
        
        # Close Button
        btn_close = QPushButton("Fechar")
        btn_close.setStyleSheet("background-color: #30363D; color: white; padding: 8px; border-radius: 4px;")
        btn_close.clicked.connect(self.accept)
        content_layout.addWidget(btn_close)
        
        layout.addLayout(content_layout, 2)
        
        # Initial Load
        self.load_file_list()
        
    def load_file_list(self):
        self.file_list.clear()
        reports_dir = "reports"
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
            
        files = [f for f in os.listdir(reports_dir) if f.endswith(".txt")]
        # Sort by newest first (assuming naming convention has date, otherwise sort by mtime)
        files.sort(reverse=True) 
        
        for f in files:
            self.file_list.addItem(f)
            
    def load_report_content(self, item):
        filename = item.text()
        path = os.path.join("reports", filename)
        try:
            with open(path, "r") as f:
                content = f.read()
            self.text_viewer.setText(content)
        except Exception as e:
            self.text_viewer.setText(f"Erro ao ler arquivo: {e}")

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
            print(f"DEBUG: CanDetailDialog Heatmap Shape: {heatmap.shape}, Min: {np.min(heatmap):.4f}, Max: {np.max(heatmap):.4f}")
            # Heatmap is likely float 0-1 or similar. Need to normalize and colorize.
            # Assuming heatmap is pre-processed or raw float map.
            try:
                # Normalize to 0-255 using fixed ceiling (15.0)
                teto_maximo = 15.0
                hm_norm = np.clip(heatmap / teto_maximo * 255, 0, 255).astype(np.uint8)
                
                # Apply colormap (Jet)
                hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
                
                # Create overlay if image exists
                if image is not None:
                    # Ensure image size matches heatmap
                    if image.shape[:2] != hm_color.shape[:2]:
                        bg_img = cv2.resize(image, (hm_color.shape[1], hm_color.shape[0]))
                    else:
                        bg_img = image.copy()
                    
                    # Blend: 20% Image + 80% Heatmap
                    final_display = cv2.addWeighted(bg_img, 0.2, hm_color, 0.8, 0)
                else:
                    final_display = hm_color

                rgb_hm = cv2.cvtColor(final_display, cv2.COLOR_BGR2RGB)
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
    # Signals
    sig_plc_trigger = Signal() # Signal to bridge GPIO thread -> Qt Main Thread

    def __init__(self, config, user_manager, dashboard=None):
        super().__init__()
        self.config = config
        self.user_manager = user_manager
        self.dashboard = dashboard
        
        # Connect Signal
        self.sig_plc_trigger.connect(self.trigger_inspection)
        
        # Determine User Role
        # Assuming user_manager has logged_in_user content or we get it from dashboard
        self.current_user = self.user_manager.current_user
        self.user_role = self.current_user.role if self.current_user else 'operador'
        
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
        self.current_job = None # Stores current job info {sku, op, qty}
        
        # Session Counters (Reset on every Start)
        self.session_counts = {
            'total': 0, 'ok': 0, 'ng': 0, 'defect': 0, 'quality': 0
        }
        
        # PLC Control
        self.plc = PLCManager()
        
        # Connect PLC Trigger (Input 23)
        if self.plc.in_trigger:
            # We use a lambda to emit a signal or call a thread-safe method? 
            # gpiozero callbacks run in a separate thread.
            # We must use Qt signals to interact with UI/MainThread.
            self.plc.in_trigger.when_activated = self.handle_plc_trigger
            print("PLC Trigger Connected to Input 23")
        else:
            print("WARNING: PLC Input 23 (Trigger) not available")

        # Inspection Trigger Logic
        self.pending_inspection = False
        self.trigger_time = None  # Track trigger time for first can timing
        self.first_can_complete_time = None  # Track when first can completes
        self.inspection_timeout_timer = QTimer(self)
        self.inspection_timeout_timer.setSingleShot(True)
        self.inspection_timeout_timer.timeout.connect(self.on_inspection_timeout)
        
        # Load Aligner - NOW LOADED PER SKU in load_model_for_sku()
        # Each SKU has its own reference image for alignment
        self.aligner = None
        # ref_path = 'models/can_reference/aligned_can_reference448.png'
        # if os.path.exists(ref_path):
        #     try:
        #         # FIX: Set target_size to 448x448 to match PatchCore model
        #         self.aligner = CanAligner(ref_path, target_size=(448, 448))
        #         print(f"Aligner loaded with reference: {ref_path}")
        #     except Exception as e:
        #         print(f"Failed to load aligner: {e}")
        # else:
        #     print(f"Reference image not found at {ref_path}")




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
        self.showMaximized()  # Start window maximized
        
        self._setup_ui()
        # Delay camera start to allow UI to show 
        QTimer.singleShot(500, self._start_camera)

        # Status Bar Custom Setup
        self.lbl_status_msg = QLabel("Pronto")
        self.lbl_status_msg.setStyleSheet("color: #E6EDF3; padding-left: 10px;")
        
        self.lbl_current_op = QLabel("SEM OP")
        self.lbl_current_op.setAlignment(Qt.AlignCenter)
        self.lbl_current_op.setStyleSheet("color: #6E7681; font-weight: bold; font-size: 14px;")
        
        self.lbl_status_right_spacer = QLabel("") 
        
        self.statusBar().addWidget(self.lbl_status_msg, 1) 
        self.statusBar().addWidget(self.lbl_current_op, 0) 
        self.statusBar().addWidget(self.lbl_status_right_spacer, 1) 
        
        # Compatibility ref
        self.status_bar = self.statusBar()

        # Load saved state if available
        self.load_op_state()
        
    def load_model_for_sku(self, sku):
        """
        Load the appropriate PatchCore model based on the SKU.
        
        Args:
            sku: SKU identifier (e.g., 'Bom Petisco Oleo - rr125', 'Bom Petisco Azeite - rr125', 'Bom Petisco Azeite - rr125')
        """
        # Map SKU to model directory
        sku_model_map = {
            'Bom Petisco Oleo - rr125': 'models/bpo_rr125_patchcore_resnet50',
            'Bom Petisco Azeite - rr125': 'models/bpAz_rr125_patchcore_v2'
        }
        
        # Map SKU to reference image for alignment
        sku_reference_map = {
            'Bom Petisco Oleo - rr125': 'models/can_reference/aligned_can_reference448_bpo-rr125.png',
            'Bom Petisco Azeite - rr125': 'models/can_reference/aligned_can_reference448_bpAz-rr125.png'
        }
        
        model_dir = sku_model_map.get(sku)
        ref_path = sku_reference_map.get(sku)
        
        if model_dir is None:
            print(f"WARNING: Unknown SKU '{sku}'. Using default model.")
            model_dir = 'models/bpo_rr125_patchcore_resnet50'
            ref_path = 'models/can_reference/aligned_can_reference448_bpo-rr125.png'
        
        if not os.path.exists(model_dir):
            print(f"ERROR: Model directory '{model_dir}' not found for SKU '{sku}'")
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Modelo Não Encontrado", 
                              f"O modelo para o SKU '{sku}' não foi encontrado em:\n{model_dir}\n\n"
                              f"Verifique se os arquivos do modelo estão presentes.")
            return False
        
        try:
            print(f"Loading PatchCore model for SKU '{sku}' from '{model_dir}'...")
            self.inferencer = PatchCoreInferencer(model_dir=model_dir)
            
            # Apply user's saved threshold (Limiar Max) to the new model
            if hasattr(self, 'threshold_spinbox'):
                user_threshold = self.threshold_spinbox.value()
                # If we are switching to ResNet50 and the threshold is the default 10 (from previous model),
                # we might want to bump it up because ResNet50 is "hotter".
                if "resnet50" in model_dir and user_threshold == 10.0:
                     print("Auto-adjusting threshold for ResNet50 to 25.0")
                     user_threshold = 25.0
                     self.threshold_spinbox.setValue(user_threshold)
                     
                self.inferencer.threshold = user_threshold
                print(f"✓ PatchCore model loaded successfully for SKU '{sku}'")
                print(f"  Default threshold: 10.0 → User threshold: {user_threshold}")
            else:
                # During initialization, spinbox doesn't exist yet
                # Set specific default for ResNet50
                if "resnet50" in model_dir:
                    self.inferencer.threshold = 25.0
                
                print(f"✓ PatchCore model loaded successfully for SKU '{sku}'")
                print(f"  Threshold: {self.inferencer.threshold} (will be updated after UI setup)")
            
            # Load SKU-specific reference image for alignment
            if ref_path and os.path.exists(ref_path):
                try:
                    self.aligner = CanAligner(ref_path, target_size=(448, 448))
                    print(f"✓ Aligner loaded with SKU-specific reference: {ref_path}")
                except Exception as e:
                    print(f"WARNING: Failed to load aligner for SKU '{sku}': {e}")
                    self.aligner = None
            else:
                print(f"WARNING: Reference image not found at {ref_path}")
                print(f"  Alignment will be skipped for SKU '{sku}'")
                self.aligner = None
            
            return True
        except Exception as e:
            print(f"Failed to load PatchCore model for SKU '{sku}': {e}")
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Erro ao Carregar Modelo", 
                               f"Falha ao carregar o modelo para SKU '{sku}':\n{str(e)}")
            return False

        
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
                border-top: 1px solid #3E4C59;

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
        
        # Reports Button (Restricted)
        if self.user_role in ['admin', 'master', 'tecnico']:
            btn_reports = QPushButton("Relatórios")
            btn_reports.setCursor(Qt.PointingHandCursor)
            btn_reports.setStyleSheet("""
                QPushButton {
                    background-color: #238636;
                    color: white;
                    border: 1px solid #2EA043;
                    border-radius: 4px;
                    padding: 5px 10px;
                    font-weight: bold;
                    margin-left: 10px;
                }
                QPushButton:hover { background-color: #2EA043; }
            """)
            btn_reports.clicked.connect(self.open_reports_dialog)
            header.addWidget(btn_reports)

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
        
        self.btn_new_op = QPushButton("NOVA OP")
        self.btn_new_op.setCursor(Qt.PointingHandCursor)
        self.btn_new_op.setStyleSheet("""
            QPushButton {
                background-color: #1F6FEB;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
                border: 1px solid #1F6FEB;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #388BFD; }
            QPushButton:disabled { background-color: #1F2428; color: #484F58; border: 1px solid #30363D; }
        """)
        self.btn_new_op.clicked.connect(self.on_new_op)
        
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
        self.btn_start.setEnabled(False) # Require job first? Or optional? User didn't specify strict requirement but said "add a button... that opens dialog... this button is only enable when is in stop mode".
        # I'll keep Start enabled by default but if they use New Job it sets context. 
        # Wait, if they WANT to enforce job, I should maybe disable Start. 
        # User said "this button [New Job] is only enable when is in stop mode". 
        # I will leave Start enabled but maybe warn if no job? 
        # Or better: Disable Start until New Job is clicked if we want to force it. 
        # For now, I'll allow Start primarily, but New Job is the preferred flow.
        # Actually user flow suggests: "add new job... after that create a log".
        # I will keep Start ENABLED for backward compatibility unless specified, 
        # but the New Job button is prominent.
        
        # controls_layout.addWidget(self.btn_new_job)
        # controls_layout.addWidget(self.btn_start)
        
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
        
        self.btn_finish_op = QPushButton("TERMINAR OP")
        self.btn_finish_op.setCursor(Qt.PointingHandCursor)
        self.btn_finish_op.setStyleSheet("""
            QPushButton {
                background-color: #8B949E;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
                border: 1px solid #6E7681;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #6E7681; }
            QPushButton:disabled { background-color: #1F2428; color: #484F58; border: 1px solid #30363D; }
        """)
        self.btn_finish_op.clicked.connect(self.on_finish_op)
        self.btn_finish_op.setEnabled(False)
        
        # Grid Layout for Controls
        controls_layout = QGridLayout()
        controls_layout.addWidget(self.btn_new_op, 0, 0)
        controls_layout.addWidget(self.btn_finish_op, 0, 1)
        controls_layout.addWidget(self.btn_start, 1, 0)
        controls_layout.addWidget(self.btn_stop, 1, 1)
        
        controls_widget = QWidget()
        controls_widget.setLayout(controls_layout)
        left_layout.addWidget(controls_widget)
        
        controls_widget.setLayout(controls_layout)
        left_layout.addWidget(controls_widget)
        
        # KPI Group (Contadores OP) - Matches "Estatísticas Turno" style
        kpi_group = QGroupBox("Contadores OP")
        kpi_layout = QVBoxLayout()
        kpi_layout.setSpacing(12)
        
        # Using requested palette:
        self.kpi_total = self._create_kpi_card("TOTAL", "0", "#21262D") # Dark Neutral
        self.kpi_ok = self._create_kpi_card("OK", "0", "#1C3323")    # Dark Green Tint #2EA043 accent
        
        self.kpi_ng = self._create_kpi_card("NG", "0", "#3E2020")    # Dark Red Tint
        self.kpi_yield = self._create_kpi_card("Perc. OK", "0%", "#382810") # Dark Orange Tint
        
        kpi_layout.addWidget(self.kpi_total)
        kpi_layout.addWidget(self.kpi_ok)
        kpi_layout.addWidget(self.kpi_ng)
        kpi_layout.addWidget(self.kpi_yield)
        
        kpi_group.setLayout(kpi_layout)
        left_layout.addWidget(kpi_group)
        
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
        stats_group = QGroupBox("Estatísticas Turno")
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
        self.lbl_stats_first_can = add_stat_row(4, "Tempo 1ª Lata:", "--ms")
        self.lbl_stats_total_time = add_stat_row(5, "Tempo Total:", "--s")
        self.lbl_stats_good_pct = add_stat_row(6, "Bons:", "0%")
        self.lbl_stats_bad_pct  = add_stat_row(7, "Maus:", "0%")
        
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
        self.btn_logs.setStyleSheet("border: 1px solid #30363D; color: #8B949E;") 
        self.btn_logs.clicked.connect(self.open_logs_dialog)
        right_layout.addWidget(self.btn_logs)
        
        # 5. IO Status Button
        self.btn_io = QPushButton("🔌 Estado I/O")
        self.btn_io.setCursor(Qt.PointingHandCursor)
        self.btn_io.setStyleSheet("border: 1px solid #30363D; color: #8B949E;")
        self.btn_io.clicked.connect(self.open_io_dialog)
        right_layout.addWidget(self.btn_io)
        
        # Access Control: Hide Logs, Defects, IO for Operators
        # Only Master and Tecnico can see these sensitive debugging/historical tools
        if self.user_manager.current_user.role not in ['admin', 'tecnico']:
            self.btn_logs.hide()
            self.btn_defects.hide()
            self.btn_io.hide()

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
        
        # Initialize Timing Variables
        self.trigger_time = None
        self.first_can_complete_time = None

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
            self.update_status("A aquecer câmara...", 0)
            
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
            self.update_status("Câmara Pronta", 2000)
        except Exception as e:
            self.video_label.setText(f"Camera Error: {str(e)}")
            self.update_status(f"Error: {str(e)}")

    def update_frame(self, qt_image, sharpness, noise, raw_frame):
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




    def on_new_op(self):
        """Open New OP Dialog and configure context"""
        dlg = NewOPDialog(self)
        if dlg.exec():
            data = dlg.get_data()
            
            # Load the appropriate model for this SKU
            sku = data['sku']
            if not self.load_model_for_sku(sku):
                # Model loading failed, don't proceed with OP creation
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Erro", 
                                  f"Não foi possível carregar o modelo para o SKU '{sku}'.\n"
                                  "A Ordem de Produção não será criada.")
                return
            
            # Capture Start Time for Report
            data['start_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.current_job = data
            
            # Log Job Start
            timestamp = data['start_time']
            log_msg = (f"OP STARTED | SKU: {data['sku']} | "
                       f"OP: {data['op']} | Qty: {data['qty']}")
            
            print(f"[{timestamp}] {log_msg}")
            
            # Persist to file (Log)
            try:
                if not os.path.exists("logs"):
                    os.makedirs("logs")
                with open("logs/jobs.log", "a") as f:
                    f.write(f"[{timestamp}] {log_msg}\n")
            except Exception as e:
                print(f"Failed to write to job log: {e}")
                
            # Persist State (Resume capability)
            self.save_op_state()
            
            # Reset ALL Counters for new OP
            self.total_count = 0
            self.ok_count = 0
            self.ng_count = 0
            self.defect_count = 0
            self.quality_count = 0
            
            self.session_counts = {
                'total': 0, 'ok': 0, 'ng': 0, 'defect': 0, 'quality': 0
            }
            self.update_kpis()
            self.update_donut()
            
            # Update Status Bar & Footer
            self.update_status(f"OP Iniciada: {data['op']}")
            if hasattr(self, 'lbl_current_op'):
                self.lbl_current_op.setText(f"OP EM CURSO: {data['op']} ({data['sku']})")
            
            # Update Stats Panel Start Time
            if hasattr(self, 'lbl_stats_start'):
                # Format to HH:MM if preferred, or maintain full date
                # data['start_time'] is YYYY-MM-DD HH:MM:SS
                try:
                    dt = datetime.strptime(data['start_time'], "%Y-%m-%d %H:%M:%S")
                    self.lbl_stats_start.setText(dt.strftime('%d/%m/%Y %H:%M'))
                except:
                    self.lbl_stats_start.setText(data['start_time'])
            if hasattr(self, 'lbl_stats_end'):
                self.lbl_stats_end.setText("--/-- --:--")
            
            # Enable Buttons
            self.btn_new_op.setEnabled(False) # Block new OP creation while active
            self.btn_start.setEnabled(True)
            self.btn_finish_op.setEnabled(True)
            
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(self, "OP Criada", 
                                  f"Ordem de Produção configurada com sucesso!\n\n{log_msg}\n\n"
                                  f"Modelo carregado: {sku}")

    def start_inspection_mode(self):
        if self.is_running:
            return
            
        if not self.current_job:
            # Fallback
            self.current_job = {"sku": "Bom Petisco Oleo - rr125", "op": "DEFAULT", "qty": 0}
            print("Warning: Starting without explicit job. Using default.")
            
        # Reset Session Counters
        self.session_counts = {
            'total': 0, 'ok': 0, 'ng': 0, 'defect': 0, 'quality': 0
        }
        self.update_kpis() # Refresh display immediately

        self.is_running = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_new_op.setEnabled(False)
        
        self.update_status(f"SISTEMA EM EXECUÇÃO ({self.current_job['op']}) - Pressione 'T' para Inspecionar", 0)
        
        # Set Start Timestamp
        # Use OP start time if available, otherwise now AND SAVE IT
        if self.current_job and 'start_time' in self.current_job:
            start_str = self.current_job['start_time']
            try:
                dt = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
                disp_str = dt.strftime('%d/%m/%Y %H:%M')
            except:
                disp_str = start_str
        else:
            now = datetime.now()
            # SAVE the start time so it persists for this "Default" job
            if not self.current_job:
                 self.current_job = {"sku": "Bom Petisco Oleo - rr125", "op": "DEFAULT", "qty": 0}
            
            self.current_job['start_time'] = now.strftime("%Y-%m-%d %H:%M:%S")
            disp_str = now.strftime('%d/%m/%Y %H:%M')
            
        if hasattr(self, 'lbl_stats_start'):
             #print(f"DEBUG: Setting Start Time to {disp_str}")
             self.lbl_stats_start.setText(disp_str)
             # Always reset end time when starting/resuming
             if hasattr(self, 'lbl_stats_end'):
                 self.lbl_stats_end.setText("--/-- --:--")
        else:
             print("DEBUG: lbl_stats_start NOT FOUND")
        
        # Force focus back to main window so KeyPressEvent works immediately
        self.setFocus()
        
    def stop_inspection_mode(self):
        self.is_running = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        # self.btn_new_op.setEnabled(True) # Only enable on Finish OP
        
        self.update_status("SISTEMA EM PAUSA", 0)
        
        # Set Stop Timestamp (Pause Time)
        now = datetime.now()
        if hasattr(self, 'lbl_stats_end'):
            stop_str = now.strftime('%d/%m/%Y %H:%M')
            #print(f"DEBUG: Setting Stop Time to {stop_str}")
            self.lbl_stats_end.setText(stop_str)
        else:
            print("DEBUG: lbl_stats_end NOT FOUND")

    def keyPressEvent(self, event):
        # Trigger inspection with 'T' key only if running
        if event.key() == Qt.Key_T:
            if self.is_running:
                self.trigger_inspection()
            else:
                self.update_status("AVISO: Sistema PARADO. Pressione INICIAR primeiro.", 2000)
        else:
            super().keyPressEvent(event)

    def trigger_inspection(self):
        """
        Capture current frame and trigger inspection for ALL cans.
        """
        if not self.is_running:
            self.update_status("Ignorado: Sistema PARADO.", 1000)
            return

        if self.camera_thread and self.camera_thread.running:
            # Check if inferencer is loaded
            if self.inferencer is None:
                self.update_status("ERROR: PatchCore model not loaded! Cannot inspect.", 5000)
                return
                
            # Briefly unpause to get fresh frame (camera keeps updating current_frame in background)
            # Briefly unpause to get fresh frame (camera keeps updating current_frame in background)
            self.camera_thread.paused = False
            
            # Capture trigger time for first can timing
            import time
            self.trigger_time = time.time()
            self.first_can_complete_time = None # Reset for this run
            
            # Set flag to capture NEXT available frame in update_frame
            self.pending_inspection = True
            
            # Start safety timeout (2 seconds)
            self.inspection_timeout_timer.start(2000) 
            self.update_status("A aguardar frame...", 2000)
            
    def on_inspection_timeout(self):
        """Called if no frame arrives within timeout period"""
        self.pending_inspection = False
        self.update_status("Erro: Timeout ao aguardar frame da câmara.", 4000)
        # Ensure we don't leave camera running if we wanted to stop it? 
        # Actually logic is we want it running to get a frame. 
        # But if it failed, maybe we should repause if that was the intent, 
        # but here we just leave it as is or user can try again.
            
    def _start_inspection(self):
        """Helper to start inspection with captured frame"""
        if self.current_frame is None:
            self.update_status("Sem sinal de câmara!", 2000)
            return
        
        # Capture the current frame for inspection
        inspection_frame = self.current_frame.copy()
        
        # Pause camera to freeze display during and after processing
        self.camera_thread.paused = True

        # Start Worker with captured frame
        self.update_status("A Inspecionar...", 1000)
        
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
        self.worker.progress.connect(lambda msg: self.update_status(msg))
        self.worker.error.connect(self.on_inspection_error)
        self.worker.finished.connect(self.on_inspection_finished)
        self.worker.progressive_update.connect(self.on_progressive_update)  # NEW
        self.worker.start()

    def on_progressive_update(self, qt_image, current, total, scores):
        """Update display progressively as cans are inspected"""
        # Capture time when first can completes
        if current == 1:
            print(f"DEBUG: on_progressive_update called for CAN 1. Trigger: {self.trigger_time}")
            if self.trigger_time is not None and self.first_can_complete_time is None:
                import time
                self.first_can_complete_time = time.time()
                print(f"DEBUG: First can complete at {self.first_can_complete_time} (Trigger: {self.trigger_time})")
        
        self.update_image_display(qt_image)
        self.update_status(f"A Inspecionar... {current}/{total} latas", 500)
        self.update_graph(scores, self.threshold_spinbox.value())

    def on_inspection_error(self, msg):
        self.update_status(f"Error: {msg}", 4000)
        # Unpause on error so user can try again
        if self.camera_thread:
            self.camera_thread.paused = False

    def on_inspection_finished(self, qt_image, stats, logs, results_data, clean_image, completion_time):
        self.update_status("Inspeção Concluída", 2000)
        
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
        
        # Update Session Counters
        self.session_counts['total'] += total_batch
        self.session_counts['ok'] += ok_batch
        self.session_counts['ng'] += ng_batch
        self.session_counts['defect'] += defect_batch
        self.session_counts['quality'] += quality_batch
        
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
        
        self.lbl_stats_avg_time.setText(f"{avg_time:.1f}ms")
        
        # Calculate time to first can (trigger to first can analysis complete)
        # Use timing from worker stats is more reliable
        t_first = stats.get('t_first_can')
        
        print(f"DEBUG: Finished. Trigger: {self.trigger_time}, FirstCan (Stats): {t_first}")
        
        # Fallback if None (shouldn't happen if cans > 0)
        if t_first is None and total_batch > 0:
             print("DEBUG: t_first is None but batch > 0. Logic error in Worker?")
        
        if self.trigger_time is not None and t_first is not None:
            total_time_to_first = (t_first - self.trigger_time) * 1000  # Convert to ms
            self.lbl_stats_first_can.setText(f"{total_time_to_first:.0f}ms")
        elif self.trigger_time is not None and self.first_can_complete_time is not None:
             # Fallback to main thread timing if worker timing is missing
             print("WARNING: Using main thread timing for first can.")
             total_time_to_first = (self.first_can_complete_time - self.trigger_time) * 1000
             self.lbl_stats_first_can.setText(f"{total_time_to_first:.0f}ms")
        else:
            print(f"DEBUG: First Can Time Missing. Trigger: {self.trigger_time}, t_first(Worker): {t_first}, t_first(Main): {self.first_can_complete_time}")
            self.lbl_stats_first_can.setText("--ms")
        
        # Calculate total inspection time (trigger to all cans complete)
        # Use completion_time from worker instead of time.time() for accuracy
        if self.trigger_time and completion_time:
            total_inspection_time = completion_time - self.trigger_time  # Keep in seconds
            self.lbl_stats_total_time.setText(f"{total_inspection_time:.3f}s")
        else:
            self.lbl_stats_total_time.setText("--s")
        
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
        self.update_status(f"Inspection Complete. {timing_msg}", 3000)

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
                # Draw Label (Match InspectionWorker styling)
                label = f"#{can_id} {status_text}\n{score:.2f}"
                font_scale = 1.2
                thickness = 2
                
                # Split lines
                lines = label.replace('\r', '').split('\n')
                
                # Calculate box size
                max_width = 0
                total_height = 0
                line_heights = []
                
                for line in lines:
                   (txt_w, txt_h), baseline = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                   max_width = max(max_width, txt_w)
                   h_with_padding = txt_h + baseline + 10
                   line_heights.append(h_with_padding)
                   total_height += h_with_padding
                   
                # Draw background box
                box_buffer = 10
                box_top_y = y1
                box_bottom_y = y1 + total_height + box_buffer
                
                cv2.rectangle(annotated_sheet, (x1, box_top_y), (x1 + max_width + 10, box_bottom_y), color, -1)
                
                text_color = (0, 0, 0) if is_normal else (255, 255, 255)
                
                # Draw each line
                current_y = box_top_y + box_buffer
                for i, line in enumerate(lines):
                    (txt_w, txt_h), baseline = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    current_y += txt_h 
                    
                    cv2.putText(annotated_sheet, line, (x1 + 5, current_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
                    
                    current_y += baseline + 10

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
        """Update KPI cards (Global) and Stats Panel (Session)"""
        
        # 1. Left Panel (Global OP Stats)
        yield_pct = (self.ok_count / self.total_count * 100) if self.total_count > 0 else 0
        
        self.update_kpi(self.kpi_total, self.total_count)
        self.update_kpi(self.kpi_ok, self.ok_count)
        self.update_kpi(self.kpi_ng, self.ng_count)
        self.update_kpi(self.kpi_yield, f"{yield_pct:.1f}%")
        
        # 2. Right Panel (Session Stats)
        s_total = self.session_counts['total']
        s_ok = self.session_counts['ok']
        s_ng = self.session_counts['ng']
        
        s_good_pct = (s_ok / s_total * 100) if s_total > 0 else 0
        s_bad_pct = (s_ng / s_total * 100) if s_total > 0 else 0
        
        self.lbl_stats_total.setText(str(s_total))
        # Start/End times are handled in start/stop methods
        # avg time? Need to track session duration or something? For now leave as is or use global
        self.lbl_stats_good_pct.setText(f"{s_good_pct:.1f}%")
        self.lbl_stats_bad_pct.setText(f"{s_bad_pct:.1f}%")

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

    def open_reports_dialog(self):
        """Open the production reports viewer"""
        dlg = ReportsDialog(self)
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
        dlg = DefectsGalleryDialog(defects_dir=os.path.join('data', 'defects'), parent=self)
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
        
        # Prepare image for display: Draw Defect Circle if Heatmap provided
        image = item.get('image')
        heatmap = item.get('heatmap')
        
        display_img = image
        if heatmap is not None and status == "NG":
            try:
                # Find location of max anomaly in heatmap
                # Heatmap is 448x448 float array
                h_map = heatmap
                # print(f"DEBUG: Heatmap shape: {h_map.shape}, type: {h_map.dtype}")

                if len(h_map.shape) > 2:
                    h_map = h_map.squeeze() 
                
                # If still 3 dims, take first channel or max across channels? 
                # Usually it's (H, W). If (H, W, 3), that's wrong for a single heatmap.
                if len(h_map.shape) == 3:
                     # If RGB heatmap was passed (mistake), convert to gray?
                     # But PatchCore returns (1, H, W) usually.
                     if h_map.shape[0] == 1:
                         h_map = h_map[0]
                     elif h_map.shape[2] == 1:
                         h_map = h_map[:, :, 0]
                     else:
                         # Fallback: take max across channels
                         h_map = np.max(h_map, axis=2)

                # Ensure float32 or uint8
                if h_map.dtype != np.float32 and h_map.dtype != np.uint8:
                     h_map = h_map.astype(np.float32)

                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(h_map)
                
                # Convert original image to BGR for drawing if not already
                # Actually image here is usually RGB from QImage conversion logic? 
                # Let's assume it's an numpy array, likely RGB or BGR.
                # If it comes from 'image' key in results_data, it's the cropped numpy array (BGR usually)
                display_img = image.copy()
                
                # If heatmap is smaller/larger, we might need to scale coordinates?
                # Usually both are 448x448.
                
                # Draw Circle
                cv2.circle(display_img, max_loc, 40, (0, 0, 255), 3) # Red Circle, Radius 40
                
                # Optional: Crosshair
                # cv2.drawMarker(display_img, max_loc, (0, 0, 255), markerType=cv2.MARKER_CROSS, thickness=2)
                
            except Exception as e:
                print(f"Error drawing defect circle: {e}")

        dlg = CanDetailDialog(
            can_id=item['can_id'],
            score=score,
            status=status,
            image=display_img,
            heatmap=item.get('heatmap'),
            threshold=threshold,
            parent=self
        )
        dlg.exec()
        
    def on_finish_op(self):
        """Finish current OP, generate report and clear state"""
        if not self.current_job:
            return
            
        # Confirm
        from PySide6.QtWidgets import QMessageBox
        reply = QMessageBox.question(self, 'Confirmar Término', 
                                   f"Deseja realmente terminar a OP {self.current_job['op']}?\nA inspeção será parada.",
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                                   
        if reply == QMessageBox.No:
            return
            
        # Stop everything
        if self.is_running:
            self.stop_inspection_mode()
            
        # Generate Stats
        end_time = datetime.now()
        start_time_str = self.current_job.get("start_time", "N/A")
        

        # Get username
        username = self.current_user.username if self.current_user else "N/A"

        report_text = f"""
========================================
RELATÓRIO DE PRODUÇÃO - {self.current_job['sku']}
========================================
OP: {self.current_job['op']}
Usuario: {username}
Data Início: {start_time_str}
Data Fim:    {end_time.strftime('%Y-%m-%d %H:%M:%S')}

Estatísticas Finais:
-------------------
Total Produzido: {self.total_count}
Aprovadas (OK):  {self.ok_count}
Reprovadas (NG): {self.ng_count}
  - Defeitos:    {self.defect_count}
  - Qualidade:   {self.quality_count}
  
Yield (Aprov.%): {(self.ok_count/self.total_count*100) if self.total_count > 0 else 0:.2f}%
========================================
"""
        # Save Report
        try:
            if not os.path.exists("reports"):
                os.makedirs("reports")
            report_name = f"reports/Report_{self.current_job['op']}_{end_time.strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_name, "w") as f:
                f.write(report_text)
            print(f"Report saved to {report_name}")
            
            # Delete State File regarding this OP
            if os.path.exists("data/current_op.json"):
                os.remove("data/current_op.json")
                print("OP State file removed.")
                
        except Exception as e:
            print(f"Failed to save report or clear state: {e}")
            
        # Show Summary
        QMessageBox.information(self, "OP Finalizada", report_text)
        
        # Reset State
        self.current_job = None
        self.lbl_current_op.setText("SEM OP ATIVA")
        
        # Reset Buttons
        self.btn_new_op.setEnabled(True)
        self.btn_start.setEnabled(False)
        self.btn_finish_op.setEnabled(False)
        self.btn_stop.setEnabled(False)
        
        # Reset Counters (Optional, user might expect clean slate for next OP)
        self.total_count = 0
        self.ok_count = 0
        self.ng_count = 0
        self.defect_count = 0
        self.quality_count = 0
        self.update_kpis()
        self.update_donut()

    def update_status(self, message, timeout=0):
        """Custom status update to preserve centered OP label"""
        if hasattr(self, 'lbl_status_msg'):
            self.lbl_status_msg.setText(message)
            if timeout > 0:
                QTimer.singleShot(timeout, lambda: self.lbl_status_msg.setText("Pronto"))
        else:
            # Fallback if init failed
            self.statusBar().showMessage(message, timeout)

    def closeEvent(self, event):
        from PySide6.QtWidgets import QMessageBox
        if self.is_running:
            reply = QMessageBox.question(self, 'Alerta de Saída',
                "⚠️ A inspeção está em andamento!\n\nDeseja realmente sair e PARAR a inspeção?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.No:
                event.ignore()
                return

        # Save state before closing
        self.save_op_state()
        
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
    def save_op_state(self):
        """Save current OP state to JSON for persistence"""
        if not self.current_job:
            return

        state = {
            "job": self.current_job,
            "stats": {
                "total": self.total_count,
                "ok": self.ok_count,
                "ng": self.ng_count,
                "defect": self.defect_count,
                "quality": self.quality_count
            }
        }
        
        try:
            if not os.path.exists("data"):
                os.makedirs("data")
            with open("data/current_op.json", "w") as f:
                json.dump(state, f)
            print("OP State saved.")
        except Exception as e:
            print(f"Failed to save OP state: {e}")

    def load_op_state(self):
        """Load OP state from JSON if exists"""
        if not os.path.exists("data/current_op.json"):
            return

        try:
            with open("data/current_op.json", "r") as f:
                state = json.load(f)
            
            self.current_job = state.get("job")
            stats = state.get("stats", {})
            
            self.total_count = stats.get("total", 0)
            self.ok_count = stats.get("ok", 0)
            self.ng_count = stats.get("ng", 0)
            self.defect_count = stats.get("defect", 0)
            self.quality_count = stats.get("quality", 0)
            
            # Restore UI Context
            if self.current_job:
                op_id = self.current_job.get("op", "UNKNOWN")
                sku = self.current_job.get("sku", "UNKNOWN")
                
                # Load the appropriate model for this SKU
                print(f"Restoring OP state for SKU: {sku}")
                self.load_model_for_sku(sku)
                
                # Update Status Bar
                if hasattr(self, 'lbl_current_op'):
                   self.lbl_current_op.setText(f"OP EM CURSO: {op_id} ({sku})")
                
                # Enable Buttons
                self.btn_new_op.setEnabled(False)
                self.btn_start.setEnabled(True)
                self.btn_finish_op.setEnabled(True)
                
                # Update KPIs
                self.update_kpis()
                self.update_donut()
                
                print(f"OP State loaded: {op_id}")
                
        except Exception as e:
            print(f"Failed to load OP state: {e}")

    def handle_plc_trigger(self):
        """Thread-safe PLC trigger handler"""
        self.sig_plc_trigger.emit()

    def open_io_dialog(self):
        dlg = IODialog(self.plc, self)
        dlg.exec()

class IODialog(QDialog):
    def __init__(self, plc_manager, parent=None):
        super().__init__(parent)
        self.plc = plc_manager
        self.setWindowTitle("Diagnóstico I/O (Raspberry Pi)")
        self.resize(500, 400)
        self.setStyleSheet("background-color: #0D1117; color: #E6EDF3;")
        
        layout = QVBoxLayout(self)
        
        # Header
        lbl_info = QLabel("Estado das Entradas e Saídas Digitais")
        lbl_info.setStyleSheet("font-size: 16px; font-weight: bold; color: #58A6FF; margin-bottom: 10px;")
        lbl_info.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl_info)
        
        # Content Split
        content_layout = QHBoxLayout()
        
        # Inputs Group
        in_group = QGroupBox("📥 Entradas (Inputs)")
        in_group.setStyleSheet("QGroupBox { border: 1px solid #30363D; border-radius: 6px; margin-top: 20px; } QGroupBox::title { color: #58A6FF; }")
        self.in_layout = QVBoxLayout()
        in_group.setLayout(self.in_layout)
        
        # Outputs Group
        out_group = QGroupBox("📤 Saídas (Outputs)")
        out_group.setStyleSheet("QGroupBox { border: 1px solid #30363D; border-radius: 6px; margin-top: 20px; } QGroupBox::title { color: #58A6FF; }")
        self.out_layout = QVBoxLayout()
        out_group.setLayout(self.out_layout)
        
        content_layout.addWidget(in_group)
        content_layout.addWidget(out_group)
        layout.addLayout(content_layout)
        
        # Refresh Timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_io_status)
        self.timer.start(200) # 5Hz Refresh
        
        # Initial Draw
        self.input_widgets = {}
        self.output_widgets = {}
        self.init_ui()
        
        # Close Button
        btn_close = QPushButton("Fechar")
        btn_close.clicked.connect(self.accept)
        btn_close.setStyleSheet("background-color: #21262D; border: 1px solid #30363D; padding: 8px;")
        layout.addWidget(btn_close)
        
    def init_ui(self):
        # Create widgets for Inputs
        inputs = self.plc.get_input_states()
        for name, state in inputs.items():
            led = QLabel()
            led.setFixedSize(16, 16)
            lbl = QLabel(name)
            
            row = QHBoxLayout()
            row.addWidget(led)
            row.addWidget(lbl)
            row.addStretch()
            self.in_layout.addLayout(row)
            self.input_widgets[name] = led

        # Create widgets for Outputs
        outputs = self.plc.get_output_states()
        for name, state in outputs.items():
            led = QLabel()
            led.setFixedSize(16, 16)
            lbl = QLabel(name)
            
            row = QHBoxLayout()
            row.addWidget(led)
            row.addWidget(lbl)
            row.addStretch()
            self.out_layout.addLayout(row)
            self.output_widgets[name] = led
            
        self.update_io_status()
            
    def update_io_status(self):
        # Update Inputs
        inputs = self.plc.get_input_states()
        for name, state in inputs.items():
            if name in self.input_widgets:
                color = "#2EA043" if state else "#30363D" # Green if ON, Dark Grey if OFF
                self.input_widgets[name].setStyleSheet(f"background-color: {color}; border-radius: 8px; border: 1px solid #555;")
                
        # Update Outputs
        outputs = self.plc.get_output_states()
        for name, state in outputs.items():
            if name in self.output_widgets:
                color = "#238636" if state else "#30363D" # Green if ON
                self.output_widgets[name].setStyleSheet(f"background-color: {color}; border-radius: 8px; border: 1px solid #555;")
