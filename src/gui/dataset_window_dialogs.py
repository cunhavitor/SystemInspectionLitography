from PySide6.QtWidgets import QDialog, QVBoxLayout, QFormLayout, QLineEdit, QPushButton, QHBoxLayout, QDialogButtonBox, QFileDialog, QWidget

class AddNewSKUDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add New SKU")
        self.setMinimumWidth(500)
        self.setStyleSheet("background-color: #2b2b2b; color: #ffffff;")
        
        layout = QVBoxLayout(self)
        
        form_layout = QFormLayout()
        
        self.name_input = QLineEdit()
        self.name_input.setStyleSheet("background-color: #444; color: white; padding: 5px; border: 1px solid #555;")
        form_layout.addRow("SKU Name:", self.name_input)
        
        # Model Path
        self.model_path_input = QLineEdit()
        self.model_path_input.setReadOnly(True)
        self.model_path_input.setStyleSheet("background-color: #444; color: #aaa; padding: 5px; border: 1px solid #555;")
        btn_browse_model = QPushButton("Browse...")
        btn_browse_model.clicked.connect(self.browse_model)
        btn_browse_model.setStyleSheet("background-color: #555; color: white; padding: 5px;")
        
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_path_input)
        model_layout.addWidget(btn_browse_model)
        form_layout.addRow("PatchCore Model:", model_layout)
        
        # Reference Image Path
        self.ref_path_input = QLineEdit()
        self.ref_path_input.setReadOnly(True)
        self.ref_path_input.setStyleSheet("background-color: #444; color: #aaa; padding: 5px; border: 1px solid #555;")
        btn_browse_ref = QPushButton("Browse...")
        btn_browse_ref.clicked.connect(self.browse_ref)
        btn_browse_ref.setStyleSheet("background-color: #555; color: white; padding: 5px;")
        
        ref_layout = QHBoxLayout()
        ref_layout.addWidget(self.ref_path_input)
        ref_layout.addWidget(btn_browse_ref)
        form_layout.addRow("Reference Image:", ref_layout)
        
        layout.addLayout(form_layout)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.setStyleSheet("QPushButton { background-color: #555; color: white; padding: 5px 15px; } QPushButton:hover { background-color: #666; }")
        
        layout.addWidget(button_box)
        
    def browse_model(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select PatchCore Model", "", "Model Files (*.pth *.pt);;All Files (*)")
        if filename:
            self.model_path_input.setText(filename)
            
    def browse_ref(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Reference Image", "", "Images (*.png *.jpg *.jpeg);;All Files (*)")
        if filename:
            self.ref_path_input.setText(filename)
            
    def get_data(self):
        return {
            "name": self.name_input.text(),
            "model_path": self.model_path_input.text(),
            "reference_path": self.ref_path_input.text()
        }
