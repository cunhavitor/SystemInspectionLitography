from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QLabel, QPushButton, QGridLayout, QMessageBox, QSizePolicy)
from PySide6.QtCore import Qt
from .user_management import UserManagementDialog
from .inspection_window import InspectionWindow
from .dataset_window import DatasetWindow
from .adjustment_window import AdjustmentWindow

class DashboardWindow(QMainWindow):
    def __init__(self, user_manager, config):
        super().__init__()
        self.user_manager = user_manager
        self.config = config
        
        self.inspection_window = None # Persist inspection window
        
        self.setWindowTitle("Inspection System - Dashboard")
        self.resize(800, 600)
        
        self._setup_ui()
        
    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(40, 40, 40, 40)
        
        # Header
        header = QHBoxLayout()
        warning_icon = "👁️" # Unicode icon
        title = QLabel(f"{warning_icon} Inspection System")
        title.setObjectName("titleLabel") # For QSS
        header.addWidget(title)
        
        header.addStretch()
        
        user_info = QLabel(f"👤 {self.user_manager.current_user.username}")
        header.addWidget(user_info)
        
        logout_btn = QPushButton("Logout")
        logout_btn.setFixedWidth(100)
        logout_btn.clicked.connect(self.logout)
        header.addWidget(logout_btn)
        
        main_layout.addLayout(header)
        
        # Divider
        line = QLabel()
        line.setStyleSheet("border-bottom: 2px solid #333333; margin-bottom: 20px;")
        main_layout.addWidget(line)
        
        # Grid of Action Buttons
        grid_layout = QGridLayout()
        grid_layout.setSpacing(30)
        main_layout.addLayout(grid_layout)
        
        # Helper to create big buttons
        def create_card(title, subtitle, icon, callback, enabled=True):
            btn = QPushButton()
            btn.setObjectName("dashboardCard") # For QSS styling
            btn.setMinimumSize(250, 180)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            btn.clicked.connect(callback)
            btn.setEnabled(enabled)
            
            # Layout
            btn_layout = QVBoxLayout(btn)
            
            lbl_icon = QLabel(icon)
            lbl_icon.setAlignment(Qt.AlignCenter)
            lbl_icon.setStyleSheet("font-size: 48px; background: transparent; border: none;")
            
            lbl_title = QLabel(title)
            lbl_title.setAlignment(Qt.AlignCenter)
            lbl_title.setStyleSheet("font-size: 20px; font-weight: bold; background: transparent; border: none; margin-top: 10px;")
            
            lbl_desc = QLabel(subtitle)
            lbl_desc.setAlignment(Qt.AlignCenter)
            lbl_desc.setWordWrap(True)
            lbl_desc.setStyleSheet("color: #888888; background: transparent; border: none;")
            
            btn_layout.addWidget(lbl_icon)
            btn_layout.addWidget(lbl_title)
            btn_layout.addWidget(lbl_desc)
            btn_layout.addStretch()
            
            return btn
            
        # 1. Inspection Mode
        btn_inspection = create_card(
            "Inspection", 
            "Start real-time product inspection and defect detection.", 
            "🔍",
            self.open_inspection
        )
        grid_layout.addWidget(btn_inspection, 0, 0)
        
        # 2. Dataset Mode
        btn_dataset = create_card(
            "Dataset", 
            "Capture and label images for training the AI models.", 
            "📸",
            self.open_dataset
        )
        grid_layout.addWidget(btn_dataset, 0, 1)
        
        # 3. Adjustment Mode (Restricted)
        is_admin = self.user_manager.current_user.role in ['admin', 'tecnico', 'master']
        btn_adjust = create_card(
            "Calibration", 
            "Fine-tune camera parameters, focus, and exposure.", 
            "⚙️",
            self.open_adjustment,
            enabled=is_admin
        )
        grid_layout.addWidget(btn_adjust, 1, 0)
        
        # 4. User Management (Restricted)
        is_master = self.user_manager.current_user.role in ['admin', 'master']
        btn_users = create_card(
            "Users", 
            "Manage user accounts, roles, and access permissions.", 
            "👥",
            self.open_users,
            enabled=is_master
        )
        grid_layout.addWidget(btn_users, 1, 1)
        
        main_layout.addStretch()
        
        # Footer
        footer = QLabel(f"System v1.0 • {self.config.get('app', {}).get('name', 'Vision Core')}")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("color: #555555; margin-top: 20px;")
        main_layout.addWidget(footer)

    def open_inspection(self):
        self.hide()
        if self.inspection_window is None:
            self.inspection_window = InspectionWindow(self.config, self)
        self.inspection_window.show()
        if self.windowState() == Qt.WindowMaximized:
            self.inspection_window.showMaximized()
        
    def open_dataset(self):
        self.hide()
        self.window = DatasetWindow(self.config, self)
        self.window.show()
        
    def open_adjustment(self):
        self.hide()
        self.window = AdjustmentWindow(self.config, self)
        self.window.show()
        
    def open_users(self):
        dialog = UserManagementDialog(self.user_manager, self)
        dialog.exec()
        
    def logout(self):
        self.user_manager.logout()
        self.close()
        # Initial login window showing logic handled by main.py
