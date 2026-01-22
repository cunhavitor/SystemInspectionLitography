from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QComboBox, QListWidget, 
                             QMessageBox, QGroupBox)
from PySide6.QtCore import Qt

class UserManagementDialog(QDialog):
    def __init__(self, user_manager, parent=None):
        super().__init__(parent)
        self.user_manager = user_manager
        self.setWindowTitle("User Management")
        self.setFixedSize(500, 400)
        
        self.layout = QVBoxLayout(self)
        
        self._setup_user_list()
        self._setup_add_user_form()
        
        self.refresh_user_list()

    def _setup_user_list(self):
        group = QGroupBox("Existing Users")
        layout = QVBoxLayout()
        
        self.user_list_widget = QListWidget()
        layout.addWidget(self.user_list_widget)
        
        delete_btn = QPushButton("Delete Selected User")
        delete_btn.clicked.connect(self.delete_user)
        layout.addWidget(delete_btn)
        
        group.setLayout(layout)
        self.layout.addWidget(group)

    def _setup_add_user_form(self):
        group = QGroupBox("Add New User")
        layout = QVBoxLayout()
        
        # Username
        h_layout1 = QHBoxLayout()
        h_layout1.addWidget(QLabel("Username:"))
        self.username_input = QLineEdit()
        h_layout1.addWidget(self.username_input)
        layout.addLayout(h_layout1)
        
        # Password
        h_layout2 = QHBoxLayout()
        h_layout2.addWidget(QLabel("Password:"))
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        h_layout2.addWidget(self.password_input)
        layout.addLayout(h_layout2)
        
        # Role
        h_layout3 = QHBoxLayout()
        h_layout3.addWidget(QLabel("Role:"))
        self.role_combo = QComboBox()
        self.role_combo.addItems(["operador", "tecnico", "admin"])
        h_layout3.addWidget(self.role_combo)
        layout.addLayout(h_layout3)
        
        add_btn = QPushButton("Add User")
        add_btn.clicked.connect(self.add_user)
        layout.addWidget(add_btn)
        
        group.setLayout(layout)
        self.layout.addWidget(group)

    def refresh_user_list(self):
        self.user_list_widget.clear()
        users = self.user_manager.get_all_users()
        for user in users:
            self.user_list_widget.addItem(f"{user.username} ({user.role})")

    def add_user(self):
        username = self.username_input.text().strip()
        password = self.password_input.text().strip()
        role = self.role_combo.currentText()
        
        if not username or not password:
            QMessageBox.warning(self, "Error", "Username and password are required.")
            return

        try:
            if self.user_manager.add_user(username, password, role):
                QMessageBox.information(self, "Success", f"User '{username}' added successfully.")
                self.username_input.clear()
                self.password_input.clear()
                self.refresh_user_list()
            else:
                QMessageBox.warning(self, "Error", "User already exists.")
        except ValueError as e:
            QMessageBox.warning(self, "Error", str(e))

    def delete_user(self):
        current_item = self.user_list_widget.currentItem()
        if not current_item:
            return
            
        username = current_item.text().split(" (")[0]
        
        if username == self.user_manager.current_user.username:
             QMessageBox.warning(self, "Error", "You cannot delete yourself.")
             return

        reply = QMessageBox.question(self, "Confirm Delete", 
                                   f"Are you sure you want to delete user '{username}'?",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            if self.user_manager.delete_user(username):
                self.refresh_user_list()
            else:
                 QMessageBox.warning(self, "Error", "Could not delete user (cannot delete admin).")
