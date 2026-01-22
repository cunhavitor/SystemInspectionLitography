from PySide6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QMessageBox)
from PySide6.QtCore import Signal

class LoginWindow(QDialog):
    login_successful = Signal(str)  # Signal emitting the username upon success

    def __init__(self, user_manager):
        super().__init__()
        self.user_manager = user_manager
        self.setWindowTitle("Login - Inspection Vision")
        self.setFixedSize(300, 150)
        
        layout = QVBoxLayout()
        
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Username")
        layout.addWidget(self.username_input)
        
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Password")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(self.password_input)
        
        self.login_btn = QPushButton("Login")
        self.login_btn.clicked.connect(self.attempt_login)
        layout.addWidget(self.login_btn)
        
        self.setLayout(layout)

    def attempt_login(self):
        username = self.username_input.text()
        password = self.password_input.text()
        
        if self.user_manager.login(username, password):
            self.login_successful.emit(username)
            self.accept()
        else:
            QMessageBox.warning(self, "Login Failed", "Invalid username or password")
