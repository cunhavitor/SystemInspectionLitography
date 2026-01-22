import sys
import yaml
import time
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from PySide6.QtCore import Qt
from qt_material import apply_stylesheet

from src.auth import UserManager
from src.gui.login import LoginWindow
from src.gui.dashboard import DashboardWindow
from src.gui.splash_screen import SplashScreen

def load_config(path="config/settings.yaml"):
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {path}")
        sys.exit(1)

import os

def main():
    app = QApplication(sys.argv)
    
    # Get project directory for resource paths
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create and show splash screen
    splash_image_path = os.path.join(project_dir, "splash.png")
    splash = SplashScreen(splash_image_path)
    splash.show()
    splash.showMessage("Initializing application...", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter)
    
    # Track start time to ensure minimum splash display duration
    splash_start_time = time.time()
    
    # 1. fix: Link the running app to the .desktop file so the taskbar uses that icon
    app.setDesktopFileName("SIL")
    
    # 2. fix: Use absolute path to ensure icon loads regardless of where script is run
    icon_path = "/home/cunhav/projects/InspectionVisionCamera/icon.png"
    app.setWindowIcon(QIcon(icon_path))
    
    # Load Config and User Manager
    splash.showMessage("Loading configuration...", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter)
    config = load_config()
    
    splash.showMessage("Initializing user manager...", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter)
    user_manager = UserManager()
    
    # Load Stylesheet
    def apply_styles():
        try:
            with open("src/gui/styles.qss", "r") as f:
                app.setStyleSheet(f.read())
        except FileNotFoundError:
            print("Warning: styles.qss not found")

    splash.showMessage("Applying styles...", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter)
    apply_styles()

    # Login Loop
    while True:
        splash.showMessage("Ready - Please login...", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter)
        login_window = LoginWindow(user_manager)
        
        # Set icon on login window explicitly
        login_window.setWindowIcon(QIcon(icon_path))
        
        # Ensure splash screen shows for minimum 5 seconds
        elapsed_time = time.time() - splash_start_time
        min_splash_duration = 5.0  # seconds
        if elapsed_time < min_splash_duration:
            remaining_time = min_splash_duration - elapsed_time
            time.sleep(remaining_time)
        
        # Close splash screen before showing login
        splash.finish(login_window)
        
        if login_window.exec():
            # Login Success
            dashboard_window = DashboardWindow(user_manager, config)
            # Set icon on dashboard window explicitly
            dashboard_window.setWindowIcon(QIcon(icon_path))
            # Re-apply styles ensures any dynamic elements get it, though app-level usually enough
            dashboard_window.show()
            
            # Application runs here
            app.exec()
            
            # If application closes, check if we should loop back to login (logout logic)
            # For simplicity, MainWindow closes app completely on close, 
            # and Logout button closes window.
            # So if we are here, main_window is closed. 
            # If it was a logout, we might want to show login again.
            # But implementing full restart logic requires more state management.
            # Current implementation: Login -> Main -> Exit.
            # To support "Logout" returning to Login, we need to handle the exit code or state.
            
            if user_manager.current_user is None: 
                # Meaning we logged out, loop continues
                continue
            else:
                # We exited normally
                break
        else:
            # Login Cancelled
            break

if __name__ == "__main__":
    main()
