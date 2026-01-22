import sys
import yaml
from PySide6.QtWidgets import QApplication
from qt_material import apply_stylesheet

from src.auth import UserManager
from src.gui.login import LoginWindow
from src.gui.dashboard import DashboardWindow

def load_config(path="config/settings.yaml"):
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {path}")
        sys.exit(1)

def main():
    app = QApplication(sys.argv)
    
    # Load Config and User Manager
    config = load_config()
    user_manager = UserManager()
    
    # Load Stylesheet
    def apply_styles():
        try:
            with open("src/gui/styles.qss", "r") as f:
                app.setStyleSheet(f.read())
        except FileNotFoundError:
            print("Warning: styles.qss not found")

    apply_styles()

    # Login Loop
    while True:
        login_window = LoginWindow(user_manager)
        if login_window.exec():
            # Login Success
            dashboard_window = DashboardWindow(user_manager, config)
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
