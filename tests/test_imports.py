import sys
try:
    from PySide6.QtWidgets import QApplication
    from src.gui.login import LoginWindow
    from src.gui.main_window import MainWindow
    from src.auth import UserManager
    print("Imports successful")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)
