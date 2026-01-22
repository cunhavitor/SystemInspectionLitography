from PySide6.QtWidgets import QSplashScreen, QApplication
from PySide6.QtGui import QPixmap, QPainter, QColor, QFont
from PySide6.QtCore import Qt, QTimer
import os


class SplashScreen(QSplashScreen):
    """
    Custom splash screen for the Inspection Vision Camera application.
    Shows during application startup with loading status updates.
    """
    
    def __init__(self, splash_image_path=None):
        # Load splash image
        if splash_image_path and os.path.exists(splash_image_path):
            pixmap = QPixmap(splash_image_path)
        else:
            # Create a default splash screen with logo
            project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            logo_path = os.path.join(project_dir, "logo.png")
            
            # Create splash screen background
            pixmap = QPixmap(800, 600)
            pixmap.fill(QColor(20, 30, 50))  # Dark blue background
            
            # Draw logo if available
            if os.path.exists(logo_path):
                painter = QPainter(pixmap)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                
                # Load and scale logo
                logo = QPixmap(logo_path)
                logo_scaled = logo.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio, 
                                        Qt.TransformationMode.SmoothTransformation)
                
                # Center logo
                x = (pixmap.width() - logo_scaled.width()) // 2
                y = (pixmap.height() - logo_scaled.height()) // 2 - 50
                painter.drawPixmap(x, y, logo_scaled)
                
                # Add application title
                painter.setPen(QColor(255, 255, 255))
                font = QFont("Arial", 24, QFont.Weight.Bold)
                painter.setFont(font)
                painter.drawText(pixmap.rect().adjusted(0, 350, 0, 0), 
                               Qt.AlignmentFlag.AlignCenter, 
                               "System Inspection Lithography")
                
                # Add version/subtitle
                font_small = QFont("Arial", 12)
                painter.setFont(font_small)
                painter.setPen(QColor(180, 180, 180))
                painter.drawText(pixmap.rect().adjusted(0, 400, 0, 0), 
                               Qt.AlignmentFlag.AlignCenter, 
                               "Vision Inspection System")
                
                painter.end()
        
        super().__init__(pixmap)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)
        
    def showMessage(self, message, alignment=Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter, 
                   color=QColor(255, 255, 255)):
        """
        Display a status message on the splash screen.
        
        Args:
            message: Text to display
            alignment: Text alignment (default: bottom center)
            color: Text color (default: white)
        """
        super().showMessage(message, alignment, color)
        QApplication.processEvents()
        
    def finish_splash(self, main_window, delay_ms=500):
        """
        Close the splash screen after a delay and show the main window.
        
        Args:
            main_window: Main window to show after splash
            delay_ms: Delay in milliseconds before closing (default: 500ms)
        """
        QTimer.singleShot(delay_ms, lambda: self.finish(main_window))
