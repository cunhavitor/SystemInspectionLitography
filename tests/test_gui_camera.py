import sys
import yaml
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
import cv2

sys.path.insert(0, '.')
from src.camera import Camera

def load_config():
    with open("config/settings.yaml", "r") as f:
        return yaml.safe_load(f)

class TestWindow(QMainWindow):
    def __init__(self, config):
        super().__init__()
        self.setWindowTitle("Camera Test")
        self.resize(800, 600)
        
        self.label = QLabel("Waiting for camera...")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCentralWidget(self.label)
        
        print("Initializing camera...")
        try:
            self.camera = Camera(
                camera_index=config['camera']['index'],
                width=config['camera']['width'],
                height=config['camera']['height'],
                fps=config['camera']['fps']
            )
            print("Camera initialized successfully")
            
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(33)
        except Exception as e:
            print(f"ERROR: Failed to initialize camera: {e}")
            self.label.setText(f"Camera Error: {e}")
    
    def update_frame(self):
        if self.camera:
            frame = self.camera.get_frame()
            if frame is not None:
                print(f"Got frame: {frame.shape}")
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                self.label.setPixmap(pixmap.scaled(
                    self.label.size(), 
                    Qt.AspectRatioMode.KeepAspectRatio
                ))
            else:
                print("WARNING: Got None frame from camera")
                
    def closeEvent(self, event):
        if hasattr(self, 'camera'):
            self.camera.release()
        event.accept()

if __name__ == "__main__":
    config = load_config()
    app = QApplication(sys.argv)
    window = TestWindow(config)
    window.show()
    sys.exit(app.exec())
