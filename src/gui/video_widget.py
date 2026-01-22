from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QPainter, QBrush, QColor, QImage, QPixmap

class VideoWidget(QWidget):
    """
    A widget that renders a QPixmap while maintaining aspect ratio,
    rendering black bars (letterboxing) where necessary.
    Uses QPainter for better performance than repeatedly scaling pixmaps.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap = None
        self.setMinimumSize(320, 240)
        # Set a black background by default
        self.setAttribute(Qt.WA_OpaquePaintEvent)

    def set_frame(self, pixmap):
        self.pixmap = pixmap
        self.update() # Trigger paintEvent

    def paintEvent(self, event):
        painter = QPainter(self)
        
        # Fill background with black
        painter.fillRect(self.rect(), QBrush(QColor(0, 0, 0)))
        
        if self.pixmap and not self.pixmap.isNull():
            # Calculate aspect-ratio correct rect
            widget_ratio = self.width() / self.height()
            pixmap_ratio = self.pixmap.width() / self.pixmap.height()
            
            target_rect = QRect()
            
            if widget_ratio > pixmap_ratio:
                # Widget is wider than image -> fit to height
                new_width = int(self.height() * pixmap_ratio)
                offset_x = (self.width() - new_width) // 2
                target_rect.setRect(offset_x, 0, new_width, self.height())
            else:
                # Widget is taller than image -> fit to width
                new_height = int(self.width() / pixmap_ratio)
                offset_y = (self.height() - new_height) // 2
                target_rect.setRect(0, offset_y, self.width(), new_height)
                
            # Draw
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            painter.drawPixmap(target_rect, self.pixmap)
