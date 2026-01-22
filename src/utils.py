import os
import cv2
from datetime import datetime

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_image(image, folder, prefix="img"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.jpg"
    path = os.path.join(folder, filename)
    cv2.imwrite(path, image)
    print(f"Saved: {path}")
    return path

def calculate_sharpness(image):
    """Calculate the sharpness of an image using the Laplacian variance method."""
    # Check if image is already grayscale
    if len(image.shape) == 2 or image.shape[2] == 1:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()
