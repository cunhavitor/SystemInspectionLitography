import os
import cv2
from datetime import datetime
import numpy as np

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

def calculate_noise_level(image):
    """
    Estimate image noise level using difference from blurred version.
    High values indicate high high-frequency activity (noise or sharp edges).
    """
    # Check if image is already grayscale
    if len(image.shape) == 2 or image.shape[2] == 1:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # Apply slight blur to remove noise
    blurred = cv2.blur(gray, (5, 5))
    
    # Calculate absolute difference (noise = original - blurred)
    diff = cv2.absdiff(gray, blurred)
    
    # Mean difference represents average noise amplitude
    return np.mean(diff)
