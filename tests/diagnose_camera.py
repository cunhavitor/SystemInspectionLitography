import cv2
import glob

def check_camera(index, backend_name, backend_id):
    print(f"Testing Camera Index: {index}, Backend: {backend_name}...")
    cap = cv2.VideoCapture(index, backend_id)
    if not cap.isOpened():
        print(f"  [FAILED] Could not open camera.")
        return False
    
    ret, frame = cap.read()
    if not ret:
        print(f"  [FAILED] Opened, but could not read frame.")
    else:
        print(f"  [SUCCESS] Frame captured. Shape: {frame.shape}")
        
    cap.release()
    return ret

def main():
    print("OpenCV Version:", cv2.__version__)
    print("Build Info:", cv2.getBuildInformation())
    
    # Check available video devices
    devices = glob.glob("/dev/video*")
    print(f"Detected Video Devices: {devices}")
    
    indices = [0, 1, -1]
    backends = [
        ("CAP_ANY", cv2.CAP_ANY),
        ("CAP_V4L2", cv2.CAP_V4L2),
    ]
    
    for index in indices:
        for b_name, b_id in backends:
            check_camera(index, b_name, b_id)

if __name__ == "__main__":
    main()
