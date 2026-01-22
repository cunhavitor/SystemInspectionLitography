import cv2

def test_pipeline(pipeline):
    print(f"Testing Pipeline: {pipeline}")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("  [FAILED] Could not open pipeline.")
        return
    
    ret, frame = cap.read()
    if ret:
        print(f"  [SUCCESS] Frame captured. Shape: {frame.shape}")
    else:
        print("  [FAILED] Opened but no frame.")
    cap.release()

def main():
    # Common pipelines for Raspberry Pi
    pipelines = [
        "libcamerasrc ! video/x-raw, width=640, height=480, framerate=30/1 ! videoconvert ! appsink",
        "v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480, framerate=30/1 ! videoconvert ! appsink"
    ]
    
    for p in pipelines:
        test_pipeline(p)

if __name__ == "__main__":
    main()
