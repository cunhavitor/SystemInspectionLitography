#!/usr/bin/env python3
from picamera2 import Picamera2
import time

print("Testing picamera2...")
picam = Picamera2()

# Print camera controls
print("\nAvailable controls:")
controls = picam.camera_controls
for name, (min_val, max_val, default) in controls.items():
    print(f"  {name}: min={min_val}, max={max_val}, default={default}")

# Configure camera
config = picam.create_video_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam.configure(config)
picam.start()

# Wait for auto-exposure to settle
time.sleep(2)

# Capture metadata
metadata = picam.capture_metadata()
print("\nCurrent metadata:")
for key, value in metadata.items():
    print(f"  {key}: {value}")

# Capture frame
frame = picam.capture_array("main")
print(f"\nFrame shape: {frame.shape}")
print(f"Frame min: {frame.min()}, max: {frame.max()}, mean: {frame.mean()}")

picam.stop()
print("\nTest complete!")
