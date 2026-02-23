import cv2
import numpy as np

anomaly_map = np.ones((448, 448), dtype=np.float32)

noise_floor = 0.6
anomaly_map = np.where(anomaly_map < noise_floor, 0, anomaly_map - noise_floor)

print(anomaly_map.shape)
anomaly_map = cv2.GaussianBlur(anomaly_map, (21, 21), 0)
print(anomaly_map.shape)
