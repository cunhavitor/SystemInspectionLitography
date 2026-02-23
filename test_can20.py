import cv2
import sys
import numpy as np

sys.path.append('src')

try:
    from inference.efficientad_inference import EfficientAdInferencer
    model_dir = "models/bpo_rr125_efficientAD_M"
    inferencer = EfficientAdInferencer(model_dir)
    print("Mask shape:", inferencer.can_mask.shape if inferencer.can_mask is not None else "None")
except Exception as e:
    print(f"Error loading inferencer: {e}")

img_path = "/home/cunhav/projects/InspectionVisionCamera/data/defects/2026/02/NOK_20260223_140807_can20_score3.35.png"
img = cv2.imread(img_path)

if img is None:
    print(f"Failed to load image at {img_path}")
else:
    print(f"Image loaded: {img.shape}")
    try:
        score, anomaly_map, resized_img, timings = inferencer.infer(img)
        print(f"Inference complete. Score: {score}")
        print(f"Anomaly map shape: {anomaly_map.shape}, min: {anomaly_map.min()}, max: {anomaly_map.max()}")
        
        # Save visualization to disk so we can conceptually check where the heat is
        heatmap = inferencer.visualize(anomaly_map, resized_img)
        cv2.imwrite("debug_can20_heatmap.png", heatmap)
        print("Saved debug_can20_heatmap.png")
        
        # Also print the location of the highest intensity pixels to understand where the anomaly is
        y_indices, x_indices = np.unravel_index(np.argsort(anomaly_map.flatten())[-10:], anomaly_map.shape)
        
        print("\nTop 10 hottest pixel locations (y, x) and intensities:")
        for y, x in zip(y_indices, x_indices):
            print(f"  ({y}, {x}): {anomaly_map[y, x]:.4f}")
            
        # Draw small circles around top 10 hottest pixels to see where they are
        for y, x in zip(y_indices, x_indices):
            cv2.circle(resized_img, (x, y), 5, (0, 0, 255), 2)
            
        cv2.imwrite("debug_can20_hotspots.png", resized_img)
        print("Saved debug_can20_hotspots.png with top 10 pixels circled.")

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()

