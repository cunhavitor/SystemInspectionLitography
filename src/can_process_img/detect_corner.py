import cv2
import numpy as np
import json
import os

class CornerDetector:
    def __init__(self):
        # Default Parameters
        self.roi_size = 250 
        self.margin_top = 200
        self.margin_bottom = 200
        self.margin_left = 200
        self.margin_right = 200
        self.dist_TL_BL = 0 # X-Offset TL -> BL
        self.dist_TR_BR = 0 # X-Offset TR -> BR

    def load_params(self, filepath='config/corner_params.json'):
        """Load parameters from JSON"""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    params = json.load(f)
                    self.roi_size = params.get('roi_size', self.roi_size)
                    self.margin_top = params.get('margin_top', self.margin_top)
                    self.margin_bottom = params.get('margin_bottom', self.margin_bottom)
                    self.margin_left = params.get('margin_left', self.margin_left)
                    self.margin_right = params.get('margin_right', self.margin_right)
                    self.dist_TL_BL = params.get('dist_TL_BL', self.dist_TL_BL)
                    self.dist_TR_BR = params.get('dist_TR_BR', self.dist_TR_BR)
                print(f"Corner params loaded from {filepath}")
                return True
            except Exception as e:
                print(f"Error loading corner params: {e}")
                return False
        return False

    def save_params(self, filepath='config/corner_params.json'):
        """Save parameters to JSON"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        try:
            params = {
                'roi_size': self.roi_size,
                'margin_top': self.margin_top,
                'margin_bottom': self.margin_bottom,
                'margin_left': self.margin_left,
                'margin_right': self.margin_right,
                'dist_TL_BL': self.dist_TL_BL,
                'dist_TR_BR': self.dist_TR_BR
            }
            with open(filepath, 'w') as f:
                json.dump(params, f, indent=4)
            print(f"Corner params saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving corner params: {e}")
            return False

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Calculate centers
        # TL: (left, top)
        # TR: (w - right, top)
        
        # BL: (left + x_offset, h - bottom)
        # BR: (w - right + x_offset, h - bottom)
        
        bl_x = max(0, int(self.margin_left + self.dist_TL_BL))
        br_x = min(w, int(w - self.margin_right + self.dist_TR_BR))
        
        search_centers = {
            "TL": (self.margin_left, self.margin_top),
            "TR": (w - self.margin_right, self.margin_top),       
            "BR": (br_x, h - self.margin_bottom),
            "BL": (bl_x, h - self.margin_bottom)        
        }

        pts = {}
        for key in ["TL", "TR", "BR", "BL"]:
            center = search_centers[key]
            x_s = max(0, int(center[0] - self.roi_size // 2))
            y_s = max(0, int(center[1] - self.roi_size // 2))
            
            # Ensure ROI is within image bounds
            y_end = min(h, y_s + self.roi_size)
            x_end = min(w, x_s + self.roi_size)
            
            roi = gray[y_s : y_end, x_s : x_end]

            if roi.size == 0:
                pts[key] = [float(center[0]), float(center[1])]
                continue

            # Otsu para detetar o metal contra o fundo preto
            _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                if key == "TL": p = c[np.argmin(c[:, :, 0] + c[:, :, 1])][0]
                elif key == "TR": p = c[np.argmax(c[:, :, 0] - c[:, :, 1])][0]
                elif key == "BR": p = c[np.argmax(c[:, :, 0] + c[:, :, 1])][0]
                elif key == "BL": p = c[np.argmin(c[:, :, 0] - c[:, :, 1])][0]
                pts[key] = [float(p[0] + x_s), float(p[1] + y_s)]
            else:
                pts[key] = [float(center[0]), float(center[1])]

        return np.array([pts["TL"], pts["TR"], pts["BR"], pts["BL"]], dtype="float32")

    def draw_preview(self, image, corners):
        """Draw detected corners on the image"""
        preview = image.copy()
        for i, p in enumerate(corners):
            cv2.circle(preview, (int(p[0]), int(p[1])), 5, (0, 255, 0), -1)
            cv2.putText(preview, f"C{i}", (int(p[0])+10, int(p[1])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return preview
        
    def visualize_search_areas(self, image):
        """Draw rectangles representing the search ROIs"""
        preview = image.copy()
        h, w = image.shape[:2]
        
        bl_x = max(0, int(self.margin_left + self.dist_TL_BL))
        br_x = min(w, int(w - self.margin_right + self.dist_TR_BR))
        
        search_centers = {
            "TL": (self.margin_left, self.margin_top),
            "TR": (w - self.margin_right, self.margin_top),       
            "BR": (br_x, h - self.margin_bottom),
            "BL": (bl_x, h - self.margin_bottom)        
        }
        
        for key, center in search_centers.items():
            x_s = int(center[0] - self.roi_size // 2)
            y_s = int(center[1] - self.roi_size // 2)
            pt1 = (x_s, y_s)
            pt2 = (x_s + self.roi_size, y_s + self.roi_size)
            cv2.rectangle(preview, pt1, pt2, (0, 255, 255), 2) # Yellow boxes
            
        return preview