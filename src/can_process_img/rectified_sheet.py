import cv2
import numpy as np

class SheetRectifier:
    def __init__(self, pixels_per_mm=3.82):
        """
        Inicializa o retificador com medidas reais da folha.
        
        Medidas da folha:
        - Largura (top e bottom): 1026.54 mm
        - Altura (vertical): 819 mm
        - Deslocamento horizontal (skew): 60.54 mm (TL para BL em X)
        - Margem de segurança: 30mm em volta de toda a folha
        
        Args:
            pixels_per_mm: Resolução da imagem de saída (pixels por milímetro)
        """
    def __init__(self, pixels_per_mm=3.82):
        """
        Inicializa o retificador com medidas reais da folha.
        """
        # Medidas reais em milímetros
        self.sheet_width_mm = 1026.54
        self.sheet_height_mm = 819.0
        self.sheet_skew_mm = 60.54  # Offset horizontal de TL para BL
        self.margin_mm = 30.0       # Margem extra em toda a volta
        self.pixels_per_mm = pixels_per_mm
        
        self.update_pixel_values()
        
    def update_pixel_values(self):
        # Converter para pixels
        self.width_px = int(self.sheet_width_mm * self.pixels_per_mm)
        self.height_px = int(self.sheet_height_mm * self.pixels_per_mm)
        self.skew_px = int(self.sheet_skew_mm * self.pixels_per_mm)
        self.margin_px = int(self.margin_mm * self.pixels_per_mm)

    def get_warped(self, image, corners):
        """
        Aplica transformação de perspectiva para retificar a folha.
        Calcula pixels_per_mm dinamicamente com base na largura detetada (TL-TR).
        """
        # Calcular largura em pixels na imagem original (Distância Euclidiana entre TL e TR)
        # corners shape: (4, 2) -> TL, TR, BR, BL
        tl, tr, br, bl = corners[0], corners[1], corners[2], corners[3]
        
        # Calculate raw dimensions from source detection
        width_px_src_top = np.linalg.norm(tr - tl)
        width_px_src_bot = np.linalg.norm(br - bl)
        avg_width_px = (width_px_src_top + width_px_src_bot) / 2.0
        
        height_px_src_left = np.linalg.norm(bl - tl)
        height_px_src_right = np.linalg.norm(br - tr)
        avg_height_px = (height_px_src_left + height_px_src_right) / 2.0
        
        # Calculate scales independently
        px_per_mm_x = avg_width_px / self.sheet_width_mm
        
        # LOGIC CHANGE: Y-Scale Correction for Parallelogram Skew
        # The detected height (avg_height_px) corresponds to the SLANTED edge (hypotenuse).
        # We must divide by the slanted physical length, not vertical height.
        physical_slant_height_mm = np.sqrt(self.sheet_height_mm**2 + self.sheet_skew_mm**2)
        px_per_mm_y = avg_height_px / physical_slant_height_mm
        
        print(f"[Rectifier] Scale X (Width): {px_per_mm_x:.4f} px/mm")
        print(f"[Rectifier] Slant Height mm: {physical_slant_height_mm:.2f}")
        print(f"[Rectifier] Scale Y (Edge): {px_per_mm_y:.4f} px/mm")
        
        # LOGIC CHANGE: Implement Anisotropic Scaling to fix X vs Y drift
        # User confirmed Y steps are correct with Y-based scale, but X steps are wrong.
        # This implies the effective scale in X is different from Y.
        
        # Recalculate destination dimensions using SEPARATE scales
        width_px = int(self.sheet_width_mm * px_per_mm_x)
        height_px = int(self.sheet_height_mm * px_per_mm_y)
        
        # For parameters that are primarily horizontal (skew), use X scale
        skew_px = int(self.sheet_skew_mm * px_per_mm_x)
        
        # Default margin uses Y scale for safety (or average), but let's use Y as it's the "trusted" axis
        margin_px = int(self.margin_mm * px_per_mm_y)
        
        # Construir pontos de destino como PARALLELOGRAMO com MARGEM
        # TL -> (margin, margin)
        # TR -> (margin + width, margin)
        # BR -> (margin + width + skew, margin + height)
        # BL -> (margin + skew, margin + height)
        
        dst = np.array([
            [margin_px, margin_px],                                          # TL
            [margin_px + width_px, margin_px],                          # TR
            [margin_px + width_px + skew_px, margin_px + height_px],  # BR
            [margin_px + skew_px, margin_px + height_px]           # BL
        ], dtype="float32")
        
        # Aplicar transformação de perspectiva
        M = cv2.getPerspectiveTransform(corners, dst)
        
        # Dimensões da imagem de saída
        output_width = margin_px + skew_px + width_px + margin_px
        output_height = margin_px + height_px + margin_px
        
        warped = cv2.warpPerspective(image, M, (output_width, output_height))
        
        # Return tuple of scales for X and Y
        return warped, (px_per_mm_x, px_per_mm_y)

    def load_params(self, filepath='config/rect_params.json'):
        """Load parameters from JSON"""
        import json
        import os
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    params = json.load(f)
                    self.sheet_width_mm = params.get('sheet_width_mm', self.sheet_width_mm)
                    self.sheet_height_mm = params.get('sheet_height_mm', self.sheet_height_mm)
                    self.sheet_skew_mm = params.get('sheet_skew_mm', self.sheet_skew_mm)
                    self.margin_mm = params.get('margin_mm', self.margin_mm)
                    
                    self.update_pixel_values()
                print(f"Rect params loaded from {filepath}")
                return True
            except Exception as e:
                print(f"Error loading rect params: {e}")
                return False
        return False

    def save_params(self, filepath='config/rect_params.json'):
        """Save parameters to JSON"""
        import json
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        try:
            params = {
                'sheet_width_mm': self.sheet_width_mm,
                'sheet_height_mm': self.sheet_height_mm,
                'sheet_skew_mm': self.sheet_skew_mm,
                'margin_mm': self.margin_mm
            }
            with open(filepath, 'w') as f:
                json.dump(params, f, indent=4)
            print(f"Rect params saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving rect params: {e}")
            return False

    def rectify(self, image, corners):
        """
        Wrapper principal para retificação.
        Returns:
            warped: Imagem retificada
            pixels_per_mm: Escala calculada (pixels/mm)
        """
        if corners is None or len(corners) != 4:
            return None, None
            
        warped, pixels_per_mm = self.get_warped(image, corners)
        return warped, pixels_per_mm