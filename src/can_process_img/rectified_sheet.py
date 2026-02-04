import cv2
import numpy as np

class SheetRectifier:
    def __init__(self, pixels_per_mm=3.0):
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
    def __init__(self, pixels_per_mm=3.7):
        """
        Inicializa o retificador com medidas reais da folha.
        """
        # Medidas reais em milímetros
        self.sheet_width_mm = 966.0
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
        """
        # Construir pontos de destino como PARALLELOGRAMO com MARGEM
        # O usuario confirmou que existe um skew fisico de 'sheet_skew_mm' entre TL e BL.
        # TL -> (margin, margin)
        # TR -> (margin + width, margin)
        # BR -> (margin + width + skew, margin + height)  <-- Skew Restored
        # BL -> (margin + skew, margin + height)          <-- Skew Restored
        
        dst = np.array([
            [self.margin_px, self.margin_px],                                          # TL
            [self.margin_px + self.width_px, self.margin_px],                          # TR
            [self.margin_px + self.width_px + self.skew_px, self.margin_px + self.height_px],  # BR
            [self.margin_px + self.skew_px, self.margin_px + self.height_px]           # BL
        ], dtype="float32")
        
        # Aplicar transformação de perspectiva
        M = cv2.getPerspectiveTransform(corners, dst)
        
        # Dimensões da imagem de saída (folha + margens em ambos os lados)
        # Largura = margem_esq + skew + largura + margem_dir
        # Altura = margem_top + altura + margem_bottom
        output_width = self.margin_px + self.skew_px + self.width_px + self.margin_px
        output_height = self.margin_px + self.height_px + self.margin_px
        
        warped = cv2.warpPerspective(image, M, (output_width, output_height))
        
        return warped

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

    def crop_cans(self, rectified_image, rows=6, cols=8):
        h, w = rectified_image.shape[:2]
        h_step = h // rows
        w_step = w // cols
        
        cans = []
        for r in range(rows):
            for c in range(cols):
                y1, y2 = r * h_step, (r + 1) * h_step
                x1, x2 = c * w_step, (c + 1) * w_step
                cans.append(rectified_image[y1:y2, x1:x2])
        return cans