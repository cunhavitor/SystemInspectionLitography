import cv2
import numpy as np
import json
import os

class CanCropper:
    def __init__(self, pixels_per_mm=3.7):
        """
        Inicializa o cropper com medidas reais das latas.
        """
        self.pixels_per_mm = pixels_per_mm
        
        # Default Medidas em milímetros
        self.first_can_center_x_mm = 61.04
        self.first_can_center_y_mm = 70.05
        self.can_width_mm = 119.47
        self.can_height_mm = 156.0
        self.step_x_mm = 120.52
        self.step_y_mm = 132.16
        self.tolerance_box_mm = 10.0
        self.sheet_margin_mm = 30.0  # Margem adicionada na folha retificada
        self.margin_top_bottom_mm = 10.05
        self.margin_left_right_mm = 1.59
        
        self.rows = 6
        self.cols = 8

        # Calculate initial pixel values
        self.update_pixel_values()

    def update_pixel_values(self):
        """Recalculate pixel values based on current mm parameters"""
        # Converter para pixels
        # Adicionamos sheet_margin_mm às coordenadas iniciais porque a imagem shiftou (30mm, 30mm)
        self.first_can_center_x = int((self.first_can_center_x_mm + self.margin_left_right_mm + self.sheet_margin_mm) * self.pixels_per_mm)
        self.first_can_center_y = int((self.first_can_center_y_mm + self.margin_top_bottom_mm + self.sheet_margin_mm) * self.pixels_per_mm)
        self.can_width = int(self.can_width_mm * self.pixels_per_mm)
        self.can_height = int(self.can_height_mm * self.pixels_per_mm)
        self.step_x = int(self.step_x_mm * self.pixels_per_mm)
        self.step_y = int(self.step_y_mm * self.pixels_per_mm)
        self.tolerance_box = int(self.tolerance_box_mm * self.pixels_per_mm)

    def load_params(self, filepath='config/crop_params.json'):
        """Load parameters from JSON"""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    params = json.load(f)
                    self.first_can_center_x_mm = params.get('first_can_center_x_mm', self.first_can_center_x_mm)
                    self.first_can_center_y_mm = params.get('first_can_center_y_mm', self.first_can_center_y_mm)
                    self.step_x_mm = params.get('step_x_mm', self.step_x_mm)
                    self.step_y_mm = params.get('step_y_mm', self.step_y_mm)
                    self.tolerance_box_mm = params.get('tolerance_box_mm', self.tolerance_box_mm)
                    self.margin_left_right_mm = params.get('margin_left_right_mm', self.margin_left_right_mm)
                    self.margin_top_bottom_mm = params.get('margin_top_bottom_mm', self.margin_top_bottom_mm)
                    
                    self.update_pixel_values()
                print(f"Crop params loaded from {filepath}")
                return True
            except Exception as e:
                print(f"Error loading crop params: {e}")
                return False
        return False

    def save_params(self, filepath='config/crop_params.json'):
        """Save parameters to JSON"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        try:
            params = {
                'first_can_center_x_mm': self.first_can_center_x_mm,
                'first_can_center_y_mm': self.first_can_center_y_mm,
                'step_x_mm': self.step_x_mm,
                'step_y_mm': self.step_y_mm,
                'tolerance_box_mm': self.tolerance_box_mm,
                'margin_left_right_mm': self.margin_left_right_mm,
                'margin_top_bottom_mm': self.margin_top_bottom_mm
            }
            with open(filepath, 'w') as f:
                json.dump(params, f, indent=4)
            print(f"Crop params saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving crop params: {e}")
            return False
    


    def crop_cans(self, rectified_image):
        """
        Extrai todas as latas da imagem retificada usando padrão zigzag.
        Args:
            rectified_image: Imagem retificada do SheetRectifier
            
        Returns:
            Lista de dicionários: [{'id': int, 'image': np.array, 'bbox': (x1,y1,x2,y2)}]
        """
        cans = []
        
        for row in range(self.rows):
            # Linhas ímpares (0, 2, 4) vão da esquerda para direita
            # Linhas pares (1, 3, 5) vão da direita para esquerda com offset
            is_even_row = (row % 2 == 1)
            
            # Calcular centro Y para esta linha
            center_y = self.first_can_center_y + (row * self.step_y)
            
            for col in range(self.cols):
                if is_even_row:
                    # Linha par: começa com offset de step_x/2 e vai da direita para esquerda
                    center_x = self.first_can_center_x + (self.step_x // 2) + ((self.cols - 1 - col) * self.step_x)
                else:
                    # Linha ímpar: vai da esquerda para direita
                    center_x = self.first_can_center_x + (col * self.step_x)
                
                # Calcular coordenadas do retângulo de crop (tamanho completo da lata)
                x1 = center_x - (self.can_width // 2)
                y1 = center_y - (self.can_height // 2)
                x2 = x1 + self.can_width
                y2 = y1 + self.can_height
                
                # Aplicar tolerância (expandir área de crop)
                x1 -= self.tolerance_box
                y1 -= self.tolerance_box
                x2 += self.tolerance_box
                y2 += self.tolerance_box
                
                # Garantir que não saímos dos limites da imagem
                h, w = rectified_image.shape[:2]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                # Extrair a lata
                can_image = rectified_image[y1:y2, x1:x2]
                
                # Calcular ID (1-based)
                can_id = row * self.cols + col + 1
                
                cans.append({
                    'id': can_id,
                    'image': can_image,
                    'bbox': (x1, y1, x2, y2)
                })
        
        return cans
    
    def draw_grid_preview(self, rectified_image):
        """
        Desenha uma visualização do grid de latas sobre a imagem retificada.
        Útil para verificar se as posições estão corretas.
        
        Args:
            rectified_image: Imagem retificada
            
        Returns:
            Imagem com o grid desenhado
        """
        preview = rectified_image.copy()
        
        for row in range(self.rows):
            is_even_row = (row % 2 == 1)
            center_y = self.first_can_center_y + (row * self.step_y)
            
            for col in range(self.cols):
                if is_even_row:
                    center_x = self.first_can_center_x + (self.step_x // 2) + ((self.cols - 1 - col) * self.step_x)
                else:
                    center_x = self.first_can_center_x + (col * self.step_x)
                
                # Calcular retângulo da lata (base)
                x1 = center_x - (self.can_width // 2)
                y1 = center_y - (self.can_height // 2)
                x2 = x1 + self.can_width
                y2 = y1 + self.can_height
                
                # Calcular retângulo com tolerância (área de crop)
                x1_tol = x1 - self.tolerance_box
                y1_tol = y1 - self.tolerance_box
                x2_tol = x2 + self.tolerance_box
                y2_tol = y2 + self.tolerance_box
                
                # Desenhar retângulo e número
                can_number = row * self.cols + col + 1
                color = (0, 255, 0) if not is_even_row else (255, 0, 0)  # Verde para ímpar, vermelho para par
                
                # Retângulo base (tamanho da lata) - linha fina
                cv2.rectangle(preview, (x1, y1), (x2, y2), color, 1)
                
                # Retângulo com tolerância (área real de crop) - linha grossa
                pt1_tol = (max(0, x1_tol), max(0, y1_tol))
                pt2_tol = (min(preview.shape[1], x2_tol), min(preview.shape[0], y2_tol))
                cv2.rectangle(preview, pt1_tol, pt2_tol, color, 3)
                
                cv2.circle(preview, (center_x, center_y), 5, (0, 0, 255), -1)
                cv2.putText(preview, f"{can_number}", 
                           (x1 + 10, y1 + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, color, 2)
        
        return preview
