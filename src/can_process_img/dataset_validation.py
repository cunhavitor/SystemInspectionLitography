import cv2
import numpy as np
import os

def check_specular_reflection(img, threshold=250, max_pixel_count=500):
    # Converte para cinzento se necessário
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Conta quantos pixéis estão quase em branco puro
    white_pixels = np.sum(gray > threshold)
    
    if white_pixels > max_pixel_count:
        return False, f"Reflexo excessivo: {white_pixels}px"
    return True, "Luz OK"
    
def is_image_good_for_dataset(img, ref_img, threshold_mse=0.8, min_sharpness=50, check_alignment=False):
    # 1. Verificar Nitidez (Laplacian)
    sharpness = cv2.Laplacian(img, cv2.CV_64F).var()
    if sharpness < min_sharpness:
        return False, f"Blurry ({sharpness:.1f} < {min_sharpness})"

    # 2. Verificar Brilho Médio (antes da normalização Z-score)
    mean_val = np.mean(img) / 255.0
    if mean_val < 0.05 or mean_val > 0.95:  # Very permissive range for dataset collection
        return False, f"Lighting ({mean_val:.2f})"

    # 3. Verificar Alinhamento (MSE face à referência)
    # NOTE: For dataset collection, alignment is usually already done in the pipeline
    # So we skip this check by default (check_alignment=False)
    alignment_score = None
    if check_alignment and ref_img is not None:
        # Compara a estrutura básica
        # Images with similar structure should pass even if not perfectly aligned
        try:
            # Ensure images are same size
            if img.shape != ref_img.shape:
                # Resize reference to match
                ref_resized = cv2.resize(ref_img, (img.shape[1], img.shape[0]))
            else:
                ref_resized = ref_img
            
            res = cv2.matchTemplate(img, ref_resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            alignment_score = max_val
            
            # Relaxed threshold for dataset collection: 50% instead of 85%
            # This allows more variation while still filtering out completely different images
            if max_val < 0.50:
                return False, f"Alignment ({max_val:.2f} < 0.50)"
        except Exception as e:
            # If alignment check fails, just warn but don't reject
            print(f"Alignment check failed: {e}, accepting image anyway")

    # 4. Verificar Reflexos Especulares
    is_specular_ok, spec_reason = check_specular_reflection(img)
    if not is_specular_ok:
        return False, spec_reason

    # Build success message with scores
    if alignment_score is not None:
        return True, f"OK (align: {alignment_score:.2f})"
    else:
        return True, "OK"

# Loop para filtrar o teu dataset de 960 latas
# Se for OK -> move para 'train_folder'
# Se for NOK -> move para 'debug_folder'