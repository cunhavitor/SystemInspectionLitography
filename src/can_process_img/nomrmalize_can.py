import cv2
import numpy as np

def prepare_for_autoencoder(img, target_size=(448, 448)):
    """
    Realiza CLAHE (brilho), Rescale (0-1) e Standardization.
    Assume que a imagem já vem com fundo preto puro.
    """
    # 1. CLAHE (Melhoria de Contraste e Brilho)
    # Convertemos para LAB para mexer apenas na luminosidade (L)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Criamos uma máscara para o CLAHE não atuar no fundo preto
    mask = (l > 5).astype(np.uint8) * 255 
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # Aplicamos o CLAHE
    l_enhanced = clahe.apply(l)
    
    # Voltamos a forçar o fundo a preto puro usando a máscara
    l_final = cv2.bitwise_and(l_enhanced, l_enhanced, mask=mask)
    
    # Reconstrói a imagem BGR
    enhanced_img = cv2.merge((l_final, a, b))
    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_LAB2BGR)

    # 2. Rescale e Standardization foram removidos para salvar a imagem corretamente.
    # A normalização matemática (Z-score) deve ser feita no DataLoader durante o treino/inferência.
    # Se salvarmos floats normalizados como JPG, a imagem fica preta (valores < 0 ou pequenos).
    
    return enhanced_img

# Exemplo de uso no teu loop:
# can_ready = prepare_for_autoencoder(aligned_can_from_step_4)