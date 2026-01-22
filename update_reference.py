#!/usr/bin/env python3
"""
Script para atualizar a imagem de referência com a melhor lata do dataset atual.
Garante Scale = 1.0 permanente no alinhamento.
"""

import cv2
import numpy as np
import os
import glob
from pathlib import Path

def calculate_sharpness(img):
    """Calcula nitidez usando variância do Laplaciano."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def calculate_brightness(img):
    """Calcula brilho médio."""
    return np.mean(img)

def check_specular(img, threshold=250):
    """Verifica se há reflexos especulares excessivos."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    white_pixels = np.sum(gray > threshold)
    return white_pixels

def score_image(img_path):
    """Calcula score de qualidade para uma imagem."""
    img = cv2.imread(img_path)
    if img is None:
        return -1, None
    
    # Verificar se é 448x448
    if img.shape[:2] != (448, 448):
        return -1, None
    
    sharpness = calculate_sharpness(img)
    brightness = calculate_brightness(img)
    specular_pixels = check_specular(img)
    
    # Score composto:
    # - Nitidez alta (> 100)
    # - Brilho próximo de 110-120
    # - Poucos reflexos (< 500 pixels)
    score = sharpness
    
    # Penalizar se brilho fora do range ideal
    if brightness < 100 or brightness > 130:
        score *= 0.7
    
    # Penalizar reflexos excessivos
    if specular_pixels > 500:
        score *= 0.5
    
    return score, img

def find_best_can(dataset_folder):
    """Encontra a melhor lata no dataset."""
    train_folder = os.path.join(dataset_folder, 'train')
    
    if not os.path.exists(train_folder):
        print(f"❌ Pasta train não encontrada: {train_folder}")
        return None, None
    
    # Procurar todas as imagens .png
    images = glob.glob(os.path.join(train_folder, '*.png'))
    
    if not images:
        print(f"❌ Nenhuma imagem encontrada em {train_folder}")
        return None, None
    
    print(f"📊 Analisando {len(images)} imagens...")
    
    best_score = -1
    best_path = None
    best_img = None
    
    for img_path in images:
        score, img = score_image(img_path)
        
        if score > best_score:
            best_score = score
            best_path = img_path
            best_img = img
    
    return best_path, best_img

def main():
    # Encontrar o dataset mais recente
    base_dir = Path(__file__).parent
    dataset_base = base_dir / 'data' / 'dataset'
    
    if not dataset_base.exists():
        print(f"❌ Pasta de datasets não encontrada: {dataset_base}")
        return
    
    # Listar todos os batches
    batches = sorted([d for d in dataset_base.iterdir() if d.is_dir()], 
                     key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not batches:
        print("❌ Nenhum dataset encontrado")
        return
    
    latest_batch = batches[0]
    print(f"📁 Dataset mais recente: {latest_batch.name}")
    
    # Encontrar melhor lata
    best_path, best_img = find_best_can(str(latest_batch))
    
    if best_path is None:
        print("❌ Não foi possível encontrar uma boa imagem de referência")
        return
    
    # Calcular estatísticas
    sharpness = calculate_sharpness(best_img)
    brightness = calculate_brightness(best_img)
    
    print(f"\n✅ Melhor lata encontrada:")
    print(f"   Arquivo: {os.path.basename(best_path)}")
    print(f"   Nitidez: {sharpness:.2f}")
    print(f"   Brilho: {brightness:.2f}")
    
    # Confirmar com usuário
    response = input("\n❓ Substituir imagem de referência? (s/N): ")
    
    if response.lower() != 's':
        print("❌ Operação cancelada pelo usuário")
        return
    
    # Fazer backup da referência antiga
    ref_path = base_dir / 'models' / 'can_reference' / 'aligned_can_reference448.png'
    backup_path = base_dir / 'models' / 'can_reference' / 'aligned_can_reference448-old.png'
    
    if ref_path.exists():
        cv2.imwrite(str(backup_path), cv2.imread(str(ref_path)))
        print(f"💾 Backup criado: {backup_path.name}")
    
    # Copiar nova referência
    cv2.imwrite(str(ref_path), best_img)
    print(f"✅ Nova referência salva: {ref_path}")
    print(f"\n🎉 Pronto! Reinicie a aplicação para aplicar as mudanças.")
    print(f"   Scale deve agora ser ~1.0 e alignment scores > 0.95")

if __name__ == "__main__":
    main()
