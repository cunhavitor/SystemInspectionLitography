import cv2
import numpy as np
import openvino.runtime as ov
import os
import time

class PatchCoreInferencer:
    def __init__(self, model_dir="models/bpo_rr125_patchcore_v2", device="CPU"):
        self.model_dir = model_dir
        self.device = device
        # Normalização ImageNet (Igual ao treino)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        # --- CONFIGURAÇÃO INDUSTRIAL ---
        self.threshold = 4.0        # Limiar de Rejeição (Ajustado pelos testes)
        self.noise_gate = 2.0       # Margem de Silêncio (Corta o ruído de fundo)
        
        # Load OpenVINO model
        print(f"📦 Loading model from {model_dir}...")
        core = ov.Core()
        model_xml = os.path.join(model_dir, "model.xml")
        
        if not os.path.exists(model_xml):
            raise FileNotFoundError(f"Model file not found: {model_xml}")

        # Configuração de Cache para arranque rápido nas próximas vezes
        core.set_property({'CACHE_DIR': os.path.join(model_dir, 'cache')})
        
        model = core.read_model(model=model_xml)
        self.compiled_model = core.compile_model(model=model, device_name=device)
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        
        # Load Bias Map (Threshold Map)
        # Load Bias Map (Threshold Map)
        bias_path = os.path.join(model_dir, "threshold_map.npy") # Nome atualizado
        if not os.path.exists(bias_path):
             # Fallback to standard name
             bias_path = os.path.join(model_dir, "threshold_map.npy")

        if os.path.exists(bias_path):
            self.bias_map = np.load(bias_path)
            # Forçar resize inicial para evitar custos na inferência
            if self.bias_map.shape != (448, 448):
                 self.bias_map = cv2.resize(self.bias_map, (448, 448))
            print(f"✅ Loaded bias map: {self.bias_map.shape}")
        else:
            print(f"⚠️ WARNING: Bias map not found at {bias_path}. Using flat threshold.")
            self.bias_map = np.full((448, 448), 15.0, dtype=np.float32)

        # Kernel de erosão pré-alocado (Otimização de memória)
        self.erode_kernel = np.ones((3, 3), np.uint8)

    def preprocess(self, image):
        """
        Resize -> Normalize -> Transpose (SEM CLAHE)
        """
        t0 = time.time()
        
        # 1. Resize (Se necessário)
        if image.shape[:2] != (448, 448):
            resized = cv2.resize(image, (448, 448))
        else:
            resized = image # Referência sem cópia se possível
        
        t1 = time.time()
        
        # 2. Convert to RGB & Normalize
        # Truque Numpy: fazer tudo numa operação vetorizada
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb_image /= 255.0
        rgb_image -= self.mean
        rgb_image /= self.std
        
        # 3. Transpose to [1, 3, 448, 448]
        input_tensor = np.transpose(rgb_image, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        t2 = time.time()
        
        timings = {
            'resize': (t1 - t0) * 1000,
            'norm': (t2 - t1) * 1000,
            'clahe': 0.0 # Kept for compatibility with inspection.py logging
        }
        
        return input_tensor, resized, timings

    def infer(self, image):
        """
        Pipeline: Preprocess -> OpenVINO -> Diff -> Erosão -> Score
        """
        start_time = time.time()
        
        # 1. Preprocess
        input_tensor, resized_image, timings = self.preprocess(image)
        
        # 2. Inference
        t_start_inf = time.time()
        results = self.compiled_model([input_tensor])[self.output_layer]
        timings['openvino'] = (time.time() - t_start_inf) * 1000
        
        # 3. Pós-Processamento (A Mágica da Limpeza)
        raw_map = results.squeeze() 
        
        # APLICAR O THRESHOLD MAP + NOISE GATE
        # Fórmula: Erro = Max(Raw - (Map + 2.0), 0)
        # Isto garante que o ruído de fundo (ex: 1.4) vira Zero.
        diff_map = np.maximum(raw_map - (self.bias_map + self.noise_gate), 0)
        
        # APLICAR EROSÃO (Remove pixéis isolados/poeira)
        # Se diff_map for tudo zero, saltamos isto para poupar tempo
        if np.max(diff_map) > 0:
            clean_map = cv2.erode(diff_map, self.erode_kernel, iterations=1)
        else:
            clean_map = diff_map

        # 4. Score Final (Pico Máximo)
        score = np.max(clean_map)
        
        timings['total_infer'] = (time.time() - start_time) * 1000
        
        return score, clean_map, resized_image, timings

    def visualize(self, clean_map, bg_image):
        """
        Sobrepõe o mapa de calor na imagem original.
        Otimizado para velocidade no Raspberry Pi 5.
        """
        # 1. NORMALIZAÇÃO FIXA (O Segredo da Estabilidade)
        # Em vez de usar o máximo da imagem atual (que faz o ruído brilhar),
        # usamos um teto fixo. 
        # - Valores < 4.0 (Threshold) ficarão azuis/cyanos.
        # - Valores > 10.0 (Defeito) ficarão vermelhos fortes.
        teto_maximo = 15.0 
        
        # Converte 0..15 para 0..255
        heatmap_norm = np.clip(clean_map / teto_maximo * 255, 0, 255).astype(np.uint8)
        
        # 2. COLORMAP (Azul = Frio, Vermelho = Quente)
        heatmap = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        
        # 3. REDIMENSIONAR (Segurança)
        # Garante que o mapa tem o mesmo tamanho da imagem (caso haja diferenças de arredondamento)
        if heatmap.shape[:2] != bg_image.shape[:2]:
            heatmap = cv2.resize(heatmap, (bg_image.shape[1], bg_image.shape[0]))
            
        # 4. BLENDING (Mistura)
        # 60% Imagem Original + 40% Mapa de Calor
        # Isto permite ver o rótulo da lata por baixo do "calor"
        overlay = cv2.addWeighted(bg_image, 0.6, heatmap, 0.4, 0)
        
        return overlay

if __name__ == "__main__":
    import sys
    
    # Caminho do modelo (ajusta se necessário)
    model_path = "models/bpo_rr125_patchcore_v2" 
    
    if len(sys.argv) < 2:
        print(f"Usage: python infer.py <image_path>")
        sys.exit(1)
        
    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print("Image not found")
        sys.exit(1)
        
    inferencer = PatchCoreInferencer(model_dir=model_path)
    
    # Warmup (Importante no Pi!)
    print("🔥 Warming up...", end="", flush=True)
    dummy = np.zeros((448,448,3), dtype=np.uint8)
    inferencer.infer(dummy)
    print(" Done.")

    # Inferência Real
    img = cv2.imread(img_path)
    score, amap, rez, times = inferencer.infer(img)
    
    print(f"\n⏱️  TIMINGS (ms):")
    print(f"Resize/Norm: {times['resize']+times['norm']:.2f}")
    print(f"OpenVINO:    {times['openvino']:.2f}")
    print(f"Total:       {times['total']:.2f}")
    
    print(f"\n📊 RESULTADO:")
    print(f"Score: {score:.4f}")
    status = "REJEITADO ❌" if score > inferencer.threshold else "APROVADO ✅"
    print(f"Status: {status}")
    
    # Salvar visualização
    viz = inferencer.visualize(amap, rez)
    cv2.imwrite("resultado_pi.jpg", viz)
    print("Visualização salva em 'resultado_pi.jpg'")