import openvino.runtime as ov
import cv2
import numpy as np
import time
import os

class GravityInspector:
    def __init__(self, model_dir="models/bpo_rr125_patchcore_v2", device="CPU"):
        """
        Inicializa o Motor de Inspeção.
        :param model_dir: Pasta onde estão os ficheiros model.xml e threshold_map.npy
        :param device: 'CPU' ou 'NPU' (se disponível no hardware)
        """
        self.core = ov.Core()
        
        # 1. Carregar Modelo (XML + BIN)
        model_path = os.path.join(model_dir, "model.xml")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Modelo não encontrado em: {model_path}")
            
        print(f"⚙️ A carregar modelo Gravity em {device}...")
        self.model = self.core.read_model(model_path)
        self.compiled_model = self.core.compile_model(self.model, device)
        self.infer_request = self.compiled_model.create_infer_request()
        
        # 2. Carregar Mapa de Tolerância (O Escudo)
        map_path = os.path.join(model_dir, "threshold_map.npy")
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"❌ Mapa de Threshold não encontrado: {map_path}")
        self.threshold_map = np.load(map_path)
        print("✅ Sistema Gravity Pronto.")

    def apply_clahe(self, image):
        """
        Applies CLAHE to the L channel of the LAB image.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def preprocess(self, image):
        """Prepara a imagem da câmara para o formato que a IA entende (ImageNet Norm)."""
        # 1. CLAHE (Otimização de contraste)
        img_clahe = self.apply_clahe(image)
        
        # 2. Resize para 448x448
        resized = cv2.resize(img_clahe, (448, 448))
        
        # Converter BGR (OpenCV) para RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalização Manual (Super Rápida com NumPy)
        # (pixel / 255 - mean) / std
        img_float = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        img_norm = (img_float - mean) / std
        
        # Transpor para formato NCHW (1, 3, 448, 448)
        input_tensor = img_norm.transpose(2, 0, 1)[None, :, :, :]
        return input_tensor, resized

    def inspect(self, frame):
        """
        Processa um frame e decide se é DEFEITO ou OK.
        """
        start_time = time.time()
        
        # A. Preparar
        input_tensor, display_img = self.preprocess(frame)
        
        # B. Inferência (O Modelo corre aqui)
        # O modelo devolve o heatmap bruto
        results = self.infer_request.infer([input_tensor])
        raw_heatmap = list(results.values())[0].squeeze() # Remove dimensões extra
        
        # C. Lógica de Decisão (Pós-Processamento)
        # Subtrair a tolerância calibrada
        excess_map = np.maximum(raw_heatmap - self.threshold_map, 0)
        
        # O Score é o pico máximo de anomalia restante
        score = np.max(excess_map)
        
        # Limiar de Decisão (Fixo em 10.0 baseado no treino)
        is_defect = score > 10.0
        
        process_time = (time.time() - start_time) * 1000 # ms
        
        return {
            "is_defect": is_defect,
            "score": float(score),
            "heatmap": excess_map, # Mapa do defeito isolado
            "display_img": display_img,
            "latency_ms": process_time
        }

# ==========================================
# EXEMPLO DE USO (Loop Principal do Agente)
# ==========================================
if __name__ == "__main__":
    # Apontar para a pasta descompactada
    engine = GravityInspector(model_dir="models/patchcore448_v2")
    
    # Simular câmara (ou usar cv2.VideoCapture(0))
    # Para teste, usa uma imagem local se tiveres
    cap = cv2.VideoCapture(0) # Tenta abrir webcam
    
    print("🎥 A iniciar inspeção em tempo real (Ctrl+C para sair)...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret: 
                print("A aguardar câmara..."); time.sleep(1); continue
                
            # --- O CÉREBRO ANALISA AQUI ---
            result = engine.inspect(frame)
            # ------------------------------
            
            # Visualização (Para Debug/UI)
            img = result["display_img"]
            heatmap = result["heatmap"]
            score = result["score"]
            
            # Criar Overlay colorido se houver defeito
            if result["is_defect"]:
                # Normalizar heatmap para 0-255 para colorir
                hm_norm = (heatmap / (heatmap.max() + 1e-6) * 255).astype(np.uint8)
                hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_INFERNO)
                
                # Misturar com original
                display = cv2.addWeighted(img, 0.7, hm_color, 0.3, 0)
                status_text = f"DEFEITO ({score:.1f})"
                color = (0, 0, 255) # Vermelho
            else:
                display = img
                status_text = f"OK ({score:.1f})"
                color = (0, 255, 0) # Verde
                
            # Escrever status
            cv2.putText(display, status_text, (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(display, f"{result['latency_ms']:.1f}ms", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            # Mostrar
            cv2.imshow("Gravity Inspector", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n🛑 A parar...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
