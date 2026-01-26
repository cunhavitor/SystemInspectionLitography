import cv2
import numpy as np

class CanAligner:
    def __init__(self, reference_image_path, target_size=(448, 448)):
        self.target_size = target_size
        
        # 1. Carregar a imagem de referência
        self.ref_img = cv2.imread(reference_image_path)
        if self.ref_img is None:
            raise ValueError(f"Não foi possível carregar a referência em: {reference_image_path}")
        
        self.ref_h, self.ref_w = self.ref_img.shape[:2]
        self.ref_gray = cv2.cvtColor(self.ref_img, cv2.COLOR_BGR2GRAY)
        
        # Aplicar CLAHE para melhorar features do ORB
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.ref_gray_orb = clahe.apply(self.ref_gray)
        
        # Criar a máscara robusta baseada na imagem de referência
        # 1. Threshold simples para separar fundo preto
        _, thresh = cv2.threshold(self.ref_gray, 5, 255, cv2.THRESH_BINARY)
        
        # 2. Encontrar contorno externo (para garantir máscara sólida sem buracos internos)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.mask = np.zeros_like(self.ref_gray)
        if contours:
            # Assumir que a lata é o maior contorno
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(self.mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            
            # Opcional: Erode suave para remover bordas serrilhadas do recorte de referência
            kernel = np.ones((3,3), np.uint8)
            self.mask = cv2.erode(self.mask, kernel, iterations=1)
        else:
            # Fallback se não encontrar contornos: usar threshold simples
            self.mask = thresh
        
        # 2. Configurar o detetor ORB (OPTIMIZED for speed)
        self.orb = cv2.ORB_create(
            nfeatures=2000,       # Suficiente para o rótulo da lata
            scaleFactor=1.2,      # Ajuda a detetar se a lata estiver mais longe/perto
            nlevels=8,
            edgeThreshold=31,
            fastThreshold=20      # Baixamos para 20 para detetar pontos mais sutis no rótulo
        )
        # Usamos a máscara na referência para o ORB ignorar o fundo preto
        # Usar imagem com CLAHE para detetar keypoints
        self.kp_ref, self.des_ref = self.orb.detectAndCompute(self.ref_gray_orb, self.mask)
        
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def robust_crop(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        
        # Detetar o círculo principal (a borda da lata)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
            param1=50, param2=30, minRadius=150, maxRadius=400
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            x, y, r = circles[0, 0]
            
            # Criar um recorte quadrado centrado no círculo detetado
            size = int(r * 1.2) # Margem de segurança
            y1, y2 = max(0, y-size), min(img.shape[0], y+size)
            x1, x2 = max(0, x-size), min(img.shape[1], x+size)
            
            crop = img[y1:y2, x1:x2]
            return cv2.resize(crop, (self.ref_w, self.ref_h))
        return img

    def find_and_center_can(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Procura o contorno da lata (o maior objeto branco na máscara)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            
            # Cria um quadrado perfeito baseado na maior dimensão da lata
            size = max(w, h)
            center_x, center_y = x + w//2, y + h//2
            
            # Coordenadas do recorte sem limitar às bordas ainda
            x1 = center_x - size // 2
            y1 = center_y - size // 2
            x2 = x1 + size
            y2 = y1 + size
            
            # Criar canvas preto quadrado
            square_crop = np.zeros((size, size, 3), dtype=np.uint8)
            
            # Calcular interseção com a imagem original
            src_x1 = max(0, x1)
            src_y1 = max(0, y1)
            src_x2 = min(img.shape[1], x2)
            src_y2 = min(img.shape[0], y2)
            
            # Calcular onde colar no canvas (ROI de destino)
            dst_x1 = src_x1 - x1
            dst_y1 = src_y1 - y1
            dst_x2 = dst_x1 + (src_x2 - src_x1)
            dst_y2 = dst_y1 + (src_y2 - src_y1)
            
            # Copiar região válida se houver sobreposição
            if src_x2 > src_x1 and src_y2 > src_y1:
                square_crop[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]
            
            return cv2.resize(square_crop, self.target_size)
        return img

    def _remove_excess_margins(self, img):
        """Remove margens pretas excessivas do crop para coincidir escala da referência."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Procura o contorno da lata (o maior objeto branco na máscara)
        # Threshold aumentado par 50 para garantir crop SUPER tight na lata real
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        
        # Encontrar contorno da lata
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img
        
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        # Margem de correção de escala: 0% (Confiar no threshold mais tight)
        margin = 0
        
        # Coordenadas do crop com margem (garantir dentro da imagem)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img.shape[1], x + w + margin)
        y2 = min(img.shape[0], y + h + margin)
        
        # Crop e redimensionar mantendo aspect ratio quadrado
        cropped = img[y1:y2, x1:x2]
        size = max(cropped.shape[0], cropped.shape[1])
        
        # Criar canvas quadrado preto
        square = np.zeros((size, size, 3), dtype=np.uint8)
        
        # Centrar crop no canvas
        y_offset = (size - cropped.shape[0]) // 2
        x_offset = (size - cropped.shape[1]) // 2
        square[y_offset:y_offset+cropped.shape[0], x_offset:x_offset+cropped.shape[1]] = cropped
        
        # TRUQUE FINAL PARA SCALE 1.0 SEM CORTAR DADOS:
        # A referência tem a lata ~12% maior que o nosso crop "natural".
        # Em vez de cortar o input (margin negativa), fazemos "Pre-Zoom" (Resize)
        # 448 * 1.12 ~= 502.
        # Assim, a lata no input 502x502 terá o mesmo tamanho em pixels que na Ref 448x448.
        # O Affine Transform vai calcular Scale ~1.0 e o warp reduzirá para 448x448 automaticamente.
        return cv2.resize(square, (502, 502))

    def align(self, can_crop):
        if can_crop is None: return None
        
        # 0. Remover margens pretas excessivas para coincidir com escala da referência
        can_crop = self._remove_excess_margins(can_crop)
        
        gray_crop = cv2.cvtColor(can_crop, cv2.COLOR_BGR2GRAY)
        
        # Aplicar CLAHE no input também para consistência
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_crop_orb = clahe.apply(gray_crop)
        
        kp_crop, des_crop = self.orb.detectAndCompute(gray_crop_orb, None)
        
        if des_crop is None or len(des_crop) < 10:
            return self._apply_mask_and_resize(can_crop)

        # 3. Matching de pontos
        matches = self.bf.match(des_crop, self.des_ref)
        matches = sorted(matches, key=lambda x: x.distance)
        
        #print(f"[Align] Matches found: {len(matches)}")

        if len(matches) < 25: 
            #print(f"[Align] ⚠️ Low matches ({len(matches)} < 25). Fallback to geometric center.")
            # Em vez de tentar alinhar algo impossível, apenas centra a lata geometricamente
            return self._apply_mask_and_resize(can_crop)

        src_pts = np.float32([kp_crop[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([self.kp_ref[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # 4. Cálculo da Transformação Affine Parcial
        M, mask_ransac = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)

        if M is not None:
            # Extract scale, rotation, translation from M (2x3 matrix)
            # M = [[s*cos -s*sin tx], [s*sin s*cos ty]]
            scale_x = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
            angle_rad = np.arctan2(M[1, 0], M[0, 0])
            angle_deg = np.degrees(angle_rad)
            tx, ty = M[0, 2], M[1, 2]
            
            #print(f"[Align] Transform: Scale={scale_x:.3f}, Rot={angle_deg:.2f}°, Tx={tx:.1f}, Ty={ty:.1f}")
            
            # 5. Warp com borderValue em PRETO (0, 0, 0)
            aligned = cv2.warpAffine(
                can_crop, 
                M, 
                (self.ref_w, self.ref_h), 
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0) # Fundo agora é preto
            )
            
            # Apply mask to clean up alignment artifacts outside the can boundary
            return self._apply_mask_and_resize(aligned)
        
        print("[Align] ⚠️ Transformation estimation failed. Fallback to geometric center.")
        return self._apply_mask_and_resize(can_crop)

    def _apply_mask_and_resize(self, img):
        """Aplica a máscara para garantir fundo preto e redimensiona para target_size.
        CRÍTICO: Este método DEVE ser chamado em TODOS os caminhos de retorno.
        """
        # Resize to reference size first if needed
        if img.shape[:2] != (self.ref_h, self.ref_w):
            img = cv2.resize(img, (self.ref_w, self.ref_h))
        
        # Apply mask to ensure black background
        masked = cv2.bitwise_and(img, img, mask=self.mask)
        
        # Final resize to target size
        return cv2.resize(masked, self.target_size)