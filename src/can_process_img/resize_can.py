import cv2

class CanResizer:
    def __init__(self, size=(448, 448)):
        """
        :param size: Tuplo (largura, altura) final para a IA.
        """
        self.size = size

    def process(self, can_image):
        # Usamos INTER_AREA para redução de tamanho (melhor qualidade para remover pixels)
        # Se estivéssemos a aumentar, usaríamos INTER_CUBIC.
        resized = cv2.resize(can_image, self.size, interpolation=cv2.INTER_AREA)
        return resized