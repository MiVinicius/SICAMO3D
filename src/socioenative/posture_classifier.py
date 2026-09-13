"""
Classificador postural baseado em keypoints anatômicos 3D reais.
Identifica se a pessoa está em pé, sentada no chão, agachada ou debruçada.
"""
import numpy as np
from typing import Optional

class PostureClassifier:
    @staticmethod
    def classify(keypoints_3d: Optional[np.ndarray]) -> str:
        """
        Recebe matriz (17, 4) com coordenadas [X, Y, Z, conf] métricas de cada articulação.
        Índices COCO:
          0: nariz
          5: ombro esq, 6: ombro dir
          11: quadril esq, 12: quadril dir
          13: joelho esq, 14: joelho dir
          15: tornozelo esq, 16: tornozelo dir
        """
        if keypoints_3d is None:
            return "indeterminado"

        # Verifica visibilidade dos pontos chave
        conf = keypoints_3d[:, 3]
        if conf[11] < 0.25 and conf[12] < 0.25:
            return "indeterminado"

        # Pontos médios verticais (no Kinect v2, o eixo Y é tipicamente a altura)
        # y_ombro
        y_ombros = []
        if conf[5] > 0.25: y_ombros.append(keypoints_3d[5, 1])
        if conf[6] > 0.25: y_ombros.append(keypoints_3d[6, 1])

        # y_quadril
        y_quadris = []
        if conf[11] > 0.25: y_quadris.append(keypoints_3d[11, 1])
        if conf[12] > 0.25: y_quadris.append(keypoints_3d[12, 1])

        # y_joelhos
        y_joelhos = []
        if conf[13] > 0.25: y_joelhos.append(keypoints_3d[13, 1])
        if conf[14] > 0.25: y_joelhos.append(keypoints_3d[14, 1])

        # y_tornozelos
        y_tornozelos = []
        if conf[15] > 0.25: y_tornozelos.append(keypoints_3d[15, 1])
        if conf[16] > 0.25: y_tornozelos.append(keypoints_3d[16, 1])

        if not y_quadris or not y_ombros:
            return "em_pe"

        my_ombro = np.mean(y_ombros)
        my_quadril = np.mean(y_quadris)
        altura_tronco = abs(my_quadril - my_ombro)

        # Se tivermos joelhos ou tornozelos
        if y_joelhos and y_tornozelos:
            my_joelho = np.mean(y_joelhos)
            my_tornozelo = np.mean(y_tornozelos)
            altura_perna = abs(my_tornozelo - my_quadril)

            # Relação tronco / perna
            # Se pernas estão recolhidas / quadril quase na altura dos tornozelos:
            dist_quadril_tornozelo = abs(my_quadril - my_tornozelo)
            
            if dist_quadril_tornozelo < 0.35: # Menos de 35 cm entre quadril e tornozelos
                return "sentado_chao"
            elif dist_quadril_tornozelo < 0.55 and abs(my_quadril - my_joelho) < 0.25:
                return "agachado"

        # Inclinação do tronco (debruçado sobre brinquedo ou mesa)
        # Diferença em profundidade (Z) ou lateral (X) entre ombro e quadril
        z_ombro = keypoints_3d[5:7, 2].mean()
        z_quadril = keypoints_3d[11:13, 2].mean()
        if abs(z_ombro - z_quadril) > 0.35 and altura_tronco < 0.35:
            return "debruçado"

        return "em_pe"
