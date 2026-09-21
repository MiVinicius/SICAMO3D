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
            dist_quadril_tornozelo = abs(my_quadril - my_tornozelo)
            
            # Razão normalizada em relação ao tronco (invariante à estatura infantil ou adulta)
            ref_tronco = max(0.20, altura_tronco)
            ratio_perna_tronco = dist_quadril_tornozelo / ref_tronco

            # Se quadril está muito baixo em relação ao chão ou pernas recolhidas
            if my_quadril < 0.35 or ratio_perna_tronco < 0.65:
                return "sentado_chao"
            elif ratio_perna_tronco < 1.05 and abs(my_quadril - my_joelho) < (0.50 * ref_tronco):
                return "agachado"

        # Inclinação do tronco (debruçado sobre a pelúcia ou mesa)
        z_ombro = np.nanmean(keypoints_3d[5:7, 2])
        z_quadril = np.nanmean(keypoints_3d[11:13, 2])
        if abs(z_ombro - z_quadril) > 0.30 and altura_tronco < 0.30:
            return "debruçado"

        return "em_pe"
