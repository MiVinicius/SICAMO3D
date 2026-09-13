"""
Módulo robusto de detecção de manipulação de brinquedos e atenção conjunta.
Utiliza critérios híbridos (Proximidade 3D, Sobreposição 2D, Vetor do Antebraço e Histerese de Posse).
"""
from typing import List, Dict, Tuple, Optional
import numpy as np

class ToyInteractionDetector:
    def __init__(self, hand_toy_threshold_m: float = 0.50):
        self.threshold_m = hand_toy_threshold_m
        # Histerese: limiar para manter o brinquedo após já ter sido pego (mais elástico)
        self.hold_threshold_m = hand_toy_threshold_m + 0.15 # 0.65m

    def analyze(self, active_tracks: List, detected_toys_3d: List[Dict]) -> Dict:
        """
        Calcula interações mão-brinquedo para cada pessoa e detecta atenção conjunta.
        """
        interactions = []
        toy_users: Dict[int, List[int]] = {}

        # Mapeia quais tracks continuam interagindo
        for toy_idx, toy in enumerate(detected_toys_3d):
            t_pos = np.array(toy['pos_3d'])
            toy_users[toy_idx] = []
            toy_bbox_2d = toy.get('bbox') # [x1, y1, x2, y2]

            for trk in active_tracks:
                left_3d, right_3d = trk.hands_3d
                left_2d, right_2d = trk.hands_2d

                # 1. Distâncias 3D puras entre o pulso/mão e o centro do brinquedo
                dist_left_3d = float(np.linalg.norm(np.array(left_3d) - t_pos)) if left_3d is not None else 999.0
                dist_right_3d = float(np.linalg.norm(np.array(right_3d) - t_pos)) if right_3d is not None else 999.0
                min_3d_dist = min(dist_left_3d, dist_right_3d)

                # 2. Critério de Sobreposição 2D no plano da câmera (caso o brinquedo cubra a mão)
                hand_overlap_2d = False
                which_hand = "indeterminado"
                
                if toy_bbox_2d is not None:
                    bx1, by1, bx2, by2 = toy_bbox_2d
                    # Expande a caixa do brinquedo em 40 pixels como margem de contato da mão
                    margin = 40.0
                    
                    if left_2d is not None:
                        lx, ly, _ = left_2d
                        if (bx1 - margin <= lx <= bx2 + margin) and (by1 - margin <= ly <= by2 + margin):
                            if left_3d is not None and abs(left_3d[2] - toy['pos_3d'][2]) < 0.45:
                                hand_overlap_2d = True
                                which_hand = "esquerda"

                    if right_2d is not None:
                        rx, ry, _ = right_2d
                        if (bx1 - margin <= rx <= bx2 + margin) and (by1 - margin <= ry <= by2 + margin):
                            if right_3d is not None and abs(right_3d[2] - toy['pos_3d'][2]) < 0.45:
                                hand_overlap_2d = True
                                which_hand = "direita"

                if which_hand == "indeterminado":
                    which_hand = "esquerda" if dist_left_3d <= dist_right_3d else "direita"

                # 3. Histerese de Posse: se a pessoa já estava segurando este brinquedo no frame anterior,
                # usamos um limiar mais generoso (hold_threshold_m) para não soltar por ruído temporário
                is_currently_holding = (trk.held_toy == toy['class_name'])
                active_threshold = self.hold_threshold_m if is_currently_holding else self.threshold_m

                # 4. Decisão de Interação:
                # - Detecção direta pelo Hand ROI da própria pessoa OU
                # - Proximidade 3D abaixo do limiar OU
                # - Mão visualmente tocando a caixa do brinquedo com profundidade similar
                is_hand_roi_match = (toy.get('source') == 'hand_roi' and toy.get('track_id') == trk.track_id)
                if is_hand_roi_match and toy.get('hand_label'):
                    which_hand = "esquerda" if "Esq" in toy['hand_label'] else "direita"
                
                is_interacting = is_hand_roi_match or (min_3d_dist <= active_threshold) or hand_overlap_2d

                if is_interacting:
                    toy_users[toy_idx].append(trk.track_id)
                    measured_dist = min_3d_dist if min_3d_dist < 999.0 else 0.15
                    interactions.append({
                        "track_id": trk.track_id,
                        "toy_name": toy['class_name'],
                        "toy_pos_3d": toy['pos_3d'],
                        "hand": which_hand,
                        "distance_m": round(measured_dist, 3)
                    })
                    trk.held_toy = toy['class_name']
                    trk.held_toy_timer = 20 # Mantém por ~20 frames (0.6s) de amortecimento

        # Decaimento suave do estado de posse para cada pessoa
        for trk in active_tracks:
            # Se não houve interação detectada neste quadro para este track
            if not any(it['track_id'] == trk.track_id for it in interactions):
                if hasattr(trk, 'held_toy_timer') and trk.held_toy_timer > 0:
                    trk.held_toy_timer -= 1
                else:
                    trk.held_toy = None

        # 5. Detecta atenção conjunta (2 ou mais pessoas interagindo com o mesmo brinquedo)
        joint_attentions = []
        for toy_idx, user_ids in toy_users.items():
            if len(user_ids) >= 2:
                joint_attentions.append({
                    "toy_name": detected_toys_3d[toy_idx]['class_name'],
                    "toy_pos_3d": detected_toys_3d[toy_idx]['pos_3d'],
                    "participant_ids": user_ids
                })

        return {
            "individual_interactions": interactions,
            "joint_attention_events": joint_attentions
        }
