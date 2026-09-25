"""
Módulo da Fase 2: Hand-Centric High-Resolution ROI.
Executa recortes adaptativos métricos em alta definição (1080p) ao redor de cada mão
para detectar brinquedos pequenos com nitidez 10x superior e analisar o estado da pegada.
"""
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2

class HandROIDetector:
    def __init__(self, physical_roi_size_m: float = 0.55):
        """
        physical_roi_size_m: Tamanho físico do recorte ao redor da mão no espaço real (55 cm).
        Garante que, mesmo segurando o brinquedo na ponta dos dedos a 1,30m ou mais,
        o objeto permaneça 100% dentro do campo de visão do zoom.
        """
        self.roi_size_m = physical_roi_size_m
        self.fy_1080p = 1060.0
        self.hand_turn = 0
        self.cached_toys: Dict[Tuple[int, str], List[Dict]] = {}
        self.cached_crops: Dict[Tuple[int, str], Dict] = {}

    def process(self, 
                color_1080p: np.ndarray, 
                tracks: List, 
                object_engine, 
                sensor, 
                toy_classes: Dict[int, str],
                conf_thresh: float = 0.25,
                current_toys: Optional[List[Dict]] = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Processa com alta eficiência: prioriza a mão em interação com o brinquedo e avalia
        em alta definição (1080p) para reconhecimento estável e contínuo.
        """
        img_h, img_w = color_1080p.shape[:2]
        active_track_ids = {trk.track_id for trk in tracks}

        # Limpa cache de tracks que não estão mais presentes
        self.cached_toys = {k: v for k, v in self.cached_toys.items() if k[0] in active_track_ids}
        self.cached_crops = {k: v for k, v in self.cached_crops.items() if k[0] in active_track_ids}

        # Coleta todas as mãos válidas e visíveis
        candidate_hands = []
        for trk in tracks:
            left_2d, right_2d = trk.hands_2d
            left_3d, right_3d = trk.hands_3d

            for hand_name, h2d, h3d in [("Mão Dir", right_2d, right_3d), ("Mão Esq", left_2d, left_3d)]:
                if h2d is not None and h2d[2] > 0.10:
                    candidate_hands.append((trk, hand_name, h2d, h3d))

        if not candidate_hands:
            all_toys = [t for sub in self.cached_toys.values() for t in sub]
            return all_toys, list(self.cached_crops.values())

        # Prioriza inferência na mão mais próxima de um brinquedo conhecido
        best_candidate = None
        min_toy_dist = 999.0
        if current_toys:
            for cand in candidate_hands:
                c_trk, c_name, c_h2, c_h3 = cand
                for t in current_toys:
                    t_pos = np.array(t['pos_3d'])
                    d = float(np.linalg.norm(np.array(c_h3) - t_pos)) if c_h3 is not None else 999.0
                    if d < min_toy_dist:
                        min_toy_dist = d
                        best_candidate = cand

        if best_candidate is not None and min_toy_dist < 0.70:
            trk, hand_name, h2d, h3d = best_candidate
        else:
            sel_idx = self.hand_turn % len(candidate_hands)
            self.hand_turn += 1
            trk, hand_name, h2d, h3d = candidate_hands[sel_idx]

        key = (trk.track_id, hand_name)

        hx, hy, _ = h2d
        # Projeta o centro na direção da palma / dedos usando o vetor cotovelo -> pulso
        elbow_idx = 7 if "Esq" in hand_name else 8
        kpts_2d = trk.last_keypoints_2d
        if kpts_2d is not None and kpts_2d[elbow_idx, 2] > 0.12:
            ex, ey = kpts_2d[elbow_idx, 0], kpts_2d[elbow_idx, 1]
            vx, vy = hx - ex, hy - ey
            hx = hx + 0.25 * vx
            hy = hy + 0.25 * vy

        depth_m = h3d[2] if (h3d is not None and h3d[2] > 0.4) else 1.3

        # Tamanho adaptativo em pixels para cobrir exatamente 55cm no espaço real
        box_pixels = int(np.clip((self.roi_size_m * self.fy_1080p) / max(0.5, depth_m), 200, 520))
        half_box = box_pixels // 2

        x1 = max(0, int(hx - half_box))
        y1 = max(0, int(hy - half_box))
        x2 = min(img_w, int(hx + half_box))
        y2 = min(img_h, int(hy + half_box))

        crop_w = x2 - x1
        crop_h = y2 - y1

        if crop_w >= 80 and crop_h >= 80:
            raw_crop = color_1080p[y1:y2, x1:x2]

            # Inferência única no recorte da mão selecionada com limiar confiável (rejeita pele/mãos nuas)
            c_blob, c_ratio, c_pad = object_engine.preprocess(raw_crop)
            c_raw = object_engine.run_raw(c_blob)
            c_dets = object_engine.postprocess_objects(
                c_raw, c_ratio, c_pad, (crop_h, crop_w),
                toy_classes, conf_thresh=conf_thresh
            )

            crop_annotated = cv2.resize(raw_crop, (180, 180))
            current_hand_toys = []

            for cd in c_dets:
                if cd['confidence'] < conf_thresh:
                    continue

                cbx1, cby1, cbx2, cby2 = cd['bbox']
                # Se a caixa cobrir mais de 85% do recorte inteiro, é ruído da borda do crop
                if ((cbx2 - cbx1) * (cby2 - cby1)) / max(1, crop_w * crop_h) > 0.85:
                    continue
                global_bx1 = float(cbx1 + x1)
                global_by1 = float(cby1 + y1)
                global_bx2 = float(cbx2 + x1)
                global_by2 = float(cby2 + y1)

                cx_global = (global_bx1 + global_bx2) / 2.0
                cy_global = (global_by1 + global_by2) / 2.0
                pt_3d = sensor.get_3d_point_from_color(cx_global, cy_global)

                if pt_3d is None or pt_3d[2] < 0.35:
                    pt_3d = h3d if h3d is not None else (0.0, 0.0, depth_m)

                current_hand_toys.append({
                    "class_name": cd['class_name'],
                    "bbox": [global_bx1, global_by1, global_bx2, global_by2],
                    "confidence": cd['confidence'],
                    "pos_3d": pt_3d,
                    "source": "hand_roi",
                    "hand_label": hand_name,
                    "track_id": trk.track_id
                })

                scalec_x = 180.0 / crop_w
                scalec_y = 180.0 / crop_h
                p1 = (int(cbx1 * scalec_x), int(cby1 * scalec_y))
                p2 = (int(cbx2 * scalec_x), int(cby2 * scalec_y))
                cv2.rectangle(crop_annotated, p1, p2, (0, 255, 255), 2)
                cv2.putText(crop_annotated, cd['class_name'], (p1[0], max(12, p1[1] - 4)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)

            cv2.putText(crop_annotated, f"ID#{trk.track_id} {hand_name}", (5, 172), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 150), 1)

            # Atualiza caches da mão inspecionada
            self.cached_crops[key] = {
                "track_id": trk.track_id,
                "hand_name": hand_name,
                "crop_img": crop_annotated,
                "has_toy": len(c_dets) > 0
            }
            self.cached_toys[key] = current_hand_toys

        all_toys = [t for sub in self.cached_toys.values() for t in sub]
        return all_toys, list(self.cached_crops.values())
