"""
Fase 4: Segmentação Geométrica 3D por Nuvem de Pontos (Depth Point Cloud Clustering).
Isola fisicamente aglomerados métricos 3D em torno das mãos e no espaço,
fornecendo confirmação física de brinquedos e objetos 100% agnóstica a cores e iluminação.
"""
from typing import List, Dict, Tuple, Optional
import numpy as np

class DepthCluster3D:
    def __init__(self, 
                 roi_radius_m: float = 0.28,
                 min_points: int = 40,
                 fx: float = 365.7,
                 fy: float = 365.7,
                 cx: float = 256.0,
                 cy: float = 212.0,
                 depth_w: int = 512,
                 depth_h: int = 424):
        """
        roi_radius_m: Raio métrico da esfera de busca ao redor do pulso/mão (28 cm).
        min_points: Quantidade mínima de pontos para formar um cluster físico consistente.
        """
        self.roi_radius_m = roi_radius_m
        self.min_points = min_points
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.depth_w = depth_w
        self.depth_h = depth_h

    def analyze_hands(self, 
                      depth_mm: np.ndarray, 
                      tracks: List, 
                      detected_toys: List[Dict]) -> List[Dict]:
        """
        Analisa a vizinhança métrica 3D de cada mão em busca de aglomerados de profundidade
        físicos. Retorna lista de clusters detectados e complementa brinquedos existentes.
        """
        if depth_mm is None or len(tracks) == 0:
            return []

        clusters = []

        for trk in tracks:
            lh_3d, rh_3d = trk.hands_3d
            lh_2d, rh_2d = trk.hands_2d
            kpts_3d = trk.last_keypoints_3d # (17, 4) [X, Y, Z, conf]

            for hand_label, h3, h2, elbow_idx in [("Mão Dir", rh_3d, rh_2d, 8), 
                                                   ("Mão Esq", lh_3d, lh_2d, 7)]:
                if h3 is None or h3[2] < 0.45 or h3[2] > 6.0:
                    continue

                hx, hy, hz = h3
                
                # Projeção do ponto central da mão para o espaço de profundidade (512 x 424)
                du = int(np.clip(hx * self.fx / hz + self.cx, 0, self.depth_w - 1))
                dv = int(np.clip(hy * self.fy / hz + self.cy, 0, self.depth_h - 1))

                # Raio em pixels na profundidade correspondente a 28 cm no espaço real
                r_px = int(np.clip((self.roi_radius_m * self.fx) / hz, 15, 120))
                u1 = max(0, du - r_px)
                u2 = min(self.depth_w, du + r_px)
                v1 = max(0, dv - r_px)
                v2 = min(self.depth_h, dv + r_px)

                patch = depth_mm[v1:v2, u1:u2]
                if patch.size == 0:
                    continue

                # Segmentação de profundidade no entorno métrico (+/- 25 cm ao redor da mão)
                min_depth_mm = (hz - self.roi_radius_m) * 1000.0
                max_depth_mm = (hz + self.roi_radius_m) * 1000.0
                depth_mask = (patch >= min_depth_mm) & (patch <= max_depth_mm)

                pv, pu = np.where(depth_mask)
                if len(pv) < self.min_points:
                    continue

                # Desprojeção vetorizada rápida de pixels 2D -> coordenadas métricas 3D
                p_z = patch[depth_mask].astype(np.float32) * 0.001
                p_u = (pu + u1).astype(np.float32)
                p_v = (pv + v1).astype(np.float32)

                p_x = (p_u - self.cx) * p_z / self.fx
                p_y = (p_v - self.cy) * p_z / self.fy

                pts_3d = np.column_stack((p_x, p_y, p_z)) # (N, 3)

                # Subtração vetorial do antebraço: se o cotovelo estiver visível,
                # descartamos pontos na direção de trás do pulso
                if kpts_3d is not None and kpts_3d[elbow_idx, 3] > 0.20:
                    elbow_pt = kpts_3d[elbow_idx, :3]
                    arm_vec = np.array([hx, hy, hz]) - elbow_pt
                    arm_len = np.linalg.norm(arm_vec)
                    if arm_len > 0.05:
                        arm_dir = arm_vec / arm_len
                        # Vetor de cada ponto relativo ao pulso
                        rel_pts = pts_3d - np.array([hx, hy, hz])
                        proj_arm = np.dot(rel_pts, arm_dir)
                        # Descarta pontos que apontam para trás do pulso em mais de 5 cm
                        forward_mask = proj_arm > -0.05
                        if np.sum(forward_mask) >= self.min_points:
                            pts_3d = pts_3d[forward_mask]

                # Métricas físicas do aglomerado (diâmetro, extensão e centroide 3D)
                min_bounds = np.min(pts_3d, axis=0)
                max_bounds = np.max(pts_3d, axis=0)
                extent = max_bounds - min_bounds
                max_dim = float(np.max(extent))
                centroid_3d = tuple(float(v) for v in np.mean(pts_3d, axis=0))

                # Um brinquedo/objeto seguro na mão tipicamente tem diâmetro entre 6 cm e 38 cm
                is_physical_object = (0.06 <= max_dim <= 0.38) and (len(pts_3d) >= self.min_points)

                if is_physical_object:
                    # Verifica se já existe um brinquedo detectado pelo YOLO próximo a este ponto
                    matched_yolo = None
                    c_pos = np.array(centroid_3d)
                    for toy in detected_toys:
                        t_pos = np.array(toy['pos_3d'])
                        if np.linalg.norm(t_pos - c_pos) < 0.22:
                            matched_yolo = toy
                            break

                    cluster_info = {
                        "track_id": trk.track_id,
                        "hand_label": hand_label,
                        "centroid_3d": centroid_3d,
                        "extent_3d": tuple(float(v) for v in extent),
                        "diameter_m": round(max_dim, 3),
                        "num_points": int(len(pts_3d)),
                        "matched_yolo_class": matched_yolo['class_name'] if matched_yolo else None,
                        "is_holding_object": True
                    }
                    clusters.append(cluster_info)

                    # Se um brinquedo YOLO coincidir, ancora o centroide 3D preciso da geometria
                    if matched_yolo:
                        matched_yolo['pos_3d'] = centroid_3d
                        matched_yolo['physical_diameter_m'] = round(max_dim, 3)
                        matched_yolo['cluster_points'] = len(pts_3d)

        return clusters
