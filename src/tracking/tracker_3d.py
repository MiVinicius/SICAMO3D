"""
Motor de Rastreamento 3D Multi-Alvo com filtro temporal de articulações e prevenção de sobreposição de IDs.
"""
import time
from collections import deque
from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.core.config import config
from src.tracking.kalman_filter_3d import KalmanFilter3D
from src.tracking.keypoint_filter import KeypointFilter

class Track3D:
    def __init__(self, 
                 track_id: int, 
                 init_pos: Tuple[float, float, float], 
                 keypoints_3d: Optional[np.ndarray] = None,
                 keypoints_2d: Optional[np.ndarray] = None,
                 bbox: Optional[List[float]] = None):
        self.track_id = track_id
        self.kf = KalmanFilter3D(init_pos)
        self.hits = 1
        self.time_since_update = 0
        self.is_confirmed = False # Exige pelo menos 2 detecções consistentes para confirmar
        
        # Filtro temporal de articulações para eliminar trepidação e linhas piscando
        self.kpt_filter = KeypointFilter(alpha=config.tracking.keypoint_smoothing_alpha, max_missed_frames=4)
        
        # Histórico de trajetória no espaço métrico real 3D (com deadzone anti-jitter)
        self.history: deque = deque(maxlen=200)
        self.history.append((init_pos[0], init_pos[1], init_pos[2], time.time()))

        # Histórico 2D da linha de detecção na altura do tórax na imagem da câmera
        self.trail_history_2d: deque = deque(maxlen=200)
        self.feet_history_2d = self.trail_history_2d # Alias de compatibilidade
        
        # Aplica filtragem inicial
        f_2d, f_3d = self.kpt_filter.filter(keypoints_2d, keypoints_3d)
        self.last_keypoints_3d = f_3d # (17, 4) [X, Y, Z, conf] métrico em metros
        self.last_keypoints_2d = f_2d # (17, 3) [u, v, conf] em pixels na imagem
        self.last_bbox = bbox
        self.posture: str = "desconhecido"
        self.held_toy: Optional[str] = None
        self.last_seen_time = time.time()

    def predict(self) -> Tuple[float, float, float]:
        pos = self.kf.predict()
        self.time_since_update += 1
        return (float(pos[0]), float(pos[1]), float(pos[2]))

    def update(self, 
               pos_3d: Tuple[float, float, float], 
               keypoints_3d: Optional[np.ndarray] = None,
               keypoints_2d: Optional[np.ndarray] = None,
               bbox: Optional[List[float]] = None):
        self.kf.update(pos_3d)
        self.hits += 1
        self.time_since_update = 0
        self.last_seen_time = time.time()
        if self.hits >= 2:
            self.is_confirmed = True
        
        cur_pos = self.kf.position
        # Deadzone de deslocamento (5 cm): elimina trepidação e o nó de linhas quando a pessoa está parada
        if len(self.history) == 0:
            self.history.append((cur_pos[0], cur_pos[1], cur_pos[2], self.last_seen_time))
        else:
            last_p = self.history[-1]
            dist_moved = float(np.hypot(cur_pos[0] - last_p[0], cur_pos[2] - last_p[2]))
            if dist_moved >= 0.05: # Movimento real >= 5 cm
                self.history.append((cur_pos[0], cur_pos[1], cur_pos[2], self.last_seen_time))
            else:
                # Amortece suavemente a ponta sem empilhar pontos estáticos
                self.history[-1] = (
                    0.80 * last_p[0] + 0.20 * cur_pos[0],
                    0.80 * last_p[1] + 0.20 * cur_pos[1],
                    0.80 * last_p[2] + 0.20 * cur_pos[2],
                    self.last_seen_time
                )

        # Rastreamento 2D da linha de detecção na altura exata do tórax / peito
        cx, cy = None, None
        if keypoints_2d is not None:
            sh_pts = []
            if keypoints_2d[5, 2] > 0.18: sh_pts.append(keypoints_2d[5, :2])
            if keypoints_2d[6, 2] > 0.18: sh_pts.append(keypoints_2d[6, :2])
            if sh_pts:
                sh_mean = np.mean(sh_pts, axis=0)
                # Se quadris estiverem visíveis, posiciona no tórax (30% da distância dos ombros aos quadris)
                hip_pts = []
                if keypoints_2d[11, 2] > 0.18: hip_pts.append(keypoints_2d[11, :2])
                if keypoints_2d[12, 2] > 0.18: hip_pts.append(keypoints_2d[12, :2])
                if hip_pts:
                    hip_mean = np.mean(hip_pts, axis=0)
                    chest = 0.70 * sh_mean + 0.30 * hip_mean
                else:
                    # Estimativa anatômica abaixo dos ombros
                    chest = sh_mean + np.array([0.0, 18.0])
                cx, cy = float(chest[0]), float(chest[1])
        if cx is None and bbox is not None:
            # Caixa delimitadora: tórax fica a ~28% do topo da caixa
            cx = float((bbox[0] + bbox[2]) / 2.0)
            cy = float(bbox[1] + 0.28 * (bbox[3] - bbox[1]))

        if cx is not None and cy is not None:
            if len(self.trail_history_2d) == 0:
                self.trail_history_2d.append((cx, cy))
            else:
                lcx, lcy = self.trail_history_2d[-1]
                if np.hypot(cx - lcx, cy - lcy) >= 5.0: # Movimento real >= 5 pixels
                    self.trail_history_2d.append((cx, cy))
                else:
                    self.trail_history_2d[-1] = (0.80 * lcx + 0.20 * cx, 0.80 * lcy + 0.20 * cy)

        # Suaviza as articulações contra jitter e interpola se houve perda de 1-2 quadros
        f_2d, f_3d = self.kpt_filter.filter(keypoints_2d, keypoints_3d)
        if f_2d is not None:
            self.last_keypoints_2d = f_2d
        if f_3d is not None:
            self.last_keypoints_3d = f_3d
        if bbox is not None:
            self.last_bbox = bbox

    @property
    def position(self) -> Tuple[float, float, float]:
        return self.kf.position

    @property
    def velocity(self) -> Tuple[float, float, float]:
        return self.kf.velocity

    @property
    def speed(self) -> float:
        return self.kf.speed

    @property
    def hands_3d(self) -> Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]]]:
        """Retorna coordenadas 3D métricas da mão esquerda (idx 9) e mão direita (idx 10)."""
        if self.last_keypoints_3d is None:
            return None, None
        
        left_hand = None
        right_hand = None
        # COCO Keypoints: 9 = left_wrist, 10 = right_wrist
        if self.last_keypoints_3d[9, 3] > 0.18 and self.last_keypoints_3d[9, 2] > 0.35:
            left_hand = (float(self.last_keypoints_3d[9, 0]), 
                         float(self.last_keypoints_3d[9, 1]), 
                         float(self.last_keypoints_3d[9, 2]))
        if self.last_keypoints_3d[10, 3] > 0.18 and self.last_keypoints_3d[10, 2] > 0.35:
            right_hand = (float(self.last_keypoints_3d[10, 0]), 
                          float(self.last_keypoints_3d[10, 1]), 
                          float(self.last_keypoints_3d[10, 2]))
        return left_hand, right_hand

    @property
    def hands_2d(self) -> Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]]]:
        """Retorna coordenadas 2D em pixels (u, v, conf) da mão esquerda e direita."""
        if self.last_keypoints_2d is None:
            return None, None
        left = (float(self.last_keypoints_2d[9, 0]), float(self.last_keypoints_2d[9, 1]), float(self.last_keypoints_2d[9, 2])) if self.last_keypoints_2d[9, 2] > 0.18 else None
        right = (float(self.last_keypoints_2d[10, 0]), float(self.last_keypoints_2d[10, 1]), float(self.last_keypoints_2d[10, 2])) if self.last_keypoints_2d[10, 2] > 0.18 else None
        return left, right


class Tracker3D:
    def __init__(self, max_distance_m: float = 0.85, max_lost_frames: int = 30):
        self.max_distance_m = max_distance_m
        self.max_lost_frames = max_lost_frames
        self.tracks: List[Track3D] = []
        self._next_id = 1

    def update(self, detections: List[Dict]) -> List[Track3D]:
        """
        Associa detecções 3D aos tracks existentes usando o Algoritmo Húngaro (Munkres)
        com restrição de distância euclidiana para evitar que alvos próximos troquem de ID.
        """
        # 1. Predição para todos os tracks ativos
        for trk in self.tracks:
            trk.predict()

        if len(detections) == 0:
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_lost_frames]
            return [t for t in self.tracks if t.is_confirmed]

        # 2. Constrói matriz de custo com distância euclidiana 3D
        num_tracks = len(self.tracks)
        num_dets = len(detections)

        if num_tracks > 0:
            cost_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)
            for t_idx, trk in enumerate(self.tracks):
                tx, ty, tz = trk.position
                for d_idx, det in enumerate(detections):
                    dx, dy, dz = det['pos_3d']
                    dist = np.sqrt((tx - dx)**2 + (ty - dy)**2 + (tz - dz)**2)
                    cost_matrix[t_idx, d_idx] = dist

            # Algoritmo Húngaro para atribuição global ótima
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            assigned_tracks = set()
            assigned_dets = set()

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] <= self.max_distance_m:
                    self.tracks[r].update(
                        detections[c]['pos_3d'], 
                        detections[c].get('keypoints_3d'),
                        detections[c].get('keypoints_2d'),
                        detections[c].get('bbox')
                    )
                    assigned_tracks.add(r)
                    assigned_dets.add(c)

            # Detecções não associadas viram novos candidatos a tracks
            for d_idx in range(num_dets):
                if d_idx not in assigned_dets:
                    new_track = Track3D(
                        self._next_id, 
                        detections[d_idx]['pos_3d'], 
                        detections[d_idx].get('keypoints_3d'),
                        detections[d_idx].get('keypoints_2d'),
                        detections[d_idx].get('bbox')
                    )
                    self._next_id += 1
                    self.tracks.append(new_track)
        else:
            for det in detections:
                new_track = Track3D(
                    self._next_id, 
                    det['pos_3d'], 
                    det.get('keypoints_3d'),
                    det.get('keypoints_2d'),
                    det.get('bbox')
                )
                self._next_id += 1
                self.tracks.append(new_track)

        # 3. Limpeza de tracks perdidos por muito tempo
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_lost_frames]

        return [t for t in self.tracks if t.is_confirmed]
