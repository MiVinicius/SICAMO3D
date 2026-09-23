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
                 bbox: Optional[List[float]] = None,
                 timestamp_s: Optional[float] = None,
                 role: str = "participante"):
        self.track_id = track_id
        self.kf = KalmanFilter3D(init_pos)
        self.hits = 1
        self.time_since_update = 0
        self.is_confirmed = False # Exige pelo menos 3 detecções para confirmar e evitar ruído
        
        # Filtro temporal de articulações para eliminar trepidação e linhas piscando
        self.kpt_filter = KeypointFilter(alpha=config.tracking.keypoint_smoothing_alpha, max_missed_frames=4)
        
        # Histórico de trajetória no espaço métrico real 3D (com deadzone anti-jitter)
        cur_time = timestamp_s if timestamp_s is not None else time.time()
        self.first_seen_time = cur_time
        self.last_seen_time = cur_time
        self.history: deque = deque(maxlen=200)
        self.history.append((init_pos[0], init_pos[1], init_pos[2], cur_time))

        # Histórico 2D da linha de detecção na altura do tórax na imagem da câmera
        self.trail_history_2d: deque = deque(maxlen=200)
        self.feet_history_2d = self.trail_history_2d # Alias de compatibilidade
        
        # Aplica filtragem inicial
        f_2d, f_3d = self.kpt_filter.filter(keypoints_2d, keypoints_3d)
        self.last_keypoints_3d = f_3d # (17, 4) [X, Y, Z, conf] métrico em metros
        self.last_keypoints_2d = f_2d # (17, 3) [u, v, conf] em pixels na imagem
        self.last_bbox = bbox

        # Papéis e estados socioenativos
        self.posture: str = "desconhecido"
        self.role: str = role # 'participante', 'facilitador'
        self.presence_state: str = "passante" # 'passante', 'plateia', 'participante_ativo'
        self.track_state: str = "medido" # 'medido' no frame da observação, 'predito' se extrapolado
        self.held_toy: Optional[str] = None
        self.dwell_time_s: float = 0.0
        self.stationary_time_s: float = 0.0  # Tempo acumulado com velocidade baixa (< 0.40 m/s)

    def predict(self, dt: Optional[float] = None) -> Tuple[float, float, float]:
        pos = self.kf.predict(dt=dt)
        self.time_since_update += 1
        self.track_state = "predito"
        return (float(pos[0]), float(pos[1]), float(pos[2]))

    def update(self, 
                pos_3d: Tuple[float, float, float], 
                keypoints_3d: Optional[np.ndarray] = None,
                keypoints_2d: Optional[np.ndarray] = None,
                bbox: Optional[List[float]] = None,
                timestamp_s: Optional[float] = None):
        self.kf.update(pos_3d)
        self.hits += 1
        self.time_since_update = 0
        self.track_state = "medido"
        cur_time = timestamp_s if timestamp_s is not None else time.time()
        frame_dt = max(0.001, min(0.20, cur_time - self.last_seen_time)) if self.last_seen_time > 0 else (1.0 / 30.0)
        self.last_seen_time = cur_time
        self.dwell_time_s = max(0.0, self.last_seen_time - self.first_seen_time)

        # Regra de confirmação: exige 3 hits para tracks recém-criados
        if self.hits >= 3:
            self.is_confirmed = True

        # Issue M1-02: Distinção de trânsito vs permanência intencional pela velocidade
        # Transeuntes cruzando a área em passos normais (ground_speed >= 0.50 m/s) permanecem como 'passante'.
        # Espectadores que param diante da cena (ground_speed < 0.40 m/s) acumulam stationary_time_s.
        spd = self.ground_speed
        if spd < 0.40:
            self.stationary_time_s += frame_dt
        elif spd >= 0.50:
            self.stationary_time_s = max(0.0, self.stationary_time_s - frame_dt * 1.5)

        if getattr(self, "held_toy", None) is not None or getattr(self, "role", "participante") == "facilitador":
            self.presence_state = "participante_ativo"
        elif self.stationary_time_s >= 10.0:
            self.presence_state = "participante_ativo"
        elif self.stationary_time_s >= 2.5:
            self.presence_state = "plateia"
        else:
            self.presence_state = "passante"
        
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
                self.history[-1] = (
                    0.80 * last_p[0] + 0.20 * cur_pos[0],
                    0.80 * last_p[1] + 0.20 * cur_pos[1],
                    0.80 * last_p[2] + 0.20 * cur_pos[2],
                    self.last_seen_time
                )

        # Rastreamento 2D da linha de detecção na altura do tórax
        cx, cy = None, None
        if keypoints_2d is not None:
            sh_pts = []
            if keypoints_2d[5, 2] > 0.18: sh_pts.append(keypoints_2d[5, :2])
            if keypoints_2d[6, 2] > 0.18: sh_pts.append(keypoints_2d[6, :2])
            if sh_pts:
                sh_mean = np.mean(sh_pts, axis=0)
                hip_pts = []
                if keypoints_2d[11, 2] > 0.18: hip_pts.append(keypoints_2d[11, :2])
                if keypoints_2d[12, 2] > 0.18: hip_pts.append(keypoints_2d[12, :2])
                if hip_pts:
                    hip_mean = np.mean(hip_pts, axis=0)
                    chest = 0.70 * sh_mean + 0.30 * hip_mean
                else:
                    chest = sh_mean + np.array([0.0, 18.0])
                cx, cy = float(chest[0]), float(chest[1])
        if cx is None and bbox is not None:
            cx = float((bbox[0] + bbox[2]) / 2.0)
            cy = float(bbox[1] + 0.28 * (bbox[3] - bbox[1]))

        if cx is not None and cy is not None:
            if len(self.trail_history_2d) == 0:
                self.trail_history_2d.append((cx, cy))
            else:
                lcx, lcy = self.trail_history_2d[-1]
                if np.hypot(cx - lcx, cy - lcy) >= 5.0:
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
    def ground_speed(self) -> float:
        return self.kf.ground_speed

    @property
    def heading_deg(self) -> Optional[float]:
        """
        Estima a orientação corporal no plano X-Z da sala em graus [-180, 180].
        0° = voltado para o fundo (+Z)
        +90° = direita (+X)
        -90° = esquerda (-X)
        180°/-180° = frente/câmera (-Z)
        """
        if self.last_keypoints_3d is not None:
            k3d = self.last_keypoints_3d
            if k3d.shape[0] > 6 and k3d[5, 3] > 0.20 and k3d[6, 3] > 0.20:
                sx_l, sz_l = k3d[5, 0], k3d[5, 2]
                sx_r, sz_r = k3d[6, 0], k3d[6, 2]
                if not (np.isnan(sx_l) or np.isnan(sx_r) or np.isnan(sz_l) or np.isnan(sz_r)):
                    v_shoulders = np.array([sx_l - sx_r, sz_l - sz_r])
                    v_facing = np.array([v_shoulders[1], -v_shoulders[0]])
                    return float(np.degrees(np.arctan2(v_facing[0], v_facing[1])))

        vx, _, vz = self.velocity
        if np.hypot(vx, vz) > 0.20:
            return float(np.degrees(np.arctan2(vx, vz)))
        return None

    @property
    def hands_3d(self) -> Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]]]:
        """Retorna coordenadas 3D métricas da mão esquerda (idx 9) e mão direita (idx 10)."""
        if self.last_keypoints_3d is None:
            return None, None
        
        left_hand = None
        right_hand = None
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
        self._last_timestamp_s: Optional[float] = None

    def update(self, detections: List[Dict], timestamp_s: Optional[float] = None) -> List[Track3D]:
        """
        Associa detecções 3D aos tracks existentes usando o Algoritmo Húngaro (Munkres)
        com restrição de distância euclidiana, cálculo de dt dinâmico e supressão de fantasmas.
        """
        cur_t = timestamp_s if timestamp_s is not None else time.time()
        dt = (cur_t - self._last_timestamp_s) if (self._last_timestamp_s is not None) else (1.0 / 30.0)
        self._last_timestamp_s = cur_t

        # 1. Predição para todos os tracks com dt dinâmico
        for trk in self.tracks:
            trk.predict(dt=dt)

        if len(detections) == 0:
            # Limpa tracks que ficaram sem observação
            # Tracks não confirmados são descartados em apenas 5 frames para não virarem fantasmas
            self.tracks = [
                t for t in self.tracks 
                if (t.time_since_update <= (self.max_lost_frames if t.is_confirmed else 5))
            ]
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
                        detections[c].get('bbox'),
                        timestamp_s=cur_t
                    )
                    assigned_tracks.add(r)
                    assigned_dets.add(c)

            # Detecções não associadas viram novos candidatos a tracks
            for d_idx in range(num_dets):
                if d_idx not in assigned_dets:
                    role = detections[d_idx].get('role', 'participante')
                    new_track = Track3D(
                        self._next_id, 
                        detections[d_idx]['pos_3d'], 
                        detections[d_idx].get('keypoints_3d'),
                        detections[d_idx].get('keypoints_2d'),
                        detections[d_idx].get('bbox'),
                        timestamp_s=cur_t,
                        role=role
                    )
                    self._next_id += 1
                    self.tracks.append(new_track)
        else:
            for det in detections:
                role = det.get('role', 'participante')
                new_track = Track3D(
                    self._next_id, 
                    det['pos_3d'], 
                    det.get('keypoints_3d'),
                    det.get('keypoints_2d'),
                    det.get('bbox'),
                    timestamp_s=cur_t,
                    role=role
                )
                self._next_id += 1
                self.tracks.append(new_track)

        # 3. Limpeza de tracks perdidos (fantasmas não confirmados morrem rápido em 5 frames)
        self.tracks = [
            t for t in self.tracks 
            if (t.time_since_update <= (self.max_lost_frames if t.is_confirmed else 5))
        ]

        # Retorna apenas tracks confirmados para garantir dados limpos ao dashboard e métricas
        return [t for t in self.tracks if t.is_confirmed]

