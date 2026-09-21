"""
Módulo de calibração espacial e transformação de coordenadas entre o referencial
da câmera (Kinect v2) e o referencial métrico da sala / palco.
Garante que o plano do chão fique em Y = 0 (ou normal vertical [0, 1, 0]),
permitindo cálculos de velocidade e distância sem contaminação pelo ângulo de inclinação do sensor.
"""
from typing import Optional, Tuple, Union, Dict, Any
import numpy as np
import os
import json

class RoomCalibration:
    def __init__(self, 
                 rotation_matrix: Optional[np.ndarray] = None,
                 translation_vector: Optional[np.ndarray] = None,
                 floor_plane: Optional[Tuple[float, float, float, float]] = None,
                 calibration_file: Optional[str] = None):
        """
        Matriz de transformação: P_room = R * P_camera + T
        P_room: [X (largura), Y (altura a partir do chão), Z (profundidade da sala)]
        """
        self.R: np.ndarray = rotation_matrix if rotation_matrix is not None else np.eye(3, dtype=np.float32)
        self.T: np.ndarray = translation_vector if translation_vector is not None else np.zeros((3,), dtype=np.float32)
        self.floor_plane = floor_plane
        self.is_calibrated = (rotation_matrix is not None and translation_vector is not None)
        if calibration_file and os.path.exists(calibration_file):
            self.load_from_file(calibration_file)

    def from_floor_plane(self_or_cls, 
                         floor_clip_plane: Tuple[float, float, float, float], 
                         sensor_height_m: Optional[float] = None,
                         origin_offset: Optional[np.ndarray] = None, 
                         save_path: Optional[str] = None) -> "RoomCalibration":
        """
        Constrói calibração a partir da equação do plano do chão:
        ax + by + cz + d = 0 no espaço da câmera (onde +Y é para cima, +Z para frente).
        Garante que o plano do chão fique exatamente em Y_room = 0 e que a altura cresça positivamente para cima.
        """
        a, b, c, d = floor_clip_plane
        norm = float(np.sqrt(a*a + b*b + c*c))
        if norm < 1e-6:
            if isinstance(self_or_cls, type):
                return self_or_cls()
            return self_or_cls

        # Garante que a normal aponte para CIMA no espaço da câmera (b > 0)
        sign = 1.0 if b >= 0 else -1.0
        n_cam = np.array([sign * a / norm, sign * b / norm, sign * c / norm], dtype=np.float32)
        d_norm = float(sign * d / norm)
        if sensor_height_m is not None:
            d_norm = float(sensor_height_m)

        # No referencial da sala:
        # Y_room aponta ao longo da normal do chão (para cima)
        y_axis = n_cam

        # Z_room = projeção de [0, 0, 1] ortogonal a Y_room (profundidade da sala)
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float32) if abs(y_axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        z_axis = ref - np.dot(ref, y_axis) * y_axis
        z_axis /= max(float(np.linalg.norm(z_axis)), 1e-6)

        # X_room = Y_room x Z_room (sistema dextro: +X aponta para a direita)
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= max(float(np.linalg.norm(x_axis)), 1e-6)

        # Matriz de rotação: linhas são os vetores unitários dos eixos da sala
        R = np.vstack([x_axis, y_axis, z_axis]).astype(np.float32)

        # Para qualquer ponto P no chão: n_cam . P + d_norm = 0 => n_cam . P = -d_norm
        # Com R, a coordenada Y_rot = n_cam . P = -d_norm.
        # Para que Y_room = 0 no chão: T_y = +d_norm (altura física da câmera em relação ao chão).
        T = np.array([0.0, d_norm, 0.0], dtype=np.float32)
        if origin_offset is not None:
            T += origin_offset

        if isinstance(self_or_cls, type):
            calib = self_or_cls(rotation_matrix=R, translation_vector=T, floor_plane=floor_clip_plane)
        else:
            self_or_cls.R = R
            self_or_cls.T = T
            self_or_cls.floor_plane = floor_clip_plane
            self_or_cls.is_calibrated = True
            calib = self_or_cls

        if save_path:
            calib.save_to_file(save_path)

        return calib

    def camera_to_room(self, points: Union[np.ndarray, Tuple[float, float, float], list]) -> np.ndarray:
        """
        Converte pontos 3D do espaço da câmera para o referencial da sala.
        Trata NaNs preservando valores inválidos (em vez de converter para 0,0,0).
        Suporta (3,), (N, 3) ou (N, 4) onde a última coluna é confiança.
        """
        pts = np.asarray(points, dtype=np.float32)
        orig_shape = pts.shape

        if pts.ndim == 1:
            if pts.shape[0] == 3:
                if np.isnan(pts).any():
                    return np.array([np.nan, np.nan, np.nan], dtype=np.float32)
                return np.dot(self.R, pts) + self.T
            elif pts.shape[0] >= 4:
                conf = pts[3]
                if np.isnan(pts[:3]).any() or pts[2] <= 0.1:
                    return np.array([np.nan, np.nan, np.nan, conf], dtype=np.float32)
                r_pt = np.dot(self.R, pts[:3]) + self.T
                return np.array([r_pt[0], r_pt[1], r_pt[2], conf], dtype=np.float32)

        elif pts.ndim == 2:
            n, cols = pts.shape
            out = np.full_like(pts, np.nan)
            valid_mask = ~np.isnan(pts[:, 0]) & ~np.isnan(pts[:, 1]) & ~np.isnan(pts[:, 2]) & (pts[:, 2] > 0.1)
            if np.any(valid_mask):
                out[valid_mask, :3] = (np.dot(pts[valid_mask, :3], self.R.T) + self.T).astype(np.float32)
            if cols >= 4:
                out[:, 3] = pts[:, 3]
            return out

        return pts

    to_room_coords = camera_to_room

    def room_to_camera(self, points: Union[np.ndarray, Tuple[float, float, float]]) -> np.ndarray:
        """Transformação inversa: da sala para a câmera."""
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim == 1 and pts.shape[0] >= 3:
            if np.isnan(pts[:3]).any():
                return pts
            cam_pt = np.dot(self.R.T, (pts[:3] - self.T))
            if pts.shape[0] >= 4:
                return np.array([cam_pt[0], cam_pt[1], cam_pt[2], pts[3]], dtype=np.float32)
            return cam_pt
        elif pts.ndim == 2:
            out = np.full_like(pts, np.nan)
            valid_mask = ~np.isnan(pts[:, 0]) & ~np.isnan(pts[:, 1]) & ~np.isnan(pts[:, 2])
            if np.any(valid_mask):
                out[valid_mask, :3] = np.dot(pts[valid_mask, :3] - self.T, self.R)
            if pts.shape[1] >= 4:
                out[:, 3] = pts[:, 3]
            return out
        return pts

    def save_to_file(self, filepath: str):
        """Salva parâmetros de calibração em JSON."""
        data = {
            "R": self.R.tolist(),
            "T": self.T.tolist(),
            "floor_plane": list(self.floor_plane) if self.floor_plane is not None else None,
            "is_calibrated": self.is_calibrated
        }
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_from_file(self_or_cls, filepath: str) -> "RoomCalibration":
        if not os.path.exists(filepath):
            if isinstance(self_or_cls, type):
                return self_or_cls()
            return self_or_cls
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        R = np.array(data["R"], dtype=np.float32)
        T = np.array(data["T"], dtype=np.float32)
        fp = tuple(data["floor_plane"]) if data.get("floor_plane") else None
        is_calib = data.get("is_calibrated", True)

        if isinstance(self_or_cls, type):
            calib = self_or_cls(rotation_matrix=R, translation_vector=T, floor_plane=fp)
            calib.is_calibrated = is_calib
            return calib
        else:
            self_or_cls.R = R
            self_or_cls.T = T
            self_or_cls.floor_plane = fp
            self_or_cls.is_calibrated = is_calib
            return self_or_cls
