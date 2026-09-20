"""
Módulo de captura e mapeamento de coordenadas 3D para o sensor Kinect v2 no Windows.
"""
import ctypes
import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict
from pykinect2 import PyKinectV2, PyKinectRuntime

class KinectSensor:
    def __init__(self, calibration: Optional["RoomCalibration"] = None):
        self._sources = (PyKinectV2.FrameSourceTypes_Color | 
                         PyKinectV2.FrameSourceTypes_Depth | 
                         PyKinectV2.FrameSourceTypes_Body)
        self._kinect = PyKinectRuntime.PyKinectRuntime(self._sources)
        self._mapper = self._kinect._sensor.CoordinateMapper
        
        self.color_width = 1920
        self.color_height = 1080
        self.depth_width = 512
        self.depth_height = 424

        self.last_color_bgr: Optional[np.ndarray] = None
        self.last_depth_mm: Optional[np.ndarray] = None
        self.floor_clip_plane: Optional[Tuple[float, float, float, float]] = None
        self.room_calibration: Optional["RoomCalibration"] = calibration

        # Parâmetros intrínsecos do modelo pinhole de profundidade do Kinect v2 pré-calculados
        self.fx = 365.7
        self.fy = 365.7
        self.cx = 256.0
        self.cy = 212.0
        self.scale_u_to_d = self.depth_width / float(self.color_width)
        self.scale_v_to_d = self.depth_height / float(self.color_height)

        # Buffer BGR pré-alocado (Zero-Allocation de ~6.2 MB por frame)
        self._bgr_buffer = np.zeros((self.color_height, self.color_width, 3), dtype=np.uint8)

    def update(self) -> bool:
        """
        Atualiza os frames mais recentes de cor e profundidade.
        Retorna True se um novo frame de cor válido estiver disponível para processamento.
        """
        has_new_color = False
        if self._kinect.has_new_color_frame():
            frame = self._kinect.get_last_color_frame()
            if frame is not None and len(frame) == self.color_width * self.color_height * 4:
                # Converte BGRA nativo para BGR diretamente no buffer pré-alocado (zero cópias intermediárias)
                bgra = np.asarray(frame, dtype=np.uint8).reshape((self.color_height, self.color_width, 4))
                cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR, dst=self._bgr_buffer)
                self.last_color_bgr = self._bgr_buffer
                has_new_color = True

        if self._kinect.has_new_depth_frame():
            frame = self._kinect.get_last_depth_frame()
            if frame is not None and len(frame) == self.depth_width * self.depth_height:
                self.last_depth_mm = np.asarray(frame, dtype=np.uint16).reshape((self.depth_height, self.depth_width))

        if self._kinect.has_new_body_frame():
            body_frame = self._kinect.get_last_body_frame()
            if body_frame is not None and hasattr(body_frame, 'floor_clip_plane') and body_frame.floor_clip_plane is not None:
                p = body_frame.floor_clip_plane
                self.floor_clip_plane = (p.x, p.y, p.z, p.w)

        # Retorna True apenas quando ambos os fluxos já foram inicializados e há um novo frame de cor
        if self.last_color_bgr is None or self.last_depth_mm is None:
            return False
        return has_new_color

    def set_room_calibration(self, calibration: "RoomCalibration"):
        self.room_calibration = calibration


    def get_auto_calibration(self) -> Optional["RoomCalibration"]:
        """Retorna calibração automática estimada a partir do plano do chão detectado."""
        if self.floor_clip_plane is not None:
            from src.core.room_calibration import RoomCalibration
            return RoomCalibration.from_floor_plane(self.floor_clip_plane)
        return None

    def get_3d_point_from_color(self, u: float, v: float, to_room: bool = False) -> Optional[Tuple[float, float, float]]:
        """
        Mapeia um pixel da imagem colorida (u: 0..1920, v: 0..1080) para a coordenada métrica 3D
        no espaço da câmera (ou da sala se to_room=True) em metros com alta performance.
        Retorna None se o ponto for inválido ou ocluso.
        """
        if self.last_depth_mm is None:
            return None

        try:
            du = int(np.clip(u * self.scale_u_to_d, 0, self.depth_width - 1))
            dv = int(np.clip(v * self.scale_v_to_d, 0, self.depth_height - 1))
            
            # Amostra em janela 5x5 ao redor do ponto para evitar ruído / reflexos
            d_patch = self.last_depth_mm[max(0, dv-2):min(self.depth_height, dv+3),
                                         max(0, du-2):min(self.depth_width, du+3)]
            valid_depths = d_patch[(d_patch > 400) & (d_patch < 8000)]
            if len(valid_depths) == 0:
                return None
            
            depth_m = float(np.median(valid_depths)) * 0.001

            x_m = (du - self.cx) * depth_m / self.fx
            y_m = (dv - self.cy) * depth_m / self.fy
            z_m = depth_m

            pt_cam = (float(x_m), float(y_m), float(z_m))
            if to_room and self.room_calibration is not None and self.room_calibration.is_calibrated:
                pt_room = self.room_calibration.camera_to_room(pt_cam)
                return (float(pt_room[0]), float(pt_room[1]), float(pt_room[2]))
            return pt_cam
        except Exception:
            return None

    def unproject_keypoints_3d(self, keypoints_2d: np.ndarray, to_room: bool = False) -> np.ndarray:
        """
        Recebe matriz (17, 3) onde cada linha é [u, v, conf].
        Retorna matriz (17, 4) onde cada linha é [X, Y, Z, conf] em metros reais.
        Articulações sem profundidade válida recebem NaN nas coordenadas (em vez de 0.0).
        """
        kpts_3d = np.full((17, 4), np.nan, dtype=np.float32)
        for i in range(17):
            u, v, conf = keypoints_2d[i]
            kpts_3d[i, 3] = conf
            if conf > 0.20:
                pt_3d = self.get_3d_point_from_color(u, v, to_room=to_room)
                if pt_3d is not None:
                    kpts_3d[i, 0] = pt_3d[0]
                    kpts_3d[i, 1] = pt_3d[1]
                    kpts_3d[i, 2] = pt_3d[2]
        return kpts_3d


    def close(self):
        """Libera o sensor e recursos COM."""
        if self._kinect is not None:
            self._kinect.close()
            self._kinect = None
