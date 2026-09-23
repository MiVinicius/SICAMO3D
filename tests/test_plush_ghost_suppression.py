"""
Testes unitários e sintéticos para supressão de pose duplicada na pelúcia (Issue M0-05).
Valida:
- IoU entre bounding boxes de pose e pelúcia;
- Rejeição de esqueleto falso quando tamanho do tronco < 0.25m;
- Preservação de pessoa real quando segurando a pelúcia (tronco > 0.25m);
- Supressão de track fantasma antes de entrar no Tracker3D.
"""
import numpy as np
import pytest
from src.core.pipeline import Pipeline

class SimpleMockSensor:
    def __init__(self):
        self.color_width = 1920
        self.color_height = 1080
        self.last_color_bgr = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.last_depth_mm = np.full((424, 512), 2000, dtype=np.uint16)
        self.room_calibration = None
    def set_room_calibration(self, calib):
        self.room_calibration = calib
    def update(self):
        return True
    def close(self):
        pass

def test_bbox_iou_computation():
    box1 = [100.0, 100.0, 300.0, 300.0]
    box2 = [100.0, 100.0, 300.0, 300.0]
    iou = Pipeline._bbox_iou(box1, box2)
    assert abs(iou - 1.0) < 1e-4

    # Sem sobreposição
    box3 = [400.0, 400.0, 500.0, 500.0]
    assert Pipeline._bbox_iou(box1, box3) == 0.0

    # Sobreposição parcial 50%
    box4 = [200.0, 100.0, 400.0, 300.0]
    iou_partial = Pipeline._bbox_iou(box1, box4)
    assert 0.30 < iou_partial < 0.40

def test_is_plush_ghost_pose_identification():
    mock_sensor = SimpleMockSensor()
    pipeline = Pipeline(sensor=mock_sensor, enable_logger=False)

    plush_candidates = [{
        "bbox": [500, 300, 650, 450],
        "confidence": 0.85,
        "class_name": "toy",
        "pos_3d": [0.0, 1.0, 2.0]
    }]

    # Caso 1: Pose sobreposta à pelúcia com tronco ínfimo (< 25 cm) -> fantasma
    ghost_pose = {
        "bbox": [490, 290, 660, 460], # IoU alto com a pelúcia
        "confidence": 0.50
    }
    kpts_3d_ghost = np.zeros((17, 4), dtype=np.float32)
    # Ombro esquerdo (5) e quadril esquerdo (11) a 15 cm de distância
    k3d_ghost_shoulder = np.array([0.0, 1.15, 2.0, 0.8], dtype=np.float32)
    k3d_ghost_hip = np.array([0.0, 1.00, 2.0, 0.8], dtype=np.float32)
    kpts_3d_ghost[5] = k3d_ghost_shoulder
    kpts_3d_ghost[11] = k3d_ghost_hip

    is_ghost = pipeline._is_plush_ghost_pose(ghost_pose, plush_candidates, kpts_3d_ghost)
    assert is_ghost is True

    # Caso 2: Criança real segurando a pelúcia (sobreposição de bbox, mas tronco > 25 cm, ex: 45 cm)
    real_person_pose = {
        "bbox": [480, 200, 670, 700], # IoU > 0.30 com a pelúcia
        "confidence": 0.85
    }
    kpts_3d_real = np.zeros((17, 4), dtype=np.float32)
    # Tronco de 45 cm
    kpts_3d_real[5] = np.array([0.0, 1.45, 2.0, 0.9], dtype=np.float32)
    kpts_3d_real[11] = np.array([0.0, 1.00, 2.0, 0.9], dtype=np.float32)

    is_ghost_real = pipeline._is_plush_ghost_pose(real_person_pose, plush_candidates, kpts_3d_real)
    assert is_ghost_real is False

    # Caso 3: Pessoa distante da pelúcia (sem sobreposição)
    distant_pose = {
        "bbox": [50, 50, 200, 400],
        "confidence": 0.90
    }
    is_ghost_distant = pipeline._is_plush_ghost_pose(distant_pose, plush_candidates, kpts_3d_real)
    assert is_ghost_distant is False
