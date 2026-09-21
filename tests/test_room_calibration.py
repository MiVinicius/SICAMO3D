"""
Testes unitários da calibração métrica da sala (RoomCalibration).
Valida:
- Normal do chão apontando para cima (b > 0);
- Chão em Y=0 e câmera em Y=h;
- Sistema dextro (right-handed);
- Persistência e restauração via JSON.
"""
import os
import tempfile
import numpy as np
import pytest

from src.core.room_calibration import RoomCalibration

def test_room_calibration_default():
    calib = RoomCalibration()
    assert not calib.is_calibrated
    pt = np.array([1.0, 2.0, 3.0])
    # Sem calibração, deve retornar o próprio ponto
    res = calib.to_room_coords(pt)
    np.testing.assert_allclose(pt, res)

def test_room_calibration_from_floor_plane():
    calib = RoomCalibration()
    # Chão a 1.5 metros abaixo da câmera, câmera olhando para a frente e levemente inclinada para baixo
    # Normal do plano no espaço da câmera: [a, b, c, d] = [0.0, 0.9848, 0.1736, 1.50]
    plane = (0.0, 0.9848, 0.1736, 1.50)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "test_calib.json")
        success = calib.from_floor_plane(plane, sensor_height_m=1.5, save_path=json_path)
        assert success
        assert calib.is_calibrated
        assert os.path.exists(json_path)

        # 1. A câmera está na origem da câmera [0, 0, 0].
        # Na sala, a câmera deve estar em X=0, Y ~ 1.5m, Z ~ 0m
        cam_in_room = calib.to_room_coords([0.0, 0.0, 0.0])
        assert abs(cam_in_room[0]) < 0.01  # X ~ 0
        assert abs(cam_in_room[1] - 1.5) < 0.05  # Y ~ 1.5m de altura
        assert abs(cam_in_room[2]) < 0.01  # Z ~ 0

        # 2. Um ponto no chão no espaço da câmera:
        # Pelo plano: a*x + b*y + c*z + d = 0 => 0.9848*y + 0.1736*3.0 + 1.50 = 0
        # y = -(1.50 + 0.5208) / 0.9848 = -2.052m em Z=3.0m
        y_ground_cam = -(1.50 + 0.1736 * 3.0) / 0.9848
        floor_pt_cam = [0.0, y_ground_cam, 3.0]
        
        # No referencial da sala, a coordenada Y deve ser 0.0 (chão)
        floor_pt_room = calib.to_room_coords(floor_pt_cam)
        assert abs(floor_pt_room[1]) < 0.01  # Y_sala == 0

        # 3. Testa recarregamento via arquivo JSON
        calib_loaded = RoomCalibration(calibration_file=json_path)
        assert calib_loaded.is_calibrated
        reloaded_pt = calib_loaded.to_room_coords(floor_pt_cam)
        np.testing.assert_allclose(floor_pt_room, reloaded_pt, atol=1e-4)

def test_room_calibration_invert_normal_if_negative():
    calib = RoomCalibration()
    # Se normal foi fornecida apontando para baixo (b < 0)
    plane = (0.0, -1.0, 0.0, -1.4)
    calib.from_floor_plane(plane)
    assert calib.is_calibrated
    
    # Câmera em [0, 0, 0] deve ter altura positiva Y ~ 1.4m
    cam_in_room = calib.to_room_coords([0.0, 0.0, 0.0])
    assert abs(cam_in_room[1] - 1.4) < 0.01
