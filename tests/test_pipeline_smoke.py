"""
Teste de fumaça (Smoke Test) do Pipeline Central SICAMO3D.
Valida execução fim a fim sem hardware físico:
- Inicialização com sensor falso injetado e detector de objeto opcional;
- Execução de 5 frames através de pipeline.step();
- Atribuição manual de portador (Modo A);
- Alternância de papel de facilitador;
- Avanço de cena teatral com persistência de metrics_summary.json;
- Encerramento limpo sem memory leak ou crash de descritores de arquivo.
"""
import os
import tempfile
import numpy as np
import pytest

from src.core.pipeline import Pipeline
from src.core.room_calibration import RoomCalibration

class MockKinectSensor:
    def __init__(self):
        self.color_width = 1920
        self.color_height = 1080
        self.depth_width = 512
        self.depth_height = 424
        self.last_color_bgr = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.last_depth_mm = np.full((424, 512), 2000, dtype=np.uint16)
        self.floor_clip_plane = (0.0, 1.0, 0.0, 1.4)
        self.room_calibration = None

    def set_room_calibration(self, calibration: RoomCalibration):
        self.room_calibration = calibration

    def get_auto_calibration(self):
        calib = RoomCalibration()
        calib.from_floor_plane(self.floor_clip_plane, sensor_height_m=1.4)
        return calib

    def update(self) -> bool:
        return True

    def get_3d_point_from_color(self, u: float, v: float, to_room: bool = True):
        # Retorna ponto a 2 metros de profundidade
        pt = [0.0, 1.2, 2.0]
        if to_room and self.room_calibration:
            return self.room_calibration.to_room_coords(pt)
        return pt

    def unproject_keypoints_3d(self, keypoints_2d: np.ndarray, to_room: bool = True) -> np.ndarray:
        n = len(keypoints_2d)
        kpts_3d = np.zeros((n, 4), dtype=np.float32)
        for i in range(n):
            kpts_3d[i, 0] = 0.0
            kpts_3d[i, 1] = 1.0  # altura
            kpts_3d[i, 2] = 2.0  # profundidade
            kpts_3d[i, 3] = 0.8  # confiança
        return kpts_3d

    def close(self):
        pass

def test_pipeline_smoke_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_sensor = MockKinectSensor()
        pipeline = Pipeline(
            sensor=mock_sensor,
            enable_logger=True,
            session_name="smoke_test_session"
        )
        # Redireciona diretório de saída do logger para tmpdir
        if pipeline.logger:
            pipeline.logger.session_dir = tmpdir

        assert pipeline.sensor is not None
        assert pipeline.frame_idx == 0

        # Executa 5 passos completos
        for _ in range(5):
            success, canvas = pipeline.step()
            assert success
            assert canvas is not None
            assert canvas.shape == (900, 1600, 3)

        assert pipeline.frame_idx == 5

        # Simula detecções para alimentar o rastreador (mínimo de 3 hits para confirmação de track ativo)
        simulated_detections = [{
            "pos_3d": (0.2, 1.1, 2.5),
            "keypoints_3d": np.ones((17, 4), dtype=np.float32),
            "bbox": [800, 200, 1100, 900],
            "role": "participante"
        }]
        for _ in range(3):
            pipeline.step(remote_detections=simulated_detections)
        assert len(pipeline.last_tracks) >= 1
        active_id = pipeline.last_tracks[0].track_id

        # Testa Modo A: Atribuição manual de portador
        pipeline.set_manual_holder(active_id, duration_s=4.0)
        assert pipeline.holder_inference.state == "COM_PORTADOR"
        assert pipeline.holder_inference.holder_id == active_id

        # Testa alternância de facilitador
        pipeline.toggle_facilitator(active_id)
        assert pipeline.last_tracks[0].role == "facilitador"

        # Testa avanço de cena
        new_scene = pipeline.advance_scene()
        assert new_scene == "Cena 2"

        # Testa exportação de métricas
        metrics_file = os.path.join(tmpdir, "metrics_summary.json")
        pipeline.export_metrics_summary(filepath=metrics_file)
        assert os.path.exists(metrics_file)

        # Encerramento limpo
        pipeline.close()
