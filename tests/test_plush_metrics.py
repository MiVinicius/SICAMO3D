"""
Testes unitários para o PlushMetricsAnalyzer e ScientificLogger Schema v3.
Valida cálculo de Gini, matriz de transição de posse, normalização de presença e log de sala vazia.
"""
import os
import shutil
import pytest
import numpy as np
import pandas as pd
from src.socioenative.plush_metrics import PlushMetricsAnalyzer
from src.socioenative.scientific_logger import ScientificLogger
from src.tracking.tracker_3d import Track3D

def test_gini_coefficient_calculation():
    analyzer = PlushMetricsAnalyzer()
    
    # 1. Distribuição perfeitamente igual (Gini = 0)
    equal_values = [10.0, 10.0, 10.0, 10.0]
    gini_equal = analyzer.compute_gini(equal_values)
    assert abs(gini_equal - 0.0) < 1e-4

    # 2. Distribuição totalmente concentrada (1 indivíduo com tudo)
    concentrated_values = [100.0, 0.0, 0.0, 0.0]
    gini_conc = analyzer.compute_gini(concentrated_values)
    assert gini_conc >= 0.70 # N=4 com 1 possuidor -> 0.75

def test_plush_metrics_accumulation_and_scene_summary():
    analyzer = PlushMetricsAnalyzer()
    analyzer.set_scene("Cena 1")

    trk_1 = Track3D(track_id=1, init_pos=(0.0, 1.0, 2.0))
    trk_2 = Track3D(track_id=2, init_pos=(1.0, 1.0, 2.0))

    # Simula 30 frames (1 segundo) onde o participante 1 tem a posse
    holder_info_1 = {
        "state": "COM_PORTADOR",
        "holder_id": 1,
        "confidence": 0.90,
        "events": []
    }
    for i in range(30):
        t = i * (1.0 / 30.0)
        analyzer.update([trk_1, trk_2], holder_info_1, scene_id="Cena 1", timestamp_s=t)

    # Simula evento de handoff de 1 para 2
    handoff_event = {
        "type": "handoff",
        "timestamp_ms": 1050,
        "donor_id": 1,
        "receiver_id": 2,
        "duration_s": 0.8,
        "distance_interpersonal_m": 0.75,
        "flags": "none"
    }
    holder_info_2 = {
        "state": "COM_PORTADOR",
        "holder_id": 2,
        "confidence": 0.90,
        "events": [handoff_event]
    }
    analyzer.update([trk_1, trk_2], holder_info_2, scene_id="Cena 1", timestamp_s=1.05)

    summary = analyzer.get_scene_summary("Cena 1")
    assert summary["scene_id"] == "Cena 1"
    assert summary["total_participants"] == 2
    assert summary["distinct_holders"] >= 1
    assert summary["handoff_count"] == 1
    assert summary["transition_matrix"] == {1: {2: 1}}

def test_scientific_logger_v3_empty_room_continuity(tmp_path):
    log_dir = str(tmp_path / "test_logs")
    logger = ScientificLogger(output_dir=log_dir, session_name="test_session")

    # Frame 1: sala vazia (sem tracks)
    empty_holder = {"state": "INDETERMINADO", "holder_id": None, "confidence": 0.0}
    logger.log_frame(t_capture_ms=1000, frame_idx=0, tracks=[], holder_info=empty_holder, scene_id="Cena 1")

    # Frame 2: 1 participante ativo
    trk = Track3D(track_id=1, init_pos=(0.0, 1.0, 2.0))
    trk.role = "participante"
    trk.presence_state = "plateia"
    logger.log_frame(t_capture_ms=1033, frame_idx=1, tracks=[trk], holder_info=empty_holder, scene_id="Cena 1")

    logger.close()

    # Verifica se os arquivos foram criados
    traj_csv = os.path.join(logger.session_dir, "trajectories.csv")
    plush_csv = os.path.join(logger.session_dir, "plush_state.csv")
    assert os.path.exists(traj_csv)
    assert os.path.exists(plush_csv)

    # Lê com pandas e valida a continuidade
    df_traj = pd.read_csv(traj_csv)
    assert len(df_traj) == 2
    # Linha 0 (sala vazia): track_id deve ser -1
    assert df_traj.iloc[0]["track_id"] == -1
    assert df_traj.iloc[0]["posture"] == "vazio"
    # Linha 1: track_id deve ser 1
    assert df_traj.iloc[1]["track_id"] == 1
    assert df_traj.iloc[1]["role"] == "participante"

def test_person_heading_and_attention_angular_diff():
    """
    Issue M0-03: Valida cálculo de heading corporal e atenção angular da plateia:
    - Ombros voltados para o portador -> angular_diff ~ 0° (atenção capturada)
    - De costas para o portador -> angular_diff ~ 180° (sem atenção)
    - Voltado para a direita (+X) -> heading ~ +90°
    - Voltado para a esquerda (-X) -> heading ~ -90°
    """
    analyzer = PlushMetricsAnalyzer()
    analyzer.set_scene("Cena 1")

    # Portador em (0.0, 1.2, 4.0)
    holder_trk = Track3D(track_id=1, init_pos=(0.0, 1.2, 4.0))

    # Participante em (0.0, 1.2, 2.0)
    audience_trk = Track3D(track_id=2, init_pos=(0.0, 1.2, 2.0))
    k3d = np.zeros((17, 4), dtype=np.float32)

    # Caso 1: Ombros voltados para o portador (+Z, costas para a câmera)
    # Ombro esquerdo anatômico (idx 5) em -X, direito anatômico (idx 6) em +X
    k3d[5] = [-0.20, 1.4, 2.0, 0.9]
    k3d[6] = [+0.20, 1.4, 2.0, 0.9]
    audience_trk.last_keypoints_3d = k3d.copy()

    heading_facing_holder = analyzer._estimate_person_heading(audience_trk)
    assert heading_facing_holder is not None
    assert abs(heading_facing_holder - 0.0) < 5.0  # ~ 0° (+Z)

    # Executa update com participante voltado para o portador
    holder_info = {
        "state": "COM_PORTADOR",
        "holder_id": 1,
        "confidence": 0.95,
        "events": []
    }
    analyzer.update([holder_trk, audience_trk], holder_info=holder_info, scene_id="Cena 1", timestamp_s=0.1)
    samples = analyzer.scenes_data["Cena 1"]["attention_samples"]
    assert len(samples) > 0
    assert samples[-1]["ratio_35"] == 1.0  # 100% da plateia atenta dentro do cone de 35°

    # Caso 2: De costas para o portador (voltado para a câmera, -Z)
    # Ombro esquerdo em +X, direito em -X
    k3d[5] = [+0.20, 1.4, 2.0, 0.9]
    k3d[6] = [-0.20, 1.4, 2.0, 0.9]
    audience_trk.last_keypoints_3d = k3d.copy()

    heading_facing_away = analyzer._estimate_person_heading(audience_trk)
    assert heading_facing_away is not None
    assert abs(abs(heading_facing_away) - 180.0) < 5.0  # ~ 180° / -180° (-Z)

    analyzer.update([holder_trk, audience_trk], holder_info=holder_info, scene_id="Cena 1", timestamp_s=0.2)
    samples = analyzer.scenes_data["Cena 1"]["attention_samples"]
    assert samples[-1]["ratio_35"] == 0.0  # 0% atenta quando de costas

    # Caso 3: Voltado para a direita (+X): ombro esquerdo em +Z, direito em -Z
    k3d[5] = [0.0, 1.4, 2.20, 0.9]
    k3d[6] = [0.0, 1.4, 1.80, 0.9]
    audience_trk.last_keypoints_3d = k3d.copy()
    heading_right = analyzer._estimate_person_heading(audience_trk)
    assert heading_right is not None
    assert abs(heading_right - 90.0) < 5.0  # ~ +90° (+X)

    # Caso 4: Voltado para a esquerda (-X): ombro esquerdo em -Z, direito em +Z
    k3d[5] = [0.0, 1.4, 1.80, 0.9]
    k3d[6] = [0.0, 1.4, 2.20, 0.9]
    audience_trk.last_keypoints_3d = k3d.copy()
    heading_left = analyzer._estimate_person_heading(audience_trk)
    assert heading_left is not None
    assert abs(heading_left - (-90.0)) < 5.0  # ~ -90° (-X)

