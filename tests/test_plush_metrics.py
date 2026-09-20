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
