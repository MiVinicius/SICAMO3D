"""
Testes de integração para as ferramentas científicas evaluate.py e replay.py.
"""
import os
import pytest
import pandas as pd
from tools.evaluate import evaluate_session
from src.socioenative.scientific_logger import ScientificLogger
from src.tracking.tracker_3d import Track3D

def test_evaluate_session_synthetic(tmp_path):
    session_dir = str(tmp_path / "test_eval_session")
    logger = ScientificLogger(output_dir=str(tmp_path), session_name="eval_session")
    
    trk1 = Track3D(1, (0.0, 1.0, 2.0))
    trk2 = Track3D(2, (1.0, 1.0, 2.0))

    # Simula 60 frames com posse do participante 1
    holder_info_1 = {
        "state": "COM_PORTADOR",
        "holder_id": 1,
        "confidence": 0.90,
        "plush_x": 0.1,
        "plush_z": 2.0,
        "events": []
    }
    for f in range(30):
        logger.log_frame(t_capture_ms=1000 + f * 33, frame_idx=f, tracks=[trk1, trk2], holder_info=holder_info_1, scene_id="Cena 1")

    # Handoff de 1 para 2
    handoff_ev = {
        "type": "handoff",
        "timestamp_ms": 2000,
        "donor_id": 1,
        "receiver_id": 2,
        "duration_s": 0.9,
        "distance_interpersonal_m": 0.8,
        "flags": "none"
    }
    holder_info_2 = {
        "state": "COM_PORTADOR",
        "holder_id": 2,
        "confidence": 0.90,
        "plush_x": 1.0,
        "plush_z": 2.0,
        "events": [handoff_ev]
    }
    for f in range(30, 60):
        events = [handoff_ev] if f == 30 else []
        holder_info_2["events"] = events
        logger.log_frame(t_capture_ms=1000 + f * 33, frame_idx=f, tracks=[trk1, trk2], holder_info=holder_info_2, scene_id="Cena 1")

    logger.close()

    # Cria CSV de gabarito para teste
    gt_csv_path = str(tmp_path / "gt_events.csv")
    df_gt = pd.DataFrame([{
        "t_capture_ms": 2010, # 10ms de diferença (bem dentro dos 1000ms de tolerância)
        "event_type": "handoff",
        "donor_id": 1,
        "receiver_id": 2
    }])
    df_gt.to_csv(gt_csv_path, index=False)

    # Executa evaluate_session
    res = evaluate_session(logger.session_dir, ground_truth_events_csv=gt_csv_path)
    
    assert res["distinct_holders"] == 2
    assert res["n_handoffs_detected"] == 1
    assert res["tp"] == 1
    assert res["fp"] == 0
    assert res["fn"] == 0
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f1_score"] == 1.0
