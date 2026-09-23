"""
Testes de integração para as ferramentas científicas evaluate.py e replay.py.
Valida estatísticas de sessão, casamento 1:1 estrito de handoffs (M4-03) e cálculo de ID switches (M4-02).
"""
import os
import pytest
import pandas as pd
from tools.evaluate import evaluate_session, calculate_id_switches
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

def test_handoff_strict_1to1_matching_multiple_close_events(tmp_path):
    """
    Testa múltiplos handoffs ocorrendo próximos no tempo com diferentes atores:
    GT 1: 1 -> 2 aos 2000ms
    GT 2: 2 -> 3 aos 2800ms
    """
    session_dir = tmp_path / "multi_handoff_session"
    session_dir.mkdir()

    # Cria plush_state.csv mínimo
    df_plush = pd.DataFrame({
        "t_capture_ms": [1000, 2000, 2800, 3500],
        "holder_id": [1, 2, 3, 3]
    })
    df_plush.to_csv(session_dir / "plush_state.csv", index=False)

    # Detecções do sistema:
    # Det 1: 1 -> 2 aos 2040ms (corresponde a GT 1)
    # Det 2: 2 -> 3 aos 2820ms (corresponde a GT 2)
    # Det 3: 4 -> 5 aos 3100ms (falso positivo extra)
    df_events = pd.DataFrame([
        {"t_capture_ms": 2040, "event_type": "handoff", "donor_id": 1, "receiver_id": 2},
        {"t_capture_ms": 2820, "event_type": "handoff", "donor_id": 2, "receiver_id": 3},
        {"t_capture_ms": 3100, "event_type": "handoff", "donor_id": 4, "receiver_id": 5}
    ])
    df_events.to_csv(session_dir / "events.csv", index=False)

    # Gabarito
    gt_csv_path = str(tmp_path / "gt_multi.csv")
    df_gt = pd.DataFrame([
        {"t_capture_ms": 2000, "event_type": "handoff", "donor_id": 1, "receiver_id": 2},
        {"t_capture_ms": 2800, "event_type": "handoff", "donor_id": 2, "receiver_id": 3}
    ])
    df_gt.to_csv(gt_csv_path, index=False)

    res = evaluate_session(str(session_dir), ground_truth_events_csv=gt_csv_path)

    assert res["gt_handoffs"] == 2
    assert res["n_handoffs_detected"] == 3
    assert res["tp"] == 2
    assert res["fp"] == 1 # O evento 4->5 é falso positivo
    assert res["fn"] == 0
    assert round(res["precision"], 2) == 0.67
    assert res["recall"] == 1.0

def test_id_switches_metric_computation(tmp_path):
    """
    Testa o cálculo da métrica de ID Switches (M4-02):
    Pessoa real 'P1' começa como track 1, sofre oclusão e reaparece como track 4 (1 switch).
    Pessoa real 'P2' permanece como track 2 durante toda a sessão (0 switches).
    """
    session_dir = tmp_path / "id_sw_session"
    session_dir.mkdir()

    # Cria trajectories.csv mínimo
    traj_data = []
    # Frame 0 a 19: P1 é track 1, P2 é track 2
    for f in range(20):
        t_ms = 1000 + f * 33
        traj_data.append({"t_capture_ms": t_ms, "frame_idx": f, "track_id": 1, "pos_x": 0.0, "pos_z": 2.0})
        traj_data.append({"t_capture_ms": t_ms, "frame_idx": f, "track_id": 2, "pos_x": 1.0, "pos_z": 2.0})

    # Frame 20 a 29: P1 ocluído (some do sistema)
    for f in range(20, 30):
        t_ms = 1000 + f * 33
        traj_data.append({"t_capture_ms": t_ms, "frame_idx": f, "track_id": 2, "pos_x": 1.0, "pos_z": 2.0})

    # Frame 30 a 59: P1 reaparece como track 4, P2 segue como track 2
    for f in range(30, 60):
        t_ms = 1000 + f * 33
        traj_data.append({"t_capture_ms": t_ms, "frame_idx": f, "track_id": 4, "pos_x": 0.1, "pos_z": 2.0})
        traj_data.append({"t_capture_ms": t_ms, "frame_idx": f, "track_id": 2, "pos_x": 1.0, "pos_z": 2.0})

    df_traj = pd.DataFrame(traj_data)
    df_traj.to_csv(session_dir / "trajectories.csv", index=False)

    # Cria plush_state.csv mínimo para evaluate_session não abortar
    pd.DataFrame({"t_capture_ms": [1000, 2000], "holder_id": [1, 2]}).to_csv(session_dir / "plush_state.csv", index=False)

    # Gabarito de identidades reais
    gt_identities = []
    for f in range(60):
        t_ms = 1000 + f * 33
        # No gabarito sabemos que P1 é a pessoa no frame
        if f < 20:
            gt_identities.append({"t_capture_ms": t_ms, "real_person_id": "P1", "track_id": 1})
        elif f >= 30:
            gt_identities.append({"t_capture_ms": t_ms, "real_person_id": "P1", "track_id": 4})
        gt_identities.append({"t_capture_ms": t_ms, "real_person_id": "P2", "track_id": 2})

    df_gt_id = pd.DataFrame(gt_identities)
    gt_id_path = str(tmp_path / "gt_identities.csv")
    df_gt_id.to_csv(gt_id_path, index=False)

    res = evaluate_session(str(session_dir), ground_truth_identities_csv=gt_id_path)

    assert "total_id_switches" in res
    assert res["total_id_switches"] == 1
    assert res["switches_by_person"]["P1"] == 1
    assert res["switches_by_person"]["P2"] == 0
    assert res["id_switches_per_min"] > 0
