"""
Testes automatizados para o Milestone M2:
1. Parametrização e sensibilidade de pesos e limiares em HolderInference (M2-04).
2. Validação da ferramenta de comparação Modo A vs Modo B (tools/compare_modes.py) (M2-05).
"""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.tracking.tracker_3d import Track3D
from src.socioenative.holder_inference import HolderInference, HolderState
from tools.compare_modes import compare_sessions

def _make_track(track_id: int, pos: tuple, hand_right: tuple = None) -> Track3D:
    trk = Track3D(track_id=track_id, init_pos=pos, timestamp_s=0.0)
    trk.is_confirmed = True
    trk.dwell_time_s = 5.0
    k3d = np.full((17, 4), np.nan, dtype=np.float32)
    k3d[:, 3] = 0.0
    if hand_right:
        k3d[10, :3] = hand_right
        k3d[10, 3] = 0.95
    trk.last_keypoints_3d = k3d
    return trk

def test_holder_custom_thresholds_and_weights():
    # Caso 1: Prioriza punho fortemente (peso punho = 0.9, peso tronco = 0.0)
    inf_wrist = HolderInference(
        wrist_thresh_m=0.20,
        torso_thresh_m=0.80,
        weights=(0.90, 0.0, 0.05, 0.05)
    )
    # Pessoa A: punho muito próximo (0.05m), tronco a 0.50m
    trkA = _make_track(1, pos=(0.50, 1.0, 2.0), hand_right=(0.05, 1.0, 2.0))
    # Pessoa B: tronco muito próximo (0.05m), mas punho longe (0.50m)
    trkB = _make_track(2, pos=(0.05, 1.0, 2.0), hand_right=(0.50, 1.0, 2.0))

    plush = [{"pos_3d": (0.0, 1.0, 2.0), "confidence": 0.90}]
    res_wrist = inf_wrist.process([trkA, trkB], plush, timestamp_s=1.0)
    assert res_wrist["holder_id"] == 1, "Com peso no punho, Pessoa 1 (punho perto) deve ser o portador"

    # Caso 2: Prioriza tronco fortemente (peso punho = 0.0, peso tronco = 0.9)
    inf_torso = HolderInference(
        wrist_thresh_m=0.20,
        torso_thresh_m=0.80,
        weights=(0.0, 0.90, 0.05, 0.05)
    )
    res_torso = inf_torso.process([trkA, trkB], plush, timestamp_s=1.0)
    assert res_torso["holder_id"] == 2, "Com peso no tronco, Pessoa 2 (tronco perto) deve ser o portador"

def test_compare_modes_synthetic_session():
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_a = Path(tmpdir) / "session_mode_a"
        dir_b = Path(tmpdir) / "session_mode_b"
        dir_a.mkdir()
        dir_b.mkdir()

        # Cria dados sintéticos para Modo A (100 frames)
        times_ms = [1000 + i * 33 for i in range(100)]
        holders_a = [1] * 50 + [2] * 50
        states_a = ["COM_PORTADOR"] * 100

        df_plush_a = pd.DataFrame({
            "t_capture_ms": times_ms,
            "frame_idx": list(range(100)),
            "state": states_a,
            "holder_id": holders_a,
            "holder_conf": [1.0] * 100,
            "source": ["manual"] * 100
        })
        df_plush_a.to_csv(dir_a / "plush_state.csv", index=False)

        # Modo A teve 1 handoff aos 2650 ms (frame 50)
        df_events_a = pd.DataFrame([{
            "t_capture_ms": 2650,
            "event_type": "handoff",
            "donor_id": 1,
            "receiver_id": 2,
            "duration_s": 0.5
        }])
        df_events_a.to_csv(dir_a / "events.csv", index=False)

        # Cria dados para Modo B:
        # 95% de concordância (errou 5 frames no início da troca)
        holders_b = [1] * 45 + [None] * 5 + [2] * 50
        states_b = ["COM_PORTADOR"] * 45 + ["PASSAGEM"] * 5 + ["COM_PORTADOR"] * 50

        df_plush_b = pd.DataFrame({
            "t_capture_ms": times_ms,
            "frame_idx": list(range(100)),
            "state": states_b,
            "holder_id": holders_b,
            "holder_conf": [0.85] * 100,
            "source": ["auto"] * 100
        })
        df_plush_b.to_csv(dir_b / "plush_state.csv", index=False)

        # Modo B detectou o handoff com 50ms de latência aos 2700 ms
        df_events_b = pd.DataFrame([{
            "t_capture_ms": 2700,
            "event_type": "handoff",
            "donor_id": 1,
            "receiver_id": 2,
            "duration_s": 0.6
        }])
        df_events_b.to_csv(dir_b / "events.csv", index=False)

        report_file = Path(tmpdir) / "comparison_report.md"
        results = compare_sessions(
            mode_a_path=str(dir_a),
            mode_b_path=str(dir_b),
            tolerance_ms=500.0,
            output_report_path=str(report_file)
        )

        assert results["total_aligned_frames"] == 100
        assert results["holder_accuracy"] == 0.95
        assert results["state_accuracy"] == 0.95
        assert results["handoff_metrics"]["true_positives"] == 1
        assert results["handoff_metrics"]["false_positives"] == 0
        assert results["handoff_metrics"]["false_negatives"] == 0
        assert results["handoff_metrics"]["precision"] == 1.0
        assert results["handoff_metrics"]["recall"] == 1.0
        assert results["handoff_metrics"]["f1_score"] == 1.0
        assert results["handoff_metrics"]["mean_latency_ms"] == 50.0

        assert report_file.exists()
        report_text = report_file.read_text(encoding="utf-8")
        assert "# Relatório de Comparação" in report_text
        assert "95.00%" in report_text
