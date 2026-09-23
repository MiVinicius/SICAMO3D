"""
Testes automatizados para Milestone M5:
1. Classificação proxêmica no momento do handoff (M5-02).
2. Pipeline de ETL em lote e relatórios analíticos (tools/etl.py) (M5-03 e M5-04).
"""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.tracking.tracker_3d import Track3D
from src.socioenative.holder_inference import HolderInference
from src.socioenative.proxemics import ProxemicsAnalyzer
from tools.etl import (
    calc_gini,
    extract_possession_intervals,
    extract_handoff_events,
    extract_presence_summary,
    extract_f_formations_intervals,
    build_scene_summary,
    run_etl,
    generate_analytics_report
)

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

def test_handoff_event_proxemic_classification():
    inference = HolderInference(handoff_window_s=2.0)
    # Pessoa 1 e 2 a 30 cm de distância no plano métrico (zona intima: < 35 cm)
    trk_a = _make_track(1, pos=(0.0, 1.0, 2.0), hand_right=(0.05, 1.0, 2.0))
    trk_b = _make_track(2, pos=(0.30, 1.0, 2.0), hand_right=(0.30, 1.0, 2.0))

    # Frame 1: A segura a pelúcia
    plush_a = [{"pos_3d": (0.05, 1.0, 2.0), "confidence": 0.90}]
    res1 = inference.process([trk_a, trk_b], plush_a, timestamp_s=1.0)
    assert res1["state"] == "COM_PORTADOR"
    assert res1["holder_id"] == 1

    # Frame 2: Pelúcia vai para B (inicia passagem)
    plush_b = [{"pos_3d": (0.30, 1.0, 2.0), "confidence": 0.90}]
    res2 = inference.process([trk_a, trk_b], plush_b, timestamp_s=1.3)
    assert res2["state"] == "PASSAGEM"

    # Frame 3: B consolida a posse -> emite handoff
    res3 = inference.process([trk_a, trk_b], plush_b, timestamp_s=1.8)
    assert res3["state"] == "COM_PORTADOR"
    assert res3["holder_id"] == 2
    events = res3.get("events", [])
    assert len(events) == 1
    ev = events[0]
    assert ev["type"] == "handoff"
    assert ev["donor_id"] == 1
    assert ev["receiver_id"] == 2
    assert "proxemic_zone" in ev
    # Distância de 30cm com threshold de 35cm para íntima
    assert ev["proxemic_zone"] == "intima"
    assert "zone_intima" in ev["flags"]

def test_etl_pipeline_and_analytics_report():
    with tempfile.TemporaryDirectory() as tmpdir:
        sess_dir = Path(tmpdir) / "session_001"
        sess_dir.mkdir()

        # Cria trajectories.csv com 2 atores e 1 cena
        n_frames = 60
        traj_data = []
        for f in range(n_frames):
            t_ms = 1000 + f * 33
            # Ator 1 parado (< 0.40 m/s)
            traj_data.append({
                "t_capture_ms": t_ms,
                "frame_idx": f,
                "track_id": 1,
                "pos_x": 0.0,
                "pos_y": 1.0,
                "pos_z": 2.0,
                "ground_speed_mps": 0.15,
                "posture": "em_pe",
                "role": "participante",
                "presence_state": "plateia",
                "track_state": "medido",
                "scene_id": "Cena 1"
            })
            # Ator 2 em trânsito (0.75 m/s)
            traj_data.append({
                "t_capture_ms": t_ms,
                "frame_idx": f,
                "track_id": 2,
                "pos_x": 0.5 + f * 0.02,
                "pos_y": 1.0,
                "pos_z": 2.2,
                "ground_speed_mps": 0.75,
                "posture": "em_pe",
                "role": "participante",
                "presence_state": "passante",
                "track_state": "medido",
                "scene_id": "Cena 1"
            })
        pd.DataFrame(traj_data).to_csv(sess_dir / "trajectories.csv", index=False)

        # Cria plush_state.csv: Ator 1 segurou por 30 frames, Ator 2 por 30 frames
        plush_data = []
        for f in range(n_frames):
            t_ms = 1000 + f * 33
            h_id = 1 if f < 30 else 2
            plush_data.append({
                "t_capture_ms": t_ms,
                "frame_idx": f,
                "state": "COM_PORTADOR",
                "holder_id": h_id,
                "holder_conf": 0.90,
                "plush_x": 0.0,
                "plush_z": 2.0,
                "n_candidates": 1,
                "source": "auto",
                "scene_id": "Cena 1"
            })
        pd.DataFrame(plush_data).to_csv(sess_dir / "plush_state.csv", index=False)

        # Cria events.csv com 1 handoff na troca
        events_data = [{
            "t_capture_ms": 2000,
            "event_type": "handoff",
            "scene_id": "Cena 1",
            "donor_id": 1,
            "receiver_id": 2,
            "distance_m": 0.65,
            "duration_s": 0.8,
            "confidence": 0.85,
            "flags": "zone_pessoal",
            "details": "zone:pessoal;dist:0.65m"
        }]
        pd.DataFrame(events_data).to_csv(sess_dir / "events.csv", index=False)

        out_dir = Path(tmpdir) / "etl_output"
        results = run_etl([sess_dir], str(out_dir))

        # Validações das tabelas agregadas
        assert "possession_intervals" in results
        df_poss = results["possession_intervals"]
        assert len(df_poss) == 2 # 1 intervalo para Ator 1, 1 para Ator 2
        assert set(df_poss["holder_id"]) == {1, 2}

        assert "handoff_events_enriched" in results
        df_hd = results["handoff_events_enriched"]
        assert len(df_hd) == 1
        assert df_hd.iloc[0]["proxemic_zone"] == "pessoal"

        assert "presence_summary" in results
        df_pres = results["presence_summary"]
        assert len(df_pres) == 2
        # Ator 1 deve ser plateia
        pres_1 = df_pres[df_pres["track_id"] == 1].iloc[0]
        assert pres_1["stationary_time_s"] > 1.5

        assert "scene_summary" in results
        df_sc = results["scene_summary"]
        assert len(df_sc) == 1
        sc_row = df_sc.iloc[0]
        assert sc_row["distinct_holders"] == 2
        assert sc_row["circulation_ratio"] == 1.0 # 2 portadores / 2 participantes
        assert sc_row["gini_possession"] == 0.0 # Posse 50%-50% -> Gini zero

        # Validação do relatório analítico
        rep_path = Path(tmpdir) / "report.md"
        report_text = generate_analytics_report(results, str(rep_path))
        assert rep_path.exists()
        assert "# Relatório Analítico Consolidado" in report_text
        assert "Pergunta 1: Concentração vs. Circulação" in report_text
        assert "Pergunta 2: Engajamento" in report_text
        assert "Pergunta 3: Proxêmica" in report_text
        assert "Pessoal" in report_text
