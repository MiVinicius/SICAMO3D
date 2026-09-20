"""
Testes sintéticos automatizados para a Máquina de Estados de Posse e Handoff (HolderInference).
Valida posse exclusiva, abraço, histerese, oclusão temporária e eventos de passagem A -> B.
"""
import pytest
import numpy as np
from src.tracking.tracker_3d import Track3D
from src.socioenative.holder_inference import HolderInference, HolderState

def _create_synthetic_track(track_id: int, pos: tuple, hand_right_pos: tuple = None, dwell_time_s: float = 5.0) -> Track3D:
    trk = Track3D(track_id=track_id, init_pos=pos, timestamp_s=0.0)
    trk.is_confirmed = True
    trk.dwell_time_s = dwell_time_s
    
    # Prepara keypoints 3D sintéticos com pulso direito na posição desejada
    k3d = np.full((17, 4), np.nan, dtype=np.float32)
    k3d[:, 3] = 0.0
    if hand_right_pos:
        k3d[10, :3] = hand_right_pos
        k3d[10, 3] = 0.90 # confiança alta
    trk.last_keypoints_3d = k3d
    return trk

def test_initial_possession_wrist():
    inference = HolderInference()
    trk_a = _create_synthetic_track(1, pos=(0.0, 1.0, 2.0), hand_right_pos=(0.15, 0.95, 2.0))
    
    # Pelúcia a 5 cm do pulso direito de A
    plush_cand = [{"pos_3d": (0.20, 0.95, 2.0), "confidence": 0.85}]
    
    result = inference.process([trk_a], plush_cand, timestamp_s=1.0)
    assert result["state"] == "COM_PORTADOR"
    assert result["holder_id"] == 1
    assert result["confidence"] > 0.50

def test_plush_on_table_no_person_near():
    inference = HolderInference()
    trk_a = _create_synthetic_track(1, pos=(-2.0, 1.0, 3.0), hand_right_pos=(-1.8, 1.0, 3.0))
    
    # Pelúcia no centro da sala (a mais de 2m de A)
    plush_cand = [{"pos_3d": (0.0, 0.75, 1.5), "confidence": 0.80}]
    
    result = inference.process([trk_a], plush_cand, timestamp_s=1.0)
    assert result["state"] == "SEM_PORTADOR"
    assert result["holder_id"] is None

def test_hugged_plush_torso_score():
    inference = HolderInference()
    # Pulso longe (ex: braços caídos), mas a pelúcia está a 15 cm do centro do tronco de A
    trk_a = _create_synthetic_track(1, pos=(0.0, 1.1, 2.0), hand_right_pos=(0.40, 0.60, 2.0))
    plush_cand = [{"pos_3d": (0.05, 1.1, 1.9), "confidence": 0.82}]
    
    result = inference.process([trk_a], plush_cand, timestamp_s=1.0)
    assert result["state"] == "COM_PORTADOR"
    assert result["holder_id"] == 1

def test_temporary_occlusion_cached_holder():
    inference = HolderInference(timeout_s=1.5)
    trk_a = _create_synthetic_track(1, pos=(0.0, 1.0, 2.0), hand_right_pos=(0.1, 1.0, 2.0))
    plush_cand = [{"pos_3d": (0.1, 1.0, 2.0), "confidence": 0.90}]
    
    # 1. A possui a pelúcia
    res1 = inference.process([trk_a], plush_cand, timestamp_s=1.0)
    assert res1["state"] == "COM_PORTADOR"
    assert res1["holder_id"] == 1

    # 2. Pelúcia oculta por 0.5s (ex: abraçada contra o corpo)
    res2 = inference.process([trk_a], [], timestamp_s=1.5)
    assert res2["state"] == "COM_PORTADOR"
    assert res2["holder_id"] == 1
    assert res2["source"] == "auto_cached"

    # 3. Pelúcia sumiu por 2.0s (> timeout de 1.5s) -> transita para INDETERMINADO
    res3 = inference.process([trk_a], [], timestamp_s=3.2)
    assert res3["state"] == "INDETERMINADO"
    assert res3["holder_id"] is None

def test_clean_handoff_a_to_b():
    inference = HolderInference(handoff_window_s=2.0)
    trk_a = _create_synthetic_track(1, pos=(0.0, 1.0, 2.0), hand_right_pos=(0.1, 1.0, 2.0))
    trk_b = _create_synthetic_track(2, pos=(0.5, 1.0, 2.0), hand_right_pos=(0.4, 1.0, 2.0))

    # Frame 1: A segura a pelúcia
    plush_a = [{"pos_3d": (0.1, 1.0, 2.0), "confidence": 0.90}]
    res1 = inference.process([trk_a, trk_b], plush_a, timestamp_s=1.0)
    assert res1["state"] == "COM_PORTADOR"
    assert res1["holder_id"] == 1

    # Frame 2: Pelúcia vai para a mão de B
    plush_b = [{"pos_3d": (0.4, 1.0, 2.0), "confidence": 0.90}]
    res2 = inference.process([trk_a, trk_b], plush_b, timestamp_s=1.3)
    # Entra em estado de PASSAGEM
    assert res2["state"] == "PASSAGEM"

    # Frame 3: B consolida a posse
    res3 = inference.process([trk_a, trk_b], plush_b, timestamp_s=1.8)
    assert res3["state"] == "COM_PORTADOR"
    assert res3["holder_id"] == 2

    # Verifica se gerou o evento de handoff com métricas corretas
    events = res3["events"]
    assert len(events) == 1
    ev = events[0]
    assert ev["type"] == "handoff"
    assert ev["donor_id"] == 1
    assert ev["receiver_id"] == 2
    assert ev["id_ambiguous"] is False

def test_operator_manual_override():
    inference = HolderInference()
    trk_a = _create_synthetic_track(1, pos=(0.0, 1.0, 2.0), hand_right_pos=(0.1, 1.0, 2.0))
    trk_b = _create_synthetic_track(2, pos=(2.0, 1.0, 3.0), hand_right_pos=(2.1, 1.0, 3.0))

    # Fisicamente a pelúcia está com A
    plush_a = [{"pos_3d": (0.1, 1.0, 2.0), "confidence": 0.90}]
    
    # O operador clica e força a pelúcia no participante 2
    inference.set_manual_holder(track_id=2, duration_s=5.0, timestamp_s=1.0)
    
    res = inference.process([trk_a, trk_b], plush_a, timestamp_s=1.0)
    assert res["state"] == "COM_PORTADOR"
    assert res["holder_id"] == 2
    assert res["source"] == "manual"
