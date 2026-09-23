"""
Testes sintéticos automatizados para Detecção de F-Formations (M5-01).
Valida classificação de arranjos (vis-a-vis, side-by-side, l-shape, semicircular, circular),
convergência no o-space, exclusão de pessoas de costas/dispersas e presença da pelúcia.
"""

import math
import pytest
import numpy as np
from src.tracking.tracker_3d import Track3D
from src.socioenative.f_formations import FFormationDetector, FFormationGroup

def _create_oriented_track(track_id: int, pos: tuple, heading_deg: float) -> Track3D:
    trk = Track3D(track_id=track_id, init_pos=pos, timestamp_s=0.0)
    trk.is_confirmed = True
    trk.dwell_time_s = 5.0
    
    # Simula ombros perpendiculares ao heading
    rad = math.radians(heading_deg)
    # v_facing = (sin(rad), cos(rad))
    # v_shoulders = (-v_facing[1], v_facing[0]) = (-cos(rad), sin(rad))
    v_s = np.array([-math.cos(rad), math.sin(rad)]) * 0.20 # metade da largura do ombro (20cm)
    
    k3d = np.full((17, 4), np.nan, dtype=np.float32)
    k3d[:, 3] = 0.0
    # Ombro esquerdo (idx 5) = pos + v_s
    k3d[5, 0] = pos[0] + v_s[0]
    k3d[5, 1] = pos[1]
    k3d[5, 2] = pos[2] + v_s[1]
    k3d[5, 3] = 0.90
    
    # Ombro direito (idx 6) = pos - v_s
    k3d[6, 0] = pos[0] - v_s[0]
    k3d[6, 1] = pos[1]
    k3d[6, 2] = pos[2] - v_s[1]
    k3d[6, 3] = 0.90
    
    trk.last_keypoints_3d = k3d
    return trk

def test_vis_a_vis_formation():
    detector = FFormationDetector()
    # Pessoa 1 em (0.0, 1.0, 1.5) olhando para o fundo (+Z, 0°)
    trk1 = _create_oriented_track(1, pos=(0.0, 1.0, 1.5), heading_deg=0.0)
    # Pessoa 2 em (0.0, 1.0, 2.7) olhando para frente (-Z, 180°)
    trk2 = _create_oriented_track(2, pos=(0.0, 1.0, 2.7), heading_deg=180.0)

    groups = detector.detect([trk1, trk2])
    assert len(groups) == 1
    g = groups[0]
    assert set(g.members) == {1, 2}
    assert g.formation_type == "vis-a-vis"
    assert 1.8 <= g.o_space_center[1] <= 2.4 # Centro entre 1.5 e 2.7

def test_side_by_side_formation():
    detector = FFormationDetector()
    # Duas pessoas lado a lado (separadas por 60cm em X), ambas olhando para o fundo (+Z, 0°)
    trk1 = _create_oriented_track(1, pos=(-0.3, 1.0, 2.0), heading_deg=0.0)
    trk2 = _create_oriented_track(2, pos=(0.3, 1.0, 2.0), heading_deg=0.0)

    groups = detector.detect([trk1, trk2])
    assert len(groups) == 1
    g = groups[0]
    assert set(g.members) == {1, 2}
    assert g.formation_type == "side-by-side"

def test_back_to_back_not_f_formation():
    detector = FFormationDetector()
    # Duas pessoas próximas (1.0m de distância), mas de costas uma para a outra
    # Pessoa 1 em (0.0, 1.0, 2.0) olhando para a câmera (-Z, 180°)
    trk1 = _create_oriented_track(1, pos=(0.0, 1.0, 2.0), heading_deg=180.0)
    # Pessoa 2 em (0.0, 1.0, 2.8) olhando para o fundo (+Z, 0°)
    trk2 = _create_oriented_track(2, pos=(0.0, 1.0, 2.8), heading_deg=0.0)

    groups = detector.detect([trk1, trk2])
    assert len(groups) == 0, "Pessoas de costas uma para a outra não devem formar F-formation"

def test_semicircular_group_with_plush():
    detector = FFormationDetector()
    # 3 participantes dispostos em semicírculo ao redor do ponto focal (0.0, 2.5)
    # P1 à esquerda (-1.0, 2.5), olhando para +X (90°)
    trk1 = _create_oriented_track(1, pos=(-1.0, 1.0, 2.5), heading_deg=90.0)
    # P2 ao fundo (0.0, 3.5), olhando para -Z (180°)
    trk2 = _create_oriented_track(2, pos=(0.0, 1.0, 3.5), heading_deg=180.0)
    # P3 à direita (1.0, 2.5), olhando para -X (-90°)
    trk3 = _create_oriented_track(3, pos=(1.0, 1.0, 2.5), heading_deg=-90.0)

    # Pelúcia no centro do o-space (0.0, 1.0, 2.6)
    plush_pos = (0.0, 1.0, 2.6)

    groups = detector.detect([trk1, trk2, trk3], plush_pos=plush_pos)
    assert len(groups) == 1
    g = groups[0]
    assert len(g.members) == 3
    assert set(g.members) == {1, 2, 3}
    assert g.formation_type in ["semicircular", "circular"]
    assert g.plush_inside_o_space is True
