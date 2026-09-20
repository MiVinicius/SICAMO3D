"""
Testes sintéticos automatizados para o Tracker3D e KalmanFilter3D (sem depender de hardware).
Valida associação, confirmação de tracks, velocidade no plano do chão e supressão de fantasmas.
"""
import pytest
import numpy as np
from src.tracking.kalman_filter_3d import KalmanFilter3D
from src.tracking.tracker_3d import Tracker3D, Track3D

def test_kalman_filter_3d_ground_speed_and_dt():
    kf = KalmanFilter3D(init_pos=(0.0, 1.0, 2.0), dt=1.0/30.0)
    assert kf.position == (0.0, 1.0, 2.0)
    assert kf.speed == 0.0

    # Simula movimento retilíneo no plano do chão X-Z com velocidade constante de 1 m/s (passo de 0.033m a 30 FPS)
    t = 0.0
    for i in range(1, 31):
        t += 1.0 / 30.0
        kf.predict(dt=1.0/30.0)
        kf.update((i * 0.0333, 1.0, 2.0))

    # Ao final de 1 segundo (30 frames), deve estar perto de x = 1.0m
    pos = kf.position
    assert abs(pos[0] - 1.0) < 0.15
    assert abs(pos[1] - 1.0) < 0.10
    # Velocidade no plano do chão deve estar próxima de 1.0 m/s
    assert abs(kf.ground_speed - 1.0) < 0.25

def test_kalman_filter_ignores_nan():
    kf = KalmanFilter3D(init_pos=(0.0, 1.0, 2.0))
    kf.predict()
    kf.update((np.nan, 1.0, 2.0)) # Medição inválida
    # Não deve estragar o estado nem propagar NaN
    assert not np.isnan(kf.position).any()

def test_tracker_3d_confirmation_and_ghost_suppression():
    tracker = Tracker3D(max_distance_m=0.85, max_lost_frames=30)

    # Frame 1: detecção única
    det1 = [{"pos_3d": (0.0, 1.0, 2.0)}]
    active = tracker.update(det1, timestamp_s=0.0)
    # No 1º hit, track ainda não está confirmado (evita falsos positivos rápidos)
    assert len(active) == 0
    assert len(tracker.tracks) == 1

    # Frame 2:
    active = tracker.update(det1, timestamp_s=0.033)
    assert len(active) == 0

    # Frame 3: agora atinge 3 hits -> CONFIRMADO!
    active = tracker.update(det1, timestamp_s=0.066)
    assert len(active) == 1
    assert active[0].track_id == 1
    assert active[0].is_confirmed is True

    # Simula alvo fantasma temporário que aparece por apenas 1 frame e some
    tracker.update([{"pos_3d": (0.0, 1.0, 2.0)}, {"pos_3d": (2.0, 1.0, 3.0)}], timestamp_s=0.10)
    assert len(tracker.tracks) == 2 # 1 confirmado e 1 não confirmado

    # O fantasma some nos próximos frames
    for i in range(1, 7):
        tracker.update([{"pos_3d": (0.0, 1.0, 2.0)}], timestamp_s=0.10 + i * 0.033)

    # O fantasma não confirmado deve ter sido excluído após 5 frames
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].track_id == 1

def test_tracker_3d_target_movement_and_id_consistency():
    tracker = Tracker3D(max_distance_m=0.85)

    # Pessoa andando a 1.2 m/s ao longo de 20 frames (4 cm por frame)
    t = 0.0
    for i in range(20):
        x = -1.0 + i * 0.04
        det = [{"pos_3d": (x, 1.0, 2.5), "role": "participante"}]
        tracker.update(det, timestamp_s=t)
        t += 0.033

    confirmed = [t for t in tracker.tracks if t.is_confirmed]
    assert len(confirmed) == 1
    assert confirmed[0].track_id == 1
    assert confirmed[0].hits == 20
    assert confirmed[0].role == "participante"
