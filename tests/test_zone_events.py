"""
Testes unitários e sintéticos para o Milestone M1:
- M1-01: Registro de eventos de entrada/saída ('enter'/'exit') do stand em ZoneManager;
- M1-02: Distinção cinemática entre trânsito e permanência intencional pela velocidade horizontal (ground_speed);
- M1-03: Filtragem de métricas por alcance confiável (max_reliable_range_m).
"""
import pytest
import numpy as np

from src.core.zone_manager import ZoneManager
from src.tracking.tracker_3d import Track3D, Tracker3D
from src.core.pipeline import Pipeline
from tests.test_pipeline_smoke import MockKinectSensor

def test_zone_enter_and_exit_events():
    """
    Issue M1-01: Valida detecção de transições enter/exit no stand:
    - Entrada na zona stand_total emite evento 'enter';
    - Permanência na zona não repete o evento;
    - Saída da zona emite evento 'exit';
    - Perda de rastreamento de participante dentro do stand emite 'exit'.
    """
    zm = ZoneManager()
    trk = Track3D(track_id=1, init_pos=(0.0, 1.0, 5.0), timestamp_s=0.0) # Fora do stand (Z > 4.5m)

    # 1. Ponto inicial fora do stand
    res = zm.evaluate_tracks([trk])
    assert res[0]["in_stand"] is False
    assert len(zm.pending_zone_events) == 0

    # 2. Entra no stand em Z=2.5m (dentro do polígono stand_total)
    trk.kf.x[2] = 2.5
    res = zm.evaluate_tracks([trk])
    assert res[0]["in_stand"] is True
    assert len(zm.pending_zone_events) == 1
    assert zm.pending_zone_events[0]["type"] == "enter"
    assert zm.pending_zone_events[0]["track_id"] == 1
    assert zm.pending_zone_events[0]["zone"] == "stand_total"

    # 3. Permanece no stand
    trk.kf.x[2] = 2.6
    res = zm.evaluate_tracks([trk])
    assert res[0]["in_stand"] is True
    assert len(zm.pending_zone_events) == 0  # Sem repetição de enter

    # 4. Sai do stand (Z=5.0m)
    trk.kf.x[2] = 5.0
    res = zm.evaluate_tracks([trk])
    assert res[0]["in_stand"] is False
    assert len(zm.pending_zone_events) == 1
    assert zm.pending_zone_events[0]["type"] == "exit"
    assert zm.pending_zone_events[0]["track_id"] == 1

    # 5. Entra novamente e desaparece
    trk.kf.x[2] = 2.5
    zm.evaluate_tracks([trk])
    assert zm._track_in_stand.get(1) is True

    # Próximo frame: lista vazia de tracks (participante sumiu enquanto estava no stand)
    zm.evaluate_tracks([])
    assert len(zm.pending_zone_events) == 1
    assert zm.pending_zone_events[0]["type"] == "exit"
    assert zm.pending_zone_events[0]["track_id"] == 1

def test_transit_vs_stationary_speed_classification():
    """
    Issue M1-02: Distingue transeunte em trânsito de espectador intencional:
    - Transeunte caminhando rápido (ground_speed >= 0.50 m/s) permanece como 'passante'
      mesmo após vários segundos;
    - Espectador parado (ground_speed < 0.40 m/s) vira 'plateia' após 2.5s e
      'participante_ativo' após 10s.
    """
    # Caso A: Transeunte em movimento contínuo a 1.0 m/s
    transeunte = Track3D(track_id=1, init_pos=(0.0, 1.0, 1.0), timestamp_s=0.0)
    t_sim = 0.0
    dt = 0.05  # 20 Hz
    # Simula 5 segundos caminhando a 1.0 m/s
    for step in range(100):
        t_sim += dt
        x_pos = step * (1.0 * dt)
        # Injeta velocidade no filtro de Kalman para ground_speed ~ 1.0 m/s
        transeunte.kf.x[3] = 1.0 # vx = 1.0 m/s
        transeunte.update(pos_3d=(x_pos, 1.0, 2.0), timestamp_s=t_sim)

    assert transeunte.dwell_time_s >= 4.5
    assert transeunte.ground_speed >= 0.80
    assert transeunte.stationary_time_s == 0.0
    assert transeunte.presence_state == "passante"  # Continua passante, não vira plateia!

    # Caso B: Espectador parado em frente à cena
    espectador = Track3D(track_id=2, init_pos=(0.0, 1.0, 2.5), timestamp_s=0.0)
    t_sim = 0.0
    # Simula 3 segundos parado (< 0.10 m/s)
    for _ in range(60):
        t_sim += dt
        espectador.kf.x[3] = 0.0
        espectador.kf.x[5] = 0.0
        espectador.update(pos_3d=(0.0, 1.0, 2.5), timestamp_s=t_sim)

    assert espectador.stationary_time_s >= 2.5
    assert espectador.presence_state == "plateia"

    # Simula até atingir 11 segundos parado
    for _ in range(160):
        t_sim += dt
        espectador.kf.x[3] = 0.0
        espectador.kf.x[5] = 0.0
        espectador.update(pos_3d=(0.0, 1.0, 2.5), timestamp_s=t_sim)

    assert espectador.stationary_time_s >= 10.0
    assert espectador.presence_state == "participante_ativo"

def test_metrics_filtered_by_reliable_range():
    """
    Issue M1-03: Participantes fora do alcance confiável (> 4.5m) não devem
    contaminar o cálculo de plateia e métricas do PlushMetricsAnalyzer.
    """
    mock_sensor = MockKinectSensor()
    pipeline = Pipeline(sensor=mock_sensor, enable_logger=False)

    # Cria dois tracks confirmados
    trk_valid = Track3D(track_id=1, init_pos=(0.0, 1.0, 2.5))
    trk_valid.hits = 5
    trk_valid.is_confirmed = True

    trk_far = Track3D(track_id=2, init_pos=(0.0, 1.0, 5.2)) # > 4.5m
    trk_far.hits = 5
    trk_far.is_confirmed = True

    pipeline.tracker.tracks = [trk_valid, trk_far]
    pipeline.last_tracks = [trk_valid, trk_far]

    # Avaliação de zonas
    evals = pipeline.zone_manager.evaluate_tracks([trk_valid, trk_far])
    assert evals[0]["in_reliable_range"] is True
    assert evals[1]["in_reliable_range"] is False

    # Executa step com detecções simuladas correspondentes
    sim_dets = [
        {"pos_3d": (0.0, 1.0, 2.5), "keypoints_3d": np.ones((17, 4), dtype=np.float32), "bbox": [100, 100, 200, 300], "role": "participante"},
        {"pos_3d": (0.0, 1.0, 5.2), "keypoints_3d": np.ones((17, 4), dtype=np.float32), "bbox": [10, 10, 50, 80], "role": "participante"}
    ]
    for _ in range(5):
        pipeline.step(remote_detections=sim_dets)

    # Verifica métricas da cena
    summary = pipeline.plush_metrics.get_scene_summary("Cena 1")
    # Apenas o participante no alcance confiável deve contar como audiência
    assert summary["total_participants"] == 1
    assert summary["total_audience_participants"] == 1
    pipeline.close()
