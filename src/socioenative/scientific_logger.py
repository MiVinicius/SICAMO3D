"""
Gravador Científico de Dados Socioenativos (Schema v3).
Implementa o esquema v3 completo:
- trajectories.csv: série temporal contínua frame-a-frame (inclusive em sala vazia), com timestamp de captura, role, presence_state, track_state e scene_id;
- plush_state.csv: estado da pelúcia (COM_PORTADOR, PASSAGEM, SEM_PORTADOR, INDETERMINADO), portador exclusivo e coordenadas;
- events.csv: passagens de posse (handoffs com doador/receptor/flags), mudanças de cena e comandos do operador;
- session.yaml: metadados da sessão científica, versão de protocolo e parâmetros de calibração;
- raw_sensor.jsonl: log bruto de articulações reduzidas e candidatos da pelúcia para replay e calibração offline.
"""
import os
import csv
import json
import time
import yaml
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

class ScientificLogger:
    def __init__(self, output_dir: str = "recordings", session_name: Optional[str] = None):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        timestamp_str = session_name or time.strftime("%Y%m%d_%H%M%S")
        self.session_id = timestamp_str

        # Cria pasta específica para a sessão
        self.session_dir = os.path.join(self.output_dir, f"session_{timestamp_str}")
        os.makedirs(self.session_dir, exist_ok=True)

        # Caminhos dos arquivos v3
        self.traj_filepath = os.path.join(self.session_dir, "trajectories.csv")
        self.plush_filepath = os.path.join(self.session_dir, "plush_state.csv")
        self.events_filepath = os.path.join(self.session_dir, "events.csv")
        self.raw_filepath = os.path.join(self.session_dir, "raw_sensor.jsonl")
        self.session_yaml_path = os.path.join(self.session_dir, "session.yaml")

        # Abre descritores mantendo-os abertos para alto desempenho de E/S
        self._f_traj = open(self.traj_filepath, mode='w', newline='', encoding='utf-8')
        self._writer_traj = csv.writer(self._f_traj)
        self._writer_traj.writerow([
            "t_capture_ms", "frame_idx", "track_id", "pos_x", "pos_y", "pos_z",
            "ground_speed_mps", "posture", "role", "presence_state", "track_state", "scene_id"
        ])

        self._f_plush = open(self.plush_filepath, mode='w', newline='', encoding='utf-8')
        self._writer_plush = csv.writer(self._f_plush)
        self._writer_plush.writerow([
            "t_capture_ms", "frame_idx", "state", "holder_id", "holder_conf",
            "plush_x", "plush_z", "n_candidates", "source", "scene_id"
        ])

        self._f_events = open(self.events_filepath, mode='w', newline='', encoding='utf-8')
        self._writer_events = csv.writer(self._f_events)
        self._writer_events.writerow([
            "t_capture_ms", "event_type", "scene_id", "donor_id", "receiver_id",
            "distance_m", "duration_s", "confidence", "flags", "details"
        ])

        self._f_raw = open(self.raw_filepath, mode='w', encoding='utf-8')

        self._flush_counter = 0
        self._write_session_metadata()

    def _write_session_metadata(self):
        meta = {
            "session_id": self.session_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "schema_version": "v3.0",
            "system_version": "0.2.0",
            "dataset_purpose": "teatro_socioenativo_pelucia_movel",
            "ethics_protocol": "esqueleto_anonimizado_e_observador_ao_vivo",
            "sensor": "Microsoft Kinect v2",
            "directml_device": "AMD Radeon RX 6600",
            "coordinate_frame": "room_calibrated_ground_plane"
        }
        with open(self.session_yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(meta, f, sort_keys=False)

    def log_frame(self, 
                  t_capture_ms: int,
                  frame_idx: int,
                  tracks: List,
                  holder_info: Dict[str, Any],
                  scene_id: str,
                  raw_observations: Optional[Dict[str, Any]] = None):
        """
        Grava o estado de um frame em todas as tabelas sincronizadas do Schema v3.
        Grava mesmo se não houver participantes na sala (preservando série temporal contínua).
        """
        # 1. Trajetórias das pessoas
        if tracks:
            for trk in tracks:
                px, py, pz = trk.position
                self._writer_traj.writerow([
                    t_capture_ms,
                    frame_idx,
                    trk.track_id,
                    round(px, 3) if not np.isnan(px) else "",
                    round(py, 3) if not np.isnan(py) else "",
                    round(pz, 3) if not np.isnan(pz) else "",
                    round(trk.ground_speed, 3),
                    trk.posture,
                    trk.role,
                    trk.presence_state,
                    trk.track_state,
                    scene_id
                ])
        else:
            # Sala vazia: registra frame nulo para manter linha do tempo ininterrupta
            self._writer_traj.writerow([
                t_capture_ms, frame_idx, -1, "", "", "", 0.0, "vazio", "nenhum", "nenhum", "nenhum", scene_id
            ])

        # 2. Estado da Pelúcia
        st = holder_info.get("state", "INDETERMINADO")
        h_id = holder_info.get("holder_id")
        h_conf = holder_info.get("confidence", 0.0)
        p_x = holder_info.get("plush_x", "")
        p_z = holder_info.get("plush_z", "")
        n_cands = 1 if holder_info.get("plush_pos_3d") is not None else 0
        src = holder_info.get("source", "auto")

        self._writer_plush.writerow([
            t_capture_ms, frame_idx, st, h_id if h_id is not None else -1,
            h_conf, p_x if p_x is not None else "", p_z if p_z is not None else "",
            n_cands, src, scene_id
        ])

        # 3. Eventos gerados no frame (ex: handoff)
        for ev in holder_info.get("events", []):
            self.log_event(
                event_type=ev.get("type", "evento"),
                scene_id=scene_id,
                t_capture_ms=ev.get("timestamp_ms", t_capture_ms),
                donor_id=ev.get("donor_id"),
                receiver_id=ev.get("receiver_id"),
                distance_m=ev.get("distance_interpersonal_m", 0.0),
                duration_s=ev.get("duration_s", 0.0),
                confidence=ev.get("confidence", 1.0),
                flags=ev.get("flags", ""),
                details=ev.get("details", "")
            )

        # 4. Log bruto de articulações reduzidas e candidatos da pelúcia (para replay offline)
        if raw_observations is not None:
            raw_entry = {
                "t_capture_ms": t_capture_ms,
                "frame_idx": frame_idx,
                "scene_id": scene_id,
                "raw": raw_observations
            }
            self._f_raw.write(json.dumps(raw_entry) + "\n")

        # Flush periódico
        self._flush_counter += 1
        if self._flush_counter % 30 == 0:
            self._f_traj.flush()
            self._f_plush.flush()
            self._f_events.flush()
            self._f_raw.flush()

    def log_event(self, 
                  event_type: str, 
                  scene_id: str,
                  t_capture_ms: Optional[int] = None,
                  donor_id: Optional[int] = None, 
                  receiver_id: Optional[int] = None,
                  distance_m: float = 0.0,
                  duration_s: float = 0.0,
                  confidence: float = 1.0,
                  flags: str = "",
                  details: str = ""):
        """Registra um evento específico em events.csv."""
        now_ms = t_capture_ms if t_capture_ms is not None else int(time.time() * 1000)
        self._writer_events.writerow([
            now_ms, event_type, scene_id,
            donor_id if donor_id is not None else "",
            receiver_id if receiver_id is not None else "",
            round(distance_m, 3), round(duration_s, 2),
            round(confidence, 3), flags, details
        ])
        self._f_events.flush()

    def close(self):
        """Fecha com segurança todos os descritores de arquivo."""
        for f in [self._f_traj, self._f_plush, self._f_events, self._f_raw]:
            try:
                if f and not f.closed:
                    f.flush()
                    f.close()
            except Exception:
                pass
