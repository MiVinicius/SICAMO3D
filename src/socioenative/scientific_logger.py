"""
Gravador de dados científicos com carimbo de data/hora em milissegundos para análise socioenativa.
"""
import os
import csv
import time
from typing import List, Dict, Optional

class ScientificLogger:
    def __init__(self, output_dir: str = "recordings"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        self.trajectory_filepath = os.path.join(self.output_dir, f"trajectories_{timestamp_str}.csv")
        self.interaction_filepath = os.path.join(self.output_dir, f"interactions_{timestamp_str}.csv")

        # Mantém handles de arquivos abertos para evitar syscalls de open/close a cada frame
        self._f_traj = open(self.trajectory_filepath, mode='w', newline='', encoding='utf-8')
        self._writer_traj = csv.writer(self._f_traj)
        self._writer_traj.writerow([
            "timestamp_ms", "track_id", "pos_x", "pos_y", "pos_z", 
            "speed_mps", "posture", "held_toy"
        ])

        self._f_inter = open(self.interaction_filepath, mode='w', newline='', encoding='utf-8')
        self._writer_inter = csv.writer(self._f_inter)
        self._writer_inter.writerow([
            "timestamp_ms", "track_id_1", "track_id_2", "distance_m", 
            "proxemic_zone", "joint_toy", "event_type"
        ])
        self._flush_counter = 0

    def log_trajectories(self, tracks: List):
        if not tracks or self._f_traj.closed:
            return
        now_ms = int(time.time() * 1000)
        for trk in tracks:
            px, py, pz = trk.position
            self._writer_traj.writerow([
                now_ms, trk.track_id, round(px, 3), round(py, 3), round(pz, 3),
                round(trk.speed, 3), trk.posture, trk.held_toy or "nenhum"
            ])
        self._flush_counter += 1
        if self._flush_counter % 30 == 0:
            self._f_traj.flush()

    def log_interactions(self, proxemic_events: List[Dict], joint_events: List[Dict]):
        if self._f_inter.closed:
            return
        now_ms = int(time.time() * 1000)
        for pe in proxemic_events:
            t1, t2 = pe["id_pair"]
            self._writer_inter.writerow([
                now_ms, t1, t2, pe["distance_m"], pe["zone"], "nenhum", "proxemica"
            ])
        for je in joint_events:
            pts = je["participant_ids"]
            if len(pts) >= 2:
                self._writer_inter.writerow([
                    now_ms, pts[0], pts[1], 0.0, "compartilhamento", je["toy_name"], "atencao_conjunta"
                ])
        if self._flush_counter % 30 == 0:
            self._f_inter.flush()

    def close(self):
        """Fecha com segurança todos os arquivos abertos."""
        if hasattr(self, '_f_traj') and not self._f_traj.closed:
            self._f_traj.flush()
            self._f_traj.close()
        if hasattr(self, '_f_inter') and not self._f_inter.closed:
            self._f_inter.flush()
            self._f_inter.close()

