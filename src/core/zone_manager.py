"""
Gerenciador de Zonas e Cenas do Espaço Teatral (ZoneManager).
Carrega as definições de zonas e cenas a partir de arquivo YAML e avalia
o pertencimento espacial dos participantes, tempos de permanência e papéis sugeridos.
"""
import os
import yaml
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

class ZoneManager:
    def __init__(self, config_filepath: str = "config/zones.yaml"):
        self.config_filepath = config_filepath
        self.room_info: Dict[str, Any] = {"width_m": 6.0, "depth_m": 6.0, "max_reliable_range_m": 4.5}
        self.zones: List[Dict[str, Any]] = []
        self.scenes: List[Dict[str, Any]] = []
        self.active_scene_idx: int = 0
        self._track_in_stand: Dict[int, bool] = {}
        self.pending_zone_events: List[Dict[str, Any]] = []

        self.load_config()

    def load_config(self):
        if not os.path.exists(self.config_filepath):
            # Cria configuração padrão se não existir
            return
        with open(self.config_filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        self.room_info = data.get("room", self.room_info)
        self.zones = data.get("zones", [])
        self.scenes = data.get("scenes", [{"id": "Cena 1", "name": "Cena Padrão"}])

    @property
    def current_scene_id(self) -> str:
        if 0 <= self.active_scene_idx < len(self.scenes):
            return self.scenes[self.active_scene_idx].get("id", "Cena 1")
        return "Cena 1"

    @property
    def current_scene_name(self) -> str:
        if 0 <= self.active_scene_idx < len(self.scenes):
            return self.scenes[self.active_scene_idx].get("name", "Cena")
        return "Cena"

    def next_scene(self) -> str:
        """Avança para a próxima cena (atalho 'N' do operador)."""
        if self.scenes:
            self.active_scene_idx = (self.active_scene_idx + 1) % len(self.scenes)
        return self.current_scene_id

    def prev_scene(self) -> str:
        """Retorna para a cena anterior."""
        if self.scenes:
            self.active_scene_idx = (self.active_scene_idx - 1) % len(self.scenes)
        return self.current_scene_id

    @staticmethod
    def _point_in_polygon(x: float, z: float, polygon: List[List[float]]) -> bool:
        """Ray-casting algorithm para verificar se (x, z) está dentro do polígono."""
        n = len(polygon)
        inside = False
        p1x, p1z = polygon[0]
        for i in range(1, n + 1):
            p2x, p2z = polygon[i % n]
            if z > min(p1z, p2z):
                if z <= max(p1z, p2z):
                    if x <= max(p1x, p2x):
                        if p1z != p2z:
                            xinters = (z - p1z) * (p2x - p1x) / (p2z - p1z) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1z = p2x, p2z
        return inside

    def get_zones_for_point(self, x: float, z: float) -> List[str]:
        """Retorna lista de IDs das zonas que contêm a coordenada (x, z)."""
        matched = []
        for z_def in self.zones:
            poly = z_def.get("polygon", [])
            if poly and self._point_in_polygon(x, z, poly):
                matched.append(z_def.get("id"))
        return matched

    def evaluate_tracks(self, tracks: List) -> List[Dict[str, Any]]:
        """
        Avalia cada track nas zonas:
        - Determina zonas ocupadas
        - Se estiver fora de 'stand_total' ou além de max_reliable_range_m, marca como fora de alcance
        - Detecta transições de entrada ('enter') e saída ('exit') do stand (Issue M1-01)
        - Se permanecer na zona de palco_facilitador por mais de 8s, sugere role 'facilitador'.
        """
        results = []
        self.pending_zone_events = []
        max_range = self.room_info.get("max_reliable_range_m", 4.5)
        current_active_ids = set()

        for trk in tracks:
            current_active_ids.add(trk.track_id)
            px, _, pz = trk.position
            in_range = (pz <= max_range) and not (np.isnan(px) or np.isnan(pz))
            matched_zones = self.get_zones_for_point(px, pz) if in_range else []

            # Verifica presença no stand
            is_in_stand = in_range and ("stand_total" in matched_zones or not any(z.get("id") == "stand_total" for z in self.zones))
            was_in_stand = self._track_in_stand.get(trk.track_id, False)

            if is_in_stand and not was_in_stand:
                # Transição: entrou no stand
                self._track_in_stand[trk.track_id] = True
                self.pending_zone_events.append({
                    "type": "enter",
                    "track_id": trk.track_id,
                    "zone": "stand_total",
                    "details": f"Track #{trk.track_id} entrou na zona stand_total"
                })
            elif not is_in_stand and was_in_stand:
                # Transição: saiu do stand
                self._track_in_stand[trk.track_id] = False
                self.pending_zone_events.append({
                    "type": "exit",
                    "track_id": trk.track_id,
                    "zone": "stand_total",
                    "details": f"Track #{trk.track_id} saiu da zona stand_total"
                })

            # Regra de facilitação em ponto fixo
            if "palco_facilitador" in matched_zones and trk.dwell_time_s >= 8.0:
                if trk.role != "facilitador":
                    trk.role = "facilitador"

            results.append({
                "track_id": trk.track_id,
                "in_reliable_range": in_range,
                "in_stand": is_in_stand,
                "zones": matched_zones,
                "role": trk.role,
                "presence_state": trk.presence_state
            })

        # Verifica tracks que deixaram de ser ativos (perda de rastreamento ou saída do campo de visão)
        tracked_ids = list(self._track_in_stand.keys())
        for tid in tracked_ids:
            if tid not in current_active_ids:
                if self._track_in_stand[tid]:
                    self.pending_zone_events.append({
                        "type": "exit",
                        "track_id": tid,
                        "zone": "stand_total",
                        "details": f"Track #{tid} saiu do stand (perda de rastreamento)"
                    })
                del self._track_in_stand[tid]

        return results
