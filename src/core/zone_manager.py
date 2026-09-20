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
        - Se permanecer na zona de palco_facilitador por mais de 8s, sugere role 'facilitador'.
        """
        results = []
        max_range = self.room_info.get("max_reliable_range_m", 4.5)

        for trk in tracks:
            px, _, pz = trk.position
            in_range = (pz <= max_range) and not (np.isnan(px) or np.isnan(pz))
            matched_zones = self.get_zones_for_point(px, pz) if in_range else []

            # Regra de facilitação em ponto fixo
            if "palco_facilitador" in matched_zones and trk.dwell_time_s >= 8.0:
                if trk.role != "facilitador":
                    trk.role = "facilitador"

            results.append({
                "track_id": trk.track_id,
                "in_reliable_range": in_range,
                "zones": matched_zones,
                "role": trk.role,
                "presence_state": trk.presence_state
            })
        return results
