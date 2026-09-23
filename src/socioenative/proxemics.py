"""
Módulo de cálculo de Proxêmica contínua entre indivíduos.
Baseado na teoria de distâncias interpessoais de Edward T. Hall calibrada para crianças e adolescentes.
"""
from typing import List, Dict, Tuple
import numpy as np
from src.core.config import config

class ProxemicsAnalyzer:
    @staticmethod
    def get_zone(distance_m: float) -> str:
        intimate = getattr(config.proxemics, "intimate_threshold_m", getattr(config.proxemics, "intimate_max_m", 0.35))
        personal = getattr(config.proxemics, "personal_threshold_m", getattr(config.proxemics, "personal_max_m", 1.00))
        social = getattr(config.proxemics, "social_threshold_m", getattr(config.proxemics, "social_max_m", 2.50))
        if distance_m < intimate:
            return "intima"
        elif distance_m < personal:
            return "pessoal"
        elif distance_m < social:
            return "social"
        else:
            return "publica"

    @classmethod
    def calculate_pairwise(cls, active_tracks: List) -> List[Dict]:
        """
        Calcula as distâncias euclidianas 3D entre todos os pares de pessoas ativas.
        """
        results = []
        n = len(active_tracks)
        if n < 2:
            return results

        for i in range(n):
            for j in range(i + 1, n):
                t1 = active_tracks[i]
                t2 = active_tracks[j]
                
                p1 = np.array(t1.position)
                p2 = np.array(t2.position)
                
                dist = float(np.linalg.norm(p1 - p2))
                zone = cls.get_zone(dist)

                # Velocidade relativa de aproximação / afastamento
                v1 = np.array(t1.velocity)
                v2 = np.array(t2.velocity)
                vel_rel = float(np.linalg.norm(v1 - v2))

                results.append({
                    "id_pair": (t1.track_id, t2.track_id),
                    "distance_m": round(dist, 3),
                    "zone": zone,
                    "relative_speed_mps": round(vel_rel, 3),
                    "pos1": t1.position,
                    "pos2": t2.position
                })
        return results
