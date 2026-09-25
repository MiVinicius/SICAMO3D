"""
Filtro de suavização temporal (EMA adaptativo com amortecimento de oclusão)
para articulações 2D e 3D do esqueleto. Elimina trepidação, jitter e linhas piscando.
"""
import numpy as np
from typing import Optional

class KeypointFilter:
    def __init__(self, alpha: float = 0.82, max_missed_frames: int = 1):
        self.alpha = alpha
        self.max_missed_frames = max_missed_frames
        
        self.prev_kpts_2d: Optional[np.ndarray] = None # (17, 3) [u, v, conf]
        self.prev_kpts_3d: Optional[np.ndarray] = None # (17, 4) [X, Y, Z, conf]
        self.missed_counts_2d = np.zeros(17, dtype=int)
        self.missed_counts_3d = np.zeros(17, dtype=int)

    def filter(self, new_kpts_2d: Optional[np.ndarray], new_kpts_3d: Optional[np.ndarray]):
        """
        Aplica suavização temporal com retenção inteligente de articulações em oclusão rápida.
        """
        filtered_2d = self._filter_2d(new_kpts_2d) if new_kpts_2d is not None else None
        filtered_3d = self._filter_3d(new_kpts_3d) if new_kpts_3d is not None else None
        return filtered_2d, filtered_3d

    def _filter_2d(self, new_kpts: np.ndarray) -> np.ndarray:
        if self.prev_kpts_2d is None:
            self.prev_kpts_2d = np.copy(new_kpts)
            return new_kpts

        result = np.copy(new_kpts)
        for i in range(17):
            conf = new_kpts[i, 2]
            if conf >= 0.30:
                # Interpolação ágil entre a medição anterior e a nova (sem arrastar sombras)
                result[i, 0] = self.alpha * new_kpts[i, 0] + (1.0 - self.alpha) * self.prev_kpts_2d[i, 0]
                result[i, 1] = self.alpha * new_kpts[i, 1] + (1.0 - self.alpha) * self.prev_kpts_2d[i, 1]
                result[i, 2] = max(conf, self.prev_kpts_2d[i, 2] * 0.90)
                self.missed_counts_2d[i] = 0
            else:
                # Oclusão: permite no máximo 1 frame com decaimento acentuado, evitando membros fantasmas
                if self.missed_counts_2d[i] < self.max_missed_frames:
                    result[i, 0] = self.prev_kpts_2d[i, 0]
                    result[i, 1] = self.prev_kpts_2d[i, 1]
                    result[i, 2] = self.prev_kpts_2d[i, 2] * 0.40
                    self.missed_counts_2d[i] += 1
                else:
                    result[i, 2] = 0.0

        self.prev_kpts_2d = np.copy(result)
        return result

    def _filter_3d(self, new_kpts: np.ndarray) -> np.ndarray:
        if self.prev_kpts_3d is None:
            self.prev_kpts_3d = np.copy(new_kpts)
            return new_kpts

        result = np.copy(new_kpts)
        for i in range(17):
            conf = new_kpts[i, 3]
            depth_valid = (new_kpts[i, 2] > 0.35)

            if conf >= 0.30 and depth_valid:
                result[i, 0] = self.alpha * new_kpts[i, 0] + (1.0 - self.alpha) * self.prev_kpts_3d[i, 0]
                result[i, 1] = self.alpha * new_kpts[i, 1] + (1.0 - self.alpha) * self.prev_kpts_3d[i, 1]
                result[i, 2] = self.alpha * new_kpts[i, 2] + (1.0 - self.alpha) * self.prev_kpts_3d[i, 2]
                result[i, 3] = conf
                self.missed_counts_3d[i] = 0
            else:
                if self.missed_counts_3d[i] < self.max_missed_frames:
                    result[i, :3] = self.prev_kpts_3d[i, :3]
                    result[i, 3] = self.prev_kpts_3d[i, 3] * 0.40
                    self.missed_counts_3d[i] += 1
                else:
                    result[i, 3] = 0.0

        self.prev_kpts_3d = np.copy(result)
        return result
