"""
Módulo de Métricas Científicas de Circulação da Pelúcia e Atenção Social (PlushMetricsAnalyzer).
Calcula:
- Tempo de posse por participante, normalizado pelo tempo de presença;
- Matriz de transição e grafo doador -> receptor;
- Índice de Gini de concentração da posse;
- Razão de circulação (portadores distintos / total de participantes);
- Atenção angular da plateia direcionada ao portador;
- Segmentação completa por cena teatral (scene_id).
"""
import math
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

class PlushMetricsAnalyzer:
    def __init__(self):
        # Estruturas por cena: scene_id -> dados
        # Cada cena mantém:
        # - dwell_times: Dict[track_id, float] (segundos presente no stand)
        # - possession_times: Dict[track_id, float] (segundos segurando a pelúcia)
        # - handoffs: List[Dict] (eventos de handoff ocorridos na cena)
        # - attention_samples: List[Dict] (amostras de atenção da plateia ao portador)
        self.scenes_data: Dict[str, Dict[str, Any]] = {}
        self.active_scene_id: str = "Cena 1"
        self._last_update_time: Optional[float] = None

    def set_scene(self, scene_id: str):
        """Altera a cena atual."""
        self.active_scene_id = scene_id
        self._ensure_scene_exists(scene_id)

    def _ensure_scene_exists(self, scene_id: str):
        if scene_id not in self.scenes_data:
            self.scenes_data[scene_id] = {
                "dwell_times": {},
                "possession_times": {},
                "handoffs": [],
                "attention_samples": []
            }

    def update(self, 
               tracks: List, 
               holder_result: Dict[str, Any], 
               scene_id: Optional[str] = None, 
               timestamp_s: Optional[float] = None):
        """
        Atualiza acumuladores de tempo e métricas para o frame atual.
        """
        if scene_id is not None:
            self.active_scene_id = scene_id
        self._ensure_scene_exists(self.active_scene_id)
        sc_data = self.scenes_data[self.active_scene_id]

        cur_t = timestamp_s if timestamp_s is not None else 0.0
        dt = (cur_t - self._last_update_time) if (self._last_update_time is not None and cur_t > self._last_update_time) else (1.0 / 30.0)
        dt = min(dt, 0.20) # Limita dt para não distorcer em pausas
        self._last_update_time = cur_t

        # 1. Acumula tempo de presença para todos os participantes confirmados
        for trk in tracks:
            t_id = trk.track_id
            sc_data["dwell_times"][t_id] = sc_data["dwell_times"].get(t_id, 0.0) + dt

        # 2. Acumula tempo de posse do portador atual
        holder_id = holder_result.get("holder_id")
        state = holder_result.get("state")
        if holder_id is not None and state == "COM_PORTADOR":
            sc_data["possession_times"][holder_id] = sc_data["possession_times"].get(holder_id, 0.0) + dt

        # 3. Registra eventos de handoff pendentes
        for ev in holder_result.get("events", []):
            if ev.get("type") == "handoff":
                ev_copy = dict(ev)
                ev_copy["scene_id"] = self.active_scene_id
                sc_data["handoffs"].append(ev_copy)

        # 4. Atenção social da plateia em direção ao portador
        if holder_id is not None and state == "COM_PORTADOR":
            holder_trk = next((t for t in tracks if t.track_id == holder_id), None)
            if holder_trk:
                hx, _, hz = holder_trk.position
                attentive_count_35 = 0
                attentive_count_45 = 0
                attentive_count_60 = 0
                total_audience = 0

                for trk in tracks:
                    if trk.track_id == holder_id:
                        continue
                    # Apenas participantes com papel 'participante' ou 'plateia'
                    if trk.role == "facilitador":
                        continue

                    total_audience += 1
                    px, _, pz = trk.position
                    # Vetor participante -> portador
                    dx = hx - px
                    dz = hz - pz
                    angle_to_holder_deg = math.degrees(math.atan2(dx, dz))

                    # Heading corporal do participante: calculado a partir dos ombros ou vetor de velocidade
                    heading_deg = self._estimate_person_heading(trk)
                    if heading_deg is not None:
                        angular_diff = abs((heading_deg - angle_to_holder_deg + 180) % 360 - 180)
                        if angular_diff <= 35.0: attentive_count_35 += 1
                        if angular_diff <= 45.0: attentive_count_45 += 1
                        if angular_diff <= 60.0: attentive_count_60 += 1

                if total_audience > 0:
                    sc_data["attention_samples"].append({
                        "timestamp_s": cur_t,
                        "ratio_35": attentive_count_35 / total_audience,
                        "ratio_45": attentive_count_45 / total_audience,
                        "ratio_60": attentive_count_60 / total_audience,
                        "audience_count": total_audience
                    })

    def _estimate_person_heading(self, trk) -> Optional[float]:
        """Estima o ângulo de orientação corporal (heading em graus) a partir dos ombros no plano X-Z."""
        if trk.last_keypoints_3d is not None:
            k3d = trk.last_keypoints_3d
            # Ombros: 5 (esq), 6 (dir)
            if k3d[5, 3] > 0.20 and k3d[6, 3] > 0.20:
                sx_l, sz_l = k3d[5, 0], k3d[5, 2]
                sx_r, sz_r = k3d[6, 0], k3d[6, 2]
                if not (np.isnan(sx_l) or np.isnan(sx_r) or np.isnan(sz_l) or np.isnan(sz_r)):
                    # Vetor ombro direito -> ombro esquerdo
                    v_shoulders = np.array([sx_l - sx_r, sz_l - sz_r])
                    # O peito / frente aponta ortogonal a v_shoulders (girado 90 graus)
                    v_facing = np.array([-v_shoulders[1], v_shoulders[0]])
                    return float(math.degrees(math.atan2(v_facing[0], v_facing[1])))

        # Fallback: direção do vetor velocidade se estiver andando
        vx, _, vz = trk.velocity
        if math.hypot(vx, vz) > 0.20:
            return float(math.degrees(math.atan2(vx, vz)))
        return None

    def compute_gini(self, values: List[float]) -> float:
        """
        Calcula o Coeficiente de Gini para medir concentração de posse.
        0.0 = perfeitamente distribuído (todos tiveram o mesmo tempo de posse).
        1.0 = totalmente concentrado em um único indivíduo.
        """
        arr = np.array(values, dtype=np.float64)
        arr = arr[arr >= 0]
        if len(arr) == 0 or np.sum(arr) == 0:
            return 0.0
        n = len(arr)
        if n == 1:
            return 1.0
        arr = np.sort(arr)
        index = np.arange(1, n + 1)
        return float((2.0 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr)))

    def get_scene_summary(self, scene_id: Optional[str] = None) -> Dict[str, Any]:
        """Gera resumo científico consolidado para uma cena (ou a cena ativa)."""
        target_scene = scene_id or self.active_scene_id
        self._ensure_scene_exists(target_scene)
        sc = self.scenes_data[target_scene]

        all_track_ids = set(sc["dwell_times"].keys()).union(set(sc["possession_times"].keys()))
        participants_data = []
        possession_list = []
        norm_possession_list = []

        for tid in all_track_ids:
            dwell = sc["dwell_times"].get(tid, 0.0)
            poss = sc["possession_times"].get(tid, 0.0)
            norm_poss = (poss / dwell) if dwell > 0.5 else 0.0
            possession_list.append(poss)
            norm_possession_list.append(norm_poss)
            participants_data.append({
                "track_id": tid,
                "dwell_time_s": round(dwell, 1),
                "possession_time_s": round(poss, 1),
                "norm_possession_ratio": round(norm_poss, 3)
            })

        # Matriz de transições doador -> receptor
        transition_matrix: Dict[int, Dict[int, int]] = {}
        for h in sc["handoffs"]:
            d_id = h.get("donor_id")
            r_id = h.get("receiver_id")
            if d_id is not None and r_id is not None:
                if d_id not in transition_matrix:
                    transition_matrix[d_id] = {}
                transition_matrix[d_id][r_id] = transition_matrix[d_id].get(r_id, 0) + 1

        # Portadores distintos
        distinct_holders = len([p for p in possession_list if p >= 1.0])
        total_participants = len(all_track_ids)
        circulation_ratio = (distinct_holders / total_participants) if total_participants > 0 else 0.0

        # Atenção média da plateia
        att_samples = sc["attention_samples"]
        avg_attention_35 = float(np.mean([s["ratio_35"] for s in att_samples])) if att_samples else 0.0
        avg_attention_45 = float(np.mean([s["ratio_45"] for s in att_samples])) if att_samples else 0.0

        return {
            "scene_id": target_scene,
            "total_participants": total_participants,
            "distinct_holders": distinct_holders,
            "circulation_ratio": round(circulation_ratio, 3),
            "gini_possession": round(self.compute_gini(possession_list), 3),
            "gini_normalized": round(self.compute_gini(norm_possession_list), 3),
            "handoff_count": len(sc["handoffs"]),
            "transition_matrix": transition_matrix,
            "avg_audience_attention_35deg": round(avg_attention_35, 3),
            "avg_audience_attention_45deg": round(avg_attention_45, 3),
            "participants": participants_data,
            "handoff_events": list(sc["handoffs"])
        }
