"""
Módulo de Inferência de Portador da Pelúcia (HolderInference).
Substitui o antigo toy_interaction.py e garante atribuição EXCLUSIVA de portador único,
máquina de estados finita (COM_PORTADOR, PASSAGEM, SEM_PORTADOR, INDETERMINADO),
histerese anti-oscilação, suporte ao abraço (distância ao tronco e co-movimento),
detecção de passagem de posse (handoff) e anotação de ambiguidade (id_ambiguous).
"""
import time
from enum import Enum
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from src.socioenative.proxemics import ProxemicsAnalyzer

class HolderState(str, Enum):
    COM_PORTADOR = "COM_PORTADOR"
    PASSAGEM = "PASSAGEM"
    SEM_PORTADOR = "SEM_PORTADOR"
    INDETERMINADO = "INDETERMINADO"

class HolderInference:
    def __init__(self, 
                 wrist_thresh_m: float = 0.35,
                 torso_thresh_m: float = 0.45,
                 handoff_window_s: float = 2.0,
                 min_margin: float = 0.12,
                 timeout_s: float = 1.5,
                 weights: Tuple[float, float, float, float] = (0.35, 0.35, 0.20, 0.10)):
        self.wrist_thresh_m = wrist_thresh_m
        self.torso_thresh_m = torso_thresh_m
        self.handoff_window_s = handoff_window_s
        self.min_margin = min_margin
        self.timeout_s = timeout_s
        self.weights = weights

        # Estado atual da máquina de estados
        self.current_state: HolderState = HolderState.INDETERMINADO
        self.current_holder_id: Optional[int] = None
        self.current_holder_conf: float = 0.0
        self.last_state_change_time: float = 0.0
        self.last_detection_time: float = 0.0

        # Contexto de transição / passagem (handoff)
        self.handoff_start_time: Optional[float] = None
        self.handoff_donor_id: Optional[int] = None
        self.handoff_receiver_id: Optional[int] = None

        # Histórico recente da posição da pelúcia para cálculo de co-movimento: deque de (x, z, t)
        self._plush_pos_history: List[Tuple[float, float, float]] = []

        # Override manual do operador (Modo A)
        self._manual_holder_id: Optional[int] = None
        self._manual_holder_until: float = 0.0

        # Eventos gerados no frame atual (handoffs, transições)
        self.pending_events: List[Dict[str, Any]] = []

    @property
    def state(self) -> str:
        return self.current_state.value if hasattr(self.current_state, "value") else str(self.current_state)

    @property
    def holder_id(self) -> Optional[int]:
        return self.current_holder_id

    def set_manual_holder(self, track_id: Optional[int], duration_s: float = 5.0, timestamp_s: Optional[float] = None):
        """Permite ao operador (Modo A) forçar a atribuição de portador."""
        cur_t = timestamp_s if timestamp_s is not None else time.time()
        self._manual_holder_id = track_id
        self._manual_holder_until = cur_t + duration_s
        if track_id is not None:
            self.current_state = HolderState.COM_PORTADOR
            self.current_holder_id = track_id
            self.current_holder_conf = 1.0

    def process(self, 
                tracks: List, 
                plush_candidates: List[Dict], 
                timestamp_s: Optional[float] = None) -> Dict[str, Any]:
        """
        Processa as evidências do frame e atualiza a máquina de estados da pelúcia.
        tracks: lista de objetos Track3D
        plush_candidates: lista de dicionários com chaves 'pos_3d', 'confidence', 'bbox', etc.
        """
        cur_t = timestamp_s if timestamp_s is not None else time.time()
        self.pending_events.clear()

        # 1. Verifica override manual ativo do operador
        if self._manual_holder_id is not None and cur_t <= self._manual_holder_until:
            active_track_ids = {t.track_id for t in tracks}
            if self._manual_holder_id in active_track_ids:
                return self._build_result(
                    state=HolderState.COM_PORTADOR,
                    holder_id=self._manual_holder_id,
                    confidence=1.0,
                    source="manual",
                    best_candidate=plush_candidates[0] if plush_candidates else None,
                    timestamp_s=cur_t
                )
            else:
                self._manual_holder_id = None # Portador manual saiu do stand

        # 2. Seleção do melhor candidato de pelúcia (se houver)
        best_cand = None
        if plush_candidates:
            # Ordena por confiança
            valid_cands = [c for c in plush_candidates if c.get('pos_3d') is not None and not np.isnan(c['pos_3d']).any()]
            if valid_cands:
                valid_cands.sort(key=lambda x: x.get('confidence', 0.0), reverse=True)
                best_cand = valid_cands[0]
                self.last_detection_time = cur_t

                # Atualiza histórico de posições da pelúcia para velocidade / co-movimento
                p_pos = best_cand['pos_3d']
                self._plush_pos_history.append((p_pos[0], p_pos[2], cur_t))
                # Mantém apenas posições dos últimos 1.5s
                self._plush_pos_history = [p for p in self._plush_pos_history if (cur_t - p[2]) <= 1.5]

        # 3. Se a pelúcia sumiu há mais tempo que o timeout
        time_since_detection = cur_t - self.last_detection_time
        if best_cand is None:
            if time_since_detection > self.timeout_s:
                self.current_state = HolderState.INDETERMINADO
                self.current_holder_id = None
                self.current_holder_conf = 0.0
                return self._build_result(self.current_state, None, 0.0, "auto", None, cur_t)
            else:
                # Oclusão momentânea (ex: pelúcia abraçada oculta por 1s): mantém temporariamente o portador anterior
                if self.current_holder_id is not None and self.current_state == HolderState.COM_PORTADOR:
                    # Verifica se o portador ainda existe na cena
                    if any(t.track_id == self.current_holder_id for t in tracks):
                        return self._build_result(HolderState.COM_PORTADOR, self.current_holder_id, 0.50, "auto_cached", None, cur_t)

        if best_cand is None:
            return self._build_result(self.current_state, self.current_holder_id, self.current_holder_conf, "auto", None, cur_t)

        # 4. Cálculo de scores multivariados para cada pessoa ativa
        p_pos = np.array(best_cand['pos_3d'])
        plush_vel = self._estimate_plush_velocity()

        person_scores: List[Tuple[int, float, Dict[str, float]]] = [] # (track_id, score, features)

        for trk in tracks:
            t_id = trk.track_id
            left_3d, right_3d = trk.hands_3d
            trk_pos = np.array(trk.position)

            # Distância punhos -> pelúcia
            d_left = float(np.linalg.norm(np.array(left_3d) - p_pos)) if left_3d is not None and not np.isnan(left_3d).any() else 999.0
            d_right = float(np.linalg.norm(np.array(right_3d) - p_pos)) if right_3d is not None and not np.isnan(right_3d).any() else 999.0
            min_wrist_dist = min(d_left, d_right)

            # Distância ao tronco (centro de massa 3D) - cobre o abraço
            torso_dist = float(np.linalg.norm(trk_pos - p_pos))

            # Features normalizadas
            f_wrist = float(np.exp(-min_wrist_dist / self.wrist_thresh_m)) if min_wrist_dist < 2.0 else 0.0
            f_torso = float(np.exp(-torso_dist / self.torso_thresh_m)) if torso_dist < 2.5 else 0.0

            # Co-movimento: correlação direcional e de magnitude de velocidade nos últimos quadros
            f_comove = 0.0
            trk_vel_ground = np.array([trk.velocity[0], trk.velocity[2]])
            trk_speed = float(np.linalg.norm(trk_vel_ground))
            plush_speed = float(np.linalg.norm(plush_vel))
            if trk_speed > 0.15 and plush_speed > 0.15:
                cos_sim = float(np.dot(trk_vel_ground, plush_vel) / (trk_speed * plush_speed))
                f_comove = max(0.0, cos_sim)

            # Bounding box 2D: se o candidato de pelúcia está dentro da bounding box da pessoa
            f_inside_bbox = 0.0
            p_bbox = best_cand.get('bbox')
            if p_bbox and trk.last_bbox:
                # Centro 2D da pelúcia
                pcx = (p_bbox[0] + p_bbox[2]) / 2.0
                pcy = (p_bbox[1] + p_bbox[3]) / 2.0
                bx1, by1, bx2, by2 = trk.last_bbox
                if (bx1 - 30 <= pcx <= bx2 + 30) and (by1 - 30 <= pcy <= by2 + 30):
                    f_inside_bbox = 1.0

            # Score combinado ponderado configurável:
            # Equilíbrio entre pegada com a mão (f_wrist), proximidade/abraço ao tronco (f_torso),
            # co-movimento cinemático (f_comove) e contenção em bounding box 2D (f_inside_bbox)
            w_w, w_t, w_c, w_b = self.weights
            base_score = w_w * f_wrist + w_t * f_torso + w_c * f_comove + w_b * f_inside_bbox

            # Bônus de abraço: se o artefato estiver colado ao tórax/tronco (< 35 cm)
            if torso_dist < 0.35:
                base_score += 0.12

            # Histerese: se este track já é o portador atual, recebe bônus de persistência
            if self.current_holder_id == t_id:
                base_score += 0.15


            person_scores.append((t_id, base_score, {
                "wrist_dist": min_wrist_dist,
                "torso_dist": torso_dist,
                "f_wrist": f_wrist,
                "f_torso": f_torso,
                "f_comove": f_comove
            }))

        # 5. Decisão de Posse Exclusiva e Transição de Estados
        if not person_scores:
            self.current_state = HolderState.SEM_PORTADOR
            self.current_holder_id = None
            self.current_holder_conf = 0.0
            return self._build_result(self.current_state, None, 0.0, "auto", best_cand, cur_t)

        # Ordena por score decrescente
        person_scores.sort(key=lambda x: x[1], reverse=True)
        best_id, best_score, best_feats = person_scores[0]
        second_score = person_scores[1][1] if len(person_scores) > 1 else 0.0
        score_margin = best_score - second_score

        # Se ninguém estiver suficientemente perto da pelúcia (nem punho < 0.8m nem tronco < 1.0m)
        if best_feats["wrist_dist"] > 0.80 and best_feats["torso_dist"] > 1.00:
            self.current_state = HolderState.SEM_PORTADOR
            self.current_holder_id = None
            self.current_holder_conf = 0.0
            return self._build_result(self.current_state, None, 0.0, "auto", best_cand, cur_t)

        # 6. Máquina de Estados: Transições de Posse
        old_state = self.current_state
        old_holder = self.current_holder_id

        # Caso A: A pelúcia estava SEM_PORTADOR ou INDETERMINADO e alguém a pegou
        if old_holder is None:
            if best_score >= 0.35 and score_margin >= self.min_margin:
                self.current_state = HolderState.COM_PORTADOR
                self.current_holder_id = best_id
                self.current_holder_conf = float(np.clip(best_score, 0.0, 1.0))
                self.last_state_change_time = cur_t

        # Caso B: Alguém já possuía a pelúcia e o 1º colocado agora é outra pessoa
        elif old_holder != best_id:
            # Iniciou disputa ou transferência de posse
            if self.current_state != HolderState.PASSAGEM:
                # Transita para PASSAGEM
                self.current_state = HolderState.PASSAGEM
                self.handoff_start_time = cur_t
                self.handoff_donor_id = old_holder
                self.handoff_receiver_id = best_id
                self.last_state_change_time = cur_t
            else:
                # Já estava em PASSAGEM: verifica se a transferência se consolidou
                elapsed_handoff = cur_t - (self.handoff_start_time or cur_t)
                if elapsed_handoff <= self.handoff_window_s:
                    if best_score >= 0.40 and score_margin >= self.min_margin:
                        # Passagem consolidada! Emite evento de handoff
                        self._emit_handoff_event(
                            donor_id=self.handoff_donor_id,
                            receiver_id=best_id,
                            duration_s=elapsed_handoff,
                            tracks=tracks,
                            timestamp_s=cur_t
                        )
                        self.current_state = HolderState.COM_PORTADOR
                        self.current_holder_id = best_id
                        self.current_holder_conf = float(np.clip(best_score, 0.0, 1.0))
                        self.handoff_start_time = None
                else:
                    # Janela de passagem estourou: decide pelo maior score
                    self.current_state = HolderState.COM_PORTADOR
                    self.current_holder_id = best_id
                    self.current_holder_conf = float(np.clip(best_score, 0.0, 1.0))
                    self.handoff_start_time = None

        else:
            # O portador continua o mesmo (mantém COM_PORTADOR)
            self.current_state = HolderState.COM_PORTADOR
            self.current_holder_id = best_id
            self.current_holder_conf = float(np.clip(best_score, 0.0, 1.0))

        # Atualiza o atributo trk.held_toy nos tracks correspondentes
        for trk in tracks:
            if trk.track_id == self.current_holder_id and self.current_state == HolderState.COM_PORTADOR:
                trk.held_toy = "pelucia"
            else:
                trk.held_toy = None

        return self._build_result(
            state=self.current_state,
            holder_id=self.current_holder_id,
            confidence=self.current_holder_conf,
            source="auto",
            best_candidate=best_cand,
            timestamp_s=cur_t,
            score_margin=score_margin
        )

    def _estimate_plush_velocity(self) -> np.ndarray:
        """Estima vetor velocidade 2D (Vx, Vz) no chão a partir do histórico de posições."""
        if len(self._plush_pos_history) < 2:
            return np.zeros((2,), dtype=np.float32)
        p_first = self._plush_pos_history[0]
        p_last = self._plush_pos_history[-1]
        dt = p_last[2] - p_first[2]
        if dt < 0.05:
            return np.zeros((2,), dtype=np.float32)
        vx = (p_last[0] - p_first[0]) / dt
        vz = (p_last[1] - p_first[1]) / dt
        return np.array([vx, vz], dtype=np.float32)

    def _emit_handoff_event(self, donor_id: Optional[int], receiver_id: int, duration_s: float, tracks: List, timestamp_s: float):
        """Gera evento científico de passagem de posse com análise de proximidade e ambiguidade."""
        donor_trk = next((t for t in tracks if t.track_id == donor_id), None)
        recv_trk = next((t for t in tracks if t.track_id == receiver_id), None)

        dist_interpersonal_m = 0.0
        id_ambiguous = False
        flags = []

        if donor_trk and recv_trk:
            dist_interpersonal_m = float(np.linalg.norm(np.array(donor_trk.position) - np.array(recv_trk.position)))
            
            # Flag id_ambiguous se um dos tracks nasceu recentemente (menos de 2s)
            if donor_trk.dwell_time_s < 2.0 or recv_trk.dwell_time_s < 2.0:
                id_ambiguous = True
                flags.append("recent_track_birth")

            # Se a distância interpessoal for muito pequena (< 30cm) no instante da troca
            if dist_interpersonal_m < 0.30:
                flags.append("close_body_contact")
        else:
            id_ambiguous = True
            flags.append("lost_donor_track")

        proxemic_zone = ProxemicsAnalyzer.get_zone(dist_interpersonal_m)
        flags.append(f"zone_{proxemic_zone}")

        event = {
            "type": "handoff",
            "timestamp_ms": int(timestamp_s * 1000),
            "donor_id": donor_id,
            "receiver_id": receiver_id,
            "duration_s": round(duration_s, 2),
            "distance_interpersonal_m": round(dist_interpersonal_m, 3),
            "proxemic_zone": proxemic_zone,
            "confidence": 0.85 if not id_ambiguous else 0.50,
            "id_ambiguous": id_ambiguous,
            "flags": ";".join(flags) if flags else "none",
            "details": f"zone:{proxemic_zone};dist:{dist_interpersonal_m:.3f}m"
        }
        self.pending_events.append(event)

    def _build_result(self, state: HolderState, holder_id: Optional[int], confidence: float, source: str, best_candidate: Optional[Dict], timestamp_s: float, score_margin: float = 0.0) -> Dict[str, Any]:
        p_x, p_z = None, None
        if best_candidate and best_candidate.get('pos_3d') is not None:
            p_pos = best_candidate['pos_3d']
            p_x = round(float(p_pos[0]), 3)
            p_z = round(float(p_pos[2]), 3)

        return {
            "state": state.value,
            "holder_id": holder_id,
            "confidence": round(confidence, 3),
            "score_margin": round(score_margin, 3),
            "source": source,
            "plush_pos_3d": best_candidate.get('pos_3d') if best_candidate else None,
            "plush_x": p_x,
            "plush_z": p_z,
            "timestamp_s": timestamp_s,
            "events": list(self.pending_events)
        }
