"""
Módulo de Avaliação Científica e Métricas de Validação (tools/evaluate.py).
Calcula:
- Acurácia de portador frame-a-frame (com tolerância de transição de +- 1.0s);
- Casamento estrito 1:1 de eventos de passagem (handoffs) com validação de doador e receptor;
- Precisão, Recall e F1 de handoffs;
- Taxa de passagens falsas por minuto (FP/min);
- Estatísticas de concentração de posse (Gini) e circulação entre os participantes;
- Métrica de trocas de identidade de rastreamento (ID switches por minuto) contra identidades reais.
"""
import os
import argparse
from typing import Dict, Any, List, Optional, Set, Tuple
import pandas as pd
import numpy as np

def calculate_id_switches(
    df_traj: pd.DataFrame, 
    df_gt_id: pd.DataFrame, 
    session_duration_min: float
) -> Dict[str, Any]:
    """
    Calcula a contagem de ID switches (trocas de identidade) por pessoa real e a taxa por minuto.
    df_gt_id pode conter:
      - ['t_capture_ms', 'real_person_id', 'track_id'] (mapeamento explícito)
      ou
      - ['t_capture_ms', 'real_person_id', 'pos_x', 'pos_z'] (mapeamento por proximidade espacial)
    """
    if df_gt_id.empty or "real_person_id" not in df_gt_id.columns:
        return {"total_id_switches": 0, "id_switches_per_min": 0.0, "switches_by_person": {}}

    merged = df_gt_id.copy()

    # Se o gabarito contém posições em vez de track_id direto, mapeia por proximidade espacial
    if "track_id" not in merged.columns and not df_traj.empty and "pos_x" in merged.columns and "pos_x" in df_traj.columns:
        assigned_tracks = []
        for _, gt_row in merged.iterrows():
            t_ms = gt_row["t_capture_ms"]
            gx, gz = float(gt_row["pos_x"]), float(gt_row["pos_z"])
            traj_frame = df_traj[
                (df_traj["t_capture_ms"] >= t_ms - 50) & 
                (df_traj["t_capture_ms"] <= t_ms + 50) & 
                (df_traj["track_id"] > 0)
            ]
            best_tid = None
            best_d = 0.80 # raio máximo de 80cm para associação espacial
            for _, trk_row in traj_frame.iterrows():
                tx, tz = float(trk_row["pos_x"]), float(trk_row["pos_z"])
                d = float(np.hypot(tx - gx, tz - gz))
                if d < best_d:
                    best_d = d
                    best_tid = int(trk_row["track_id"])
            assigned_tracks.append(best_tid)
        merged["track_id"] = assigned_tracks

    total_switches = 0
    switches_by_person = {}

    for person_id, group in merged.groupby("real_person_id"):
        sorted_grp = group.sort_values("t_capture_ms") if "t_capture_ms" in group.columns else group
        person_switches = 0
        last_track_id = None

        for _, r in sorted_grp.iterrows():
            tid = r.get("track_id")
            if pd.notna(tid) and int(tid) > 0:
                tid_int = int(tid)
                if last_track_id is not None and tid_int != last_track_id:
                    person_switches += 1
                last_track_id = tid_int

        switches_by_person[str(person_id)] = person_switches
        total_switches += person_switches

    rate_per_min = total_switches / session_duration_min if session_duration_min > 0 else 0.0

    return {
        "total_id_switches": total_switches,
        "id_switches_per_min": round(rate_per_min, 2),
        "switches_by_person": switches_by_person
    }

def evaluate_session(
    session_dir: str, 
    ground_truth_events_csv: Optional[str] = None, 
    ground_truth_frames_csv: Optional[str] = None,
    ground_truth_identities_csv: Optional[str] = None
) -> Dict[str, Any]:
    plush_path = os.path.join(session_dir, "plush_state.csv")
    events_path = os.path.join(session_dir, "events.csv")
    traj_path = os.path.join(session_dir, "trajectories.csv")

    if not os.path.exists(plush_path):
        print(f"Erro: {plush_path} não encontrado.")
        return {}

    df_plush = pd.read_csv(plush_path)
    df_events = pd.read_csv(events_path) if os.path.exists(events_path) else pd.DataFrame()
    df_traj = pd.read_csv(traj_path) if os.path.exists(traj_path) else pd.DataFrame()

    total_frames = len(df_plush)
    session_duration_s = total_frames / 30.0
    session_duration_min = max(0.01, session_duration_s / 60.0)

    # 1. Estatísticas de Posse e Gini
    holder_counts = df_plush[df_plush["holder_id"] > 0]["holder_id"].value_counts().to_dict()
    distinct_holders = len(holder_counts)

    # Identifica facilitadores para isolar do cálculo de Gini de participantes
    facilitator_ids = set()
    if not df_traj.empty and "role" in df_traj.columns:
        facilitator_ids = set(df_traj[df_traj["role"] == "facilitador"]["track_id"].unique())

    participant_holder_counts = {k: v for k, v in holder_counts.items() if k not in facilitator_ids and k > 0}
    
    def calc_gini(values: List[float]) -> float:
        if not values or sum(values) == 0:
            return 0.0
        arr = np.array(sorted(values), dtype=np.float64)
        n = len(arr)
        if n <= 1:
            return 0.0
        index = np.arange(1, n + 1)
        return float((2.0 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr)))

    gini_possession = calc_gini(list(participant_holder_counts.values()))
    
    # 2. Eventos de Handoff detectados
    handoffs_detected = df_events[df_events["event_type"] == "handoff"].copy() if not df_events.empty and "event_type" in df_events.columns else pd.DataFrame()
    n_handoffs = len(handoffs_detected)
    handoffs_per_min = n_handoffs / session_duration_min

    # Ambiguidade
    ambiguous_handoffs = 0
    if not handoffs_detected.empty and "flags" in handoffs_detected.columns:
        ambiguous_handoffs = len(handoffs_detected[handoffs_detected["flags"].str.contains("recent_track_birth|lost_donor", na=False)])

    print("=" * 60)
    print("RELATÓRIO DE AVALIAÇÃO CIENTÍFICA")
    print(f"Sessão: {session_dir}")
    print(f"Duração estimada: {session_duration_s:.1f} s ({session_duration_min:.2f} min)")
    print(f"Portadores distintos identificados: {distinct_holders}")
    print(f"Gini de posse entre participantes: {gini_possession:.3f}")
    print(f"Total de passagens (handoffs) detectadas: {n_handoffs} ({handoffs_per_min:.2f} passagens/min)")
    print(f"Passagens com flag de ambiguidade (id_ambiguous): {ambiguous_handoffs}")
    print("=" * 60)

    results = {
        "session_duration_s": round(session_duration_s, 2),
        "distinct_holders": distinct_holders,
        "gini_possession": round(gini_possession, 3),
        "n_handoffs_detected": n_handoffs,
        "handoffs_per_min": round(handoffs_per_min, 2),
        "ambiguous_handoffs": ambiguous_handoffs
    }

    # 3. Casamento 1:1 Estrito de Handoffs com Verificação de Doador e Receptor (Issue M4-03)
    if ground_truth_events_csv and os.path.exists(ground_truth_events_csv):
        df_gt = pd.read_csv(ground_truth_events_csv)
        gt_handoffs = df_gt[df_gt["event_type"] == "handoff"].copy()
        n_gt = len(gt_handoffs)

        candidate_pairs: List[Tuple[float, Any, Any]] = [] # (dt, gt_idx, det_idx)
        for gt_idx, gt_row in gt_handoffs.iterrows():
            gt_t = gt_row["t_capture_ms"]
            gt_donor = gt_row.get("donor_id")
            gt_recv = gt_row.get("receiver_id")
            has_gt_ids = pd.notna(gt_donor) and pd.notna(gt_recv)

            for det_idx, det_row in handoffs_detected.iterrows():
                det_t = det_row["t_capture_ms"]
                dt = abs(det_t - gt_t)
                if dt <= 1000: # Janela de tolerância temporal de +- 1.0s
                    # Se o gabarito especifica donor e receiver, ambos devem coincidir
                    if has_gt_ids and "donor_id" in det_row and pd.notna(det_row["donor_id"]):
                        if int(det_row["donor_id"]) != int(gt_donor) or int(det_row["receiver_id"]) != int(gt_recv):
                            continue
                    candidate_pairs.append((dt, gt_idx, det_idx))

        # Ordena candidatos pela menor discrepância temporal
        candidate_pairs.sort(key=lambda x: x[0])

        matched_gt: Set[Any] = set()
        matched_det: Set[Any] = set()
        for dt, gt_idx, det_idx in candidate_pairs:
            if gt_idx not in matched_gt and det_idx not in matched_det:
                matched_gt.add(gt_idx)
                matched_det.add(det_idx)

        tp = len(matched_gt)
        fp = len(handoffs_detected) - len(matched_det)
        fn = n_gt - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        fp_per_min = fp / session_duration_min

        print("\nCOMPARAÇÃO COM GABARITO (GROUND TRUTH EVENTOS - CASAMENTO 1:1):")
        print(f"  > Handoffs Gabarito: {n_gt}")
        print(f"  > Verdadeiros Positivos (TP): {tp}")
        print(f"  > Falsos Positivos (FP): {fp} ({fp_per_min:.2f} FP/min)")
        print(f"  > Falsos Negativos (FN): {fn}")
        print(f"  > Precisão: {precision:.3f} | Recall: {recall:.3f} | F1-Score: {f1:.3f}")

        results.update({
            "gt_handoffs": n_gt,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1, 3),
            "fp_per_min": round(fp_per_min, 2)
        })

    # 4. Avaliação frame-a-frame de portador se fornecido gabarito contínuo
    if ground_truth_frames_csv and os.path.exists(ground_truth_frames_csv):
        df_gt_f = pd.read_csv(ground_truth_frames_csv)
        merged = pd.merge(df_plush, df_gt_f, on="t_capture_ms", suffixes=("_det", "_gt"))
        if not merged.empty and "holder_id_gt" in merged.columns:
            correct = (merged["holder_id_det"] == merged["holder_id_gt"]).sum()
            total_f = len(merged)
            acc = correct / total_f
            print(f"\nCOMPARAÇÃO FRAME-A-FRAME:")
            print(f"  > Acurácia de portador: {acc * 100.0:.2f}% ({correct}/{total_f} frames)")
            results["frame_holder_accuracy"] = round(acc, 4)

    # 5. Avaliação de Trocas de Identidade / ID Switches (Issue M4-02)
    if ground_truth_identities_csv and os.path.exists(ground_truth_identities_csv):
        df_gt_id = pd.read_csv(ground_truth_identities_csv)
        id_sw_results = calculate_id_switches(df_traj, df_gt_id, session_duration_min)
        print("\nAVALIAÇÃO DE CONTINUIDADE DE RASTREAMENTO (ID SWITCHES):")
        print(f"  > Total de trocas de identidade (ID switches): {id_sw_results['total_id_switches']}")
        print(f"  > Taxa de trocas: {id_sw_results['id_switches_per_min']:.2f} switches/min")
        print(f"  > Detalhamento por pessoa: {id_sw_results['switches_by_person']}")
        results.update(id_sw_results)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliação Científica SICAMO3D")
    parser.add_argument("session_dir", type=str, help="Pasta da sessão a avaliar")
    parser.add_argument("--gt", type=str, default=None, help="Caminho do CSV de eventos de gabarito")
    parser.add_argument("--gt-frames", type=str, default=None, help="Caminho do CSV frame-a-frame de gabarito")
    parser.add_argument("--gt-identities", type=str, default=None, help="Caminho do CSV de identidades reais para cálculo de ID switches")
    args = parser.parse_args()
    evaluate_session(
        args.session_dir, 
        ground_truth_events_csv=args.gt, 
        ground_truth_frames_csv=args.gt_frames,
        ground_truth_identities_csv=args.gt_identities
    )
