"""
Módulo de Avaliação Científica e Métricas de Validação (tools/evaluate.py).
Calcula:
- Acurácia de portador frame-a-frame (com tolerância de transição de +- 1.0s);
- Precisão e Recall de eventos de passagem (handoffs);
- Taxa de passagens falsas por minuto (FP/min);
- Estatísticas de concentração de posse (Gini) e circulação entre os participantes.
"""
import os
import argparse
import pandas as pd
import numpy as np
from typing import Dict, Any, List

def evaluate_session(session_dir: str, ground_truth_events_csv: str = None, ground_truth_frames_csv: str = None) -> Dict[str, Any]:
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
    handoffs_detected = df_events[df_events["event_type"] == "handoff"].copy() if not df_events.empty else pd.DataFrame()
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

    # 3. Comparação com Gabarito (Ground Truth) se fornecido
    results = {
        "session_duration_s": round(session_duration_s, 2),
        "distinct_holders": distinct_holders,
        "gini_possession": round(gini_possession, 3),
        "n_handoffs_detected": n_handoffs,
        "handoffs_per_min": round(handoffs_per_min, 2),
        "ambiguous_handoffs": ambiguous_handoffs
    }

    if ground_truth_events_csv and os.path.exists(ground_truth_events_csv):
        df_gt = pd.read_csv(ground_truth_events_csv)
        gt_handoffs = df_gt[df_gt["event_type"] == "handoff"]
        n_gt = len(gt_handoffs)
        
        # Casamento 1:1 de handoffs com janela de tolerância de +- 1.0s (1000 ms)
        matched_gt = 0
        matched_det = set()

        for _, gt_row in gt_handoffs.iterrows():
            gt_t = gt_row["t_capture_ms"]
            # Exclui detecções já consumidas (casamento 1:1)
            candidates = handoffs_detected.loc[~handoffs_detected.index.isin(matched_det)]
            match = candidates[
                (candidates["t_capture_ms"] >= gt_t - 1000) &
                (candidates["t_capture_ms"] <= gt_t + 1000)
            ]
            if not match.empty:
                # Se GT especifica IDs, prioriza casamento com IDs coincidentes
                best_idx = None
                if "donor_id" in gt_row and pd.notna(gt_row["donor_id"]) and "donor_id" in match.columns:
                    id_match = match[(match["donor_id"] == gt_row["donor_id"]) & (match["receiver_id"] == gt_row["receiver_id"])]
                    if not id_match.empty:
                        best_idx = (id_match["t_capture_ms"] - gt_t).abs().idxmin()

                if best_idx is None:
                    best_idx = (match["t_capture_ms"] - gt_t).abs().idxmin()

                matched_gt += 1
                matched_det.add(best_idx)

        tp = matched_gt
        fp = len(handoffs_detected) - len(matched_det)
        fn = n_gt - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        fp_per_min = fp / session_duration_min

        print("\nCOMPARAÇÃO COM GABARITO (GROUND TRUTH EVENTOS):")
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

    # 4. Avaliação frame-a-frame se fornecido gabarito contínuo
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

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliação Científica SICAMO3D")
    parser.add_argument("session_dir", type=str, help="Pasta da sessão a avaliar")
    parser.add_argument("--gt", type=str, default=None, help="Caminho do CSV de eventos de gabarito")
    parser.add_argument("--gt-frames", type=str, default=None, help="Caminho do CSV frame-a-frame de gabarito")
    args = parser.parse_args()
    evaluate_session(args.session_dir, ground_truth_events_csv=args.gt, ground_truth_frames_csv=args.gt_frames)
