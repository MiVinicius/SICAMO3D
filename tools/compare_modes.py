"""
Ferramenta de Comparação Modo A (Manual / Ground Truth) vs Modo B (Automático).
Processa sessões gravadas no Schema v3 e avalia:
- Concordância temporal da atribuição de portador (% frames acordados);
- Concordância temporal da máquina de estados;
- Detecção de eventos de handoff (TP, FP, FN, Precisão, Recall, F1 e latência média em ms);
- Matrizes de confusão por estado e por ator/portador;
- Exportação de relatório detalhado em Markdown para output/evaluations/m2_comparison_<timestamp>.md.
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

def _load_dataframes(source_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega (plush_state, events) a partir de uma pasta de sessão ou arquivo CSV."""
    p = Path(source_path)
    if p.is_dir():
        plush_file = p / "plush_state.csv"
        events_file = p / "events.csv"
        df_plush = pd.read_csv(plush_file) if plush_file.exists() else pd.DataFrame()
        df_events = pd.read_csv(events_file) if events_file.exists() else pd.DataFrame()
        return df_plush, df_events
    elif p.is_file():
        if p.name == "events.csv":
            return pd.DataFrame(), pd.read_csv(p)
        else:
            return pd.read_csv(p), pd.DataFrame()
    return pd.DataFrame(), pd.DataFrame()

def compare_sessions(
    mode_a_path: str,
    mode_b_path: str,
    tolerance_ms: float = 1000.0,
    output_report_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compara uma sessão Modo A (Manual/Ground Truth) com uma sessão Modo B (Automático).
    """
    df_plush_a, df_events_a = _load_dataframes(mode_a_path)
    df_plush_b, df_events_b = _load_dataframes(mode_b_path)

    if df_plush_a.empty and df_plush_b.empty:
        raise ValueError(f"Não foram encontrados dados de plush_state válidos em {mode_a_path} ou {mode_b_path}")

    # 1. Alinhamento temporal frame a frame
    aligned = pd.DataFrame()
    if not df_plush_a.empty and not df_plush_b.empty:
        # Tenta casar por t_capture_ms se estiverem próximas, senão por frame_idx
        if "t_capture_ms" in df_plush_a.columns and "t_capture_ms" in df_plush_b.columns:
            aligned = pd.merge_asof(
                df_plush_b.sort_values("t_capture_ms"),
                df_plush_a.sort_values("t_capture_ms"),
                on="t_capture_ms",
                direction="nearest",
                tolerance=100, # até 100ms de desvio de timestamp
                suffixes=("_b", "_a")
            )
        elif "frame_idx" in df_plush_a.columns and "frame_idx" in df_plush_b.columns:
            aligned = pd.merge(
                df_plush_b,
                df_plush_a,
                on="frame_idx",
                suffixes=("_b", "_a")
            )

    # Normalização de holder_id: NaN ou < 0 vira "Nenhum"
    if not aligned.empty:
        holder_a = aligned.get("holder_id_a", pd.Series([None] * len(aligned)))
        holder_b = aligned.get("holder_id_b", pd.Series([None] * len(aligned)))
        state_a = aligned.get("state_a", pd.Series(["INDETERMINADO"] * len(aligned))).fillna("INDETERMINADO")
        state_b = aligned.get("state_b", pd.Series(["INDETERMINADO"] * len(aligned))).fillna("INDETERMINADO")

        norm_holder_a = holder_a.apply(lambda x: int(x) if pd.notna(x) and x > 0 else -1)
        norm_holder_b = holder_b.apply(lambda x: int(x) if pd.notna(x) and x > 0 else -1)

        total_frames = len(aligned)
        holder_match = (norm_holder_a == norm_holder_b).sum()
        holder_acc = float(holder_match / total_frames) if total_frames > 0 else 0.0

        state_match = (state_a == state_b).sum()
        state_acc = float(state_match / total_frames) if total_frames > 0 else 0.0

        # Matriz de Confusão de Estados
        all_states = sorted(list(set(state_a.unique()).union(set(state_b.unique()))))
        state_cm = pd.crosstab(
            pd.Categorical(state_a, categories=all_states),
            pd.Categorical(state_b, categories=all_states),
            rownames=["Ground Truth (Modo A)"],
            colnames=["Predito (Modo B)"],
            dropna=False
        )

        # Matriz de Confusão de Atores/Portadores
        all_actors = sorted(list(set(norm_holder_a.unique()).union(set(norm_holder_b.unique()))))
        actor_labels = [f"Ator {a}" if a > 0 else "Sem Portador" for a in all_actors]
        actor_cm = pd.crosstab(
            pd.Categorical(norm_holder_a, categories=all_actors),
            pd.Categorical(norm_holder_b, categories=all_actors),
            rownames=["Ground Truth (Modo A)"],
            colnames=["Predito (Modo B)"],
            dropna=False
        )
        actor_cm.index = actor_labels
        actor_cm.columns = actor_labels
    else:
        total_frames = 0
        holder_acc = 0.0
        state_acc = 0.0
        state_cm = pd.DataFrame()
        actor_cm = pd.DataFrame()

    # 2. Avaliação de Eventos de Handoff
    handoffs_a = df_events_a[df_events_a["event_type"] == "handoff"].copy() if not df_events_a.empty and "event_type" in df_events_a.columns else pd.DataFrame()
    handoffs_b = df_events_b[df_events_b["event_type"] == "handoff"].copy() if not df_events_b.empty and "event_type" in df_events_b.columns else pd.DataFrame()

    n_gt = len(handoffs_a)
    n_det = len(handoffs_b)

    latencies_ms: List[float] = []
    matched_gt = 0
    matched_det = set()

    if not handoffs_a.empty and not handoffs_b.empty:
        for _, gt_row in handoffs_a.iterrows():
            gt_t = gt_row["t_capture_ms"]
            # Candidatos na janela de tolerância
            candidates = handoffs_b.loc[~handoffs_b.index.isin(matched_det)]
            matches = candidates[
                (candidates["t_capture_ms"] >= gt_t - tolerance_ms) &
                (candidates["t_capture_ms"] <= gt_t + tolerance_ms)
            ]
            if not matches.empty:
                # Se houver coincidência de doador/receptor, prioriza
                best_idx = None
                if "donor_id" in gt_row and "donor_id" in matches.columns:
                    id_match = matches[
                        (matches["donor_id"] == gt_row["donor_id"]) &
                        (matches["receiver_id"] == gt_row["receiver_id"])
                    ]
                    if not id_match.empty:
                        best_idx = (id_match["t_capture_ms"] - gt_t).abs().idxmin()
                if best_idx is None:
                    best_idx = (matches["t_capture_ms"] - gt_t).abs().idxmin()

                matched_det.add(best_idx)
                matched_gt += 1
                det_t = handoffs_b.loc[best_idx, "t_capture_ms"]
                latencies_ms.append(float(det_t - gt_t))

    tp = matched_gt
    fp = n_det - tp
    fn = n_gt - tp

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if n_gt == 0 and n_det == 0 else 0.0)
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else (1.0 if n_gt == 0 and n_det == 0 else 0.0)
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_latency_ms = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    std_latency_ms = float(np.std(latencies_ms)) if latencies_ms else 0.0

    # 3. Consolidação dos Resultados
    results = {
        "timestamp": datetime.now().isoformat(),
        "mode_a_path": str(mode_a_path),
        "mode_b_path": str(mode_b_path),
        "total_aligned_frames": total_frames,
        "holder_accuracy": round(holder_acc, 4),
        "state_accuracy": round(state_acc, 4),
        "handoff_metrics": {
            "ground_truth_count": n_gt,
            "detected_count": n_det,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "mean_latency_ms": round(mean_latency_ms, 2),
            "std_latency_ms": round(std_latency_ms, 2)
        },
        "state_confusion_matrix": state_cm.to_dict() if not state_cm.empty else {},
        "actor_confusion_matrix": actor_cm.to_dict() if not actor_cm.empty else {}
    }

    # 4. Geração do Relatório Markdown
    if output_report_path is None:
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = Path("output/evaluations")
        eval_dir.mkdir(parents=True, exist_ok=True)
        output_report_path = str(eval_dir / f"m2_comparison_{ts_str}.md")

    md_content = _build_markdown_report(results, state_cm, actor_cm)
    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    results["report_path"] = output_report_path
    return results

def _df_to_markdown(df: pd.DataFrame) -> str:
    """Converte um DataFrame para tabela Markdown sem depender da biblioteca tabulate."""
    if df.empty:
        return ""
    headers = [str(df.index.name or "")] + [str(col) for col in df.columns]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for idx, row in df.iterrows():
        row_vals = [str(idx)] + [str(val) for val in row]
        lines.append("| " + " | ".join(row_vals) + " |")
    return "\n".join(lines)

def _build_markdown_report(results: Dict[str, Any], state_cm: pd.DataFrame, actor_cm: pd.DataFrame) -> str:
    h_met = results["handoff_metrics"]
    ts = results["timestamp"]
    acc_h = results["holder_accuracy"] * 100.0
    acc_s = results["state_accuracy"] * 100.0

    md = []
    md.append(f"# Relatório de Comparação: Modo A (Manual) vs Modo B (Automático)\n")
    md.append(f"- **Data da Avaliação:** {ts}")
    md.append(f"- **Sessão Modo A (Gabarito / Operador):** `{results['mode_a_path']}`")
    md.append(f"- **Sessão Modo B (Inferência Automática):** `{results['mode_b_path']}`")
    md.append(f"- **Total de Quadros Alinhados:** {results['total_aligned_frames']} frames (~{results['total_aligned_frames']/30.0:.1f} s)\n")

    md.append("## 1. Resumo Executivo de Concordância\n")
    md.append("| Métrica | Valor Obtido | Status M2 |")
    md.append("| :--- | :--- | :--- |")
    md.append(f"| **Acurácia de Portador (Frame a Frame)** | **{acc_h:.2f}%** | {'✅ APROVADO' if acc_h >= 80.0 else '⚠️ ATENÇÃO'} |")
    md.append(f"| **Concordância da Máquina de Estados** | **{acc_s:.2f}%** | {'✅ APROVADO' if acc_s >= 80.0 else '⚠️ ATENÇÃO'} |")
    md.append(f"| **Precisão de Handoffs (Passagens)** | **{h_met['precision']*100:.1f}%** | {'✅ APROVADO' if h_met['precision'] >= 0.70 else '⚠️ ATENÇÃO'} |")
    md.append(f"| **Recall de Handoffs (Passagens)** | **{h_met['recall']*100:.1f}%** | {'✅ APROVADO' if h_met['recall'] >= 0.70 else '⚠️ ATENÇÃO'} |")
    md.append(f"| **F1-Score de Handoffs** | **{h_met['f1_score']:.3f}** | {'✅ APROVADO' if h_met['f1_score'] >= 0.70 else '⚠️ ATENÇÃO'} |")
    md.append(f"| **Latência Média de Detecção de Handoff** | **{h_met['mean_latency_ms']:.1f} ms** (±{h_met['std_latency_ms']:.1f} ms) | {'✅ DENTRO DA JANELA' if abs(h_met['mean_latency_ms']) <= 1000 else '⚠️ FORA'} |\n")

    md.append("## 2. Detecção de Passagens de Posse (Handoffs)\n")
    md.append(f"- **Passagens Gabarito (Modo A):** {h_met['ground_truth_count']}")
    md.append(f"- **Passagens Detectadas (Modo B):** {h_met['detected_count']}")
    md.append(f"- **Verdadeiros Positivos (TP):** {h_met['true_positives']}")
    md.append(f"- **Falsos Positivos (FP):** {h_met['false_positives']}")
    md.append(f"- **Falsos Negativos (FN):** {h_met['false_negatives']}\n")

    md.append("## 3. Matriz de Confusão por Estado da Pelúcia\n")
    if not state_cm.empty:
        md.append(_df_to_markdown(state_cm))
    else:
        md.append("_Nenhum quadro disponível para matriz de estados._")
    md.append("\n")

    md.append("## 4. Matriz de Confusão por Ator / Portador\n")
    if not actor_cm.empty:
        md.append(_df_to_markdown(actor_cm))
    else:
        md.append("_Nenhum quadro disponível para matriz de atores._")
    md.append("\n")

    md.append("## 5. Parecer e Recomendações\n")
    if acc_h >= 85.0 and h_met['f1_score'] >= 0.75:
        md.append("O sistema atende plenamente aos critérios de homologação do **Milestone M2**.")
        md.append("A transição entre Modo A e Modo B apresenta estabilidade com baixa ocorrência de falsos positivos.")
    else:
        md.append("O sistema apresenta divergências pontuais que demandam atenção:")
        if acc_h < 85.0:
            md.append("- Acurácia de portador abaixo de 85%: verificar calibração de `wrist_thresh_m` e `torso_thresh_m`.")
        if h_met['precision'] < 0.70:
            md.append("- Precisão de handoff reduzida: calibrar `min_score_margin` e `handoff_window_s` contra trocas ruidosas.")
        if h_met['recall'] < 0.70:
            md.append("- Recall de handoff reduzido: verificar sensibilidade em distâncias superiores a 3.5m.")

    return "\n".join(md)

def main():
    parser = argparse.ArgumentParser(description="Comparador de Sessões Modo A vs Modo B (M2-05)")
    parser.add_argument("--mode-a", type=str, required=False, help="Pasta de sessão ou CSV do Modo A (Manual/Ground Truth)")
    parser.add_argument("--mode-b", type=str, required=False, help="Pasta de sessão ou CSV do Modo B (Automático)")
    parser.add_argument("--session", type=str, required=False, help="Pasta de sessão única (se contiver Modo B e gabarito)")
    parser.add_argument("--ground-truth", "--gt", type=str, required=False, help="Arquivo de ground truth separado")
    parser.add_argument("--tolerance-ms", type=float, default=1000.0, help="Tolerância temporal em ms para parear handoffs")
    parser.add_argument("--output", type=str, default=None, help="Caminho do arquivo markdown de saída")

    args = parser.parse_args()

    mode_a = args.mode_a
    mode_b = args.mode_b

    if not mode_a and args.ground_truth:
        mode_a = args.ground_truth
    if not mode_b and args.session:
        mode_b = args.session

    if not mode_a or not mode_b:
        print("[ERRO] É necessário especificar --mode-a e --mode-b (ou --session e --ground-truth).")
        parser.print_help()
        sys.exit(1)

    print("=== EXECUTANDO COMPARAÇÃO MODO A vs MODO B ===")
    print(f"Modo A (Referência): {mode_a}")
    print(f"Modo B (Avaliado):   {mode_b}\n")

    res = compare_sessions(
        mode_a_path=mode_a,
        mode_b_path=mode_b,
        tolerance_ms=args.tolerance_ms,
        output_report_path=args.output
    )

    print(f"\n[OK] Avaliação concluída!")
    print(f"     Acurácia de Portador: {res['holder_accuracy']*100:.2f}%")
    print(f"     Concordância de Estado: {res['state_accuracy']*100:.2f}%")
    print(f"     Handoff F1-Score: {res['handoff_metrics']['f1_score']:.3f}")
    print(f"     Relatório gerado em: {res['report_path']}")

if __name__ == "__main__":
    main()
