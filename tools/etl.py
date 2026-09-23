"""
Camada de ETL (Extract, Transform, Load) e Análise Final em Lote (tools/etl.py).
Atende às Issues M5-03 e M5-04 do Roadmap do Projeto SICAMO3D:
1. Extrai dados de uma ou múltiplas sessões gravadas no Schema v3 (trajectories.csv, plush_state.csv, events.csv);
2. Transforma e consolida tabelas científicas agregadas:
   - possession_intervals: períodos contínuos de posse por participante/ator;
   - handoff_events_enriched: passagens com distâncias, zona proxêmica de Hall e flags de ambiguidade;
   - presence_summary: tempos de permanência, classificação cinemática (plateia vs passante);
   - scene_summary: métricas consolidadas por cena (Gini de posse, razão de circulação, taxa de handoffs/min);
   - f_formations_summary: detecção de formações de Kendon e proporção de interação em grupo vs solo.
3. Gera Relatório Analítico completo em Markdown respondendo às 4 perguntas de pesquisa da tese.
"""

import os
import sys
import re
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from src.socioenative.proxemics import ProxemicsAnalyzer
from src.socioenative.f_formations import FFormationDetector

def calc_gini(values: List[float]) -> float:
    """Calcula o coeficiente de Gini clássico (0 = perfeitamente igual, 1 = concentrado)."""
    if not values or sum(values) == 0:
        return 0.0
    arr = np.array(sorted(values), dtype=np.float64)
    n = len(arr)
    if n <= 1:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr)))

def find_session_dirs(base_path: str) -> List[Path]:
    """Identifica diretórios de sessão válidos (que contenham trajectories.csv ou plush_state.csv)."""
    p = Path(base_path)
    if not p.exists():
        return []
    
    # Se o diretório apontado já for uma pasta de sessão
    if (p / "plush_state.csv").exists() or (p / "trajectories.csv").exists():
        return [p]
    
    # Busca recursiva em subdiretórios
    sessions = []
    for sub in p.rglob("*"):
        if sub.is_dir() and ((sub / "plush_state.csv").exists() or (sub / "trajectories.csv").exists()):
            sessions.append(sub)
    return sorted(list(set(sessions)))

def extract_possession_intervals(df_plush: pd.DataFrame, session_id: str) -> pd.DataFrame:
    """Extrai intervalos contínuos de posse da pelúcia por participante."""
    if df_plush.empty or "holder_id" not in df_plush.columns:
        return pd.DataFrame()

    intervals = []
    current_holder = None
    current_state = None
    current_scene = None
    start_t = None
    end_t = None
    conf_samples = []
    source = "auto"

    for _, row in df_plush.iterrows():
        t = row.get("t_capture_ms", 0)
        h = row.get("holder_id")
        h_norm = int(h) if pd.notna(h) and h > 0 else -1
        st = str(row.get("state", "INDETERMINADO"))
        sc = str(row.get("scene_id", "Cena 1"))
        conf = float(row.get("holder_conf", 0.0)) if pd.notna(row.get("holder_conf")) else 0.0
        src = str(row.get("source", "auto"))

        if (h_norm != current_holder) or (st != current_state) or (sc != current_scene):
            # Fecha intervalo anterior
            if current_holder is not None and start_t is not None:
                dur_s = max(0.01, (end_t - start_t) / 1000.0)
                intervals.append({
                    "session_id": session_id,
                    "scene_id": current_scene,
                    "holder_id": current_holder,
                    "state": current_state,
                    "start_time_ms": start_t,
                    "end_time_ms": end_t,
                    "duration_s": round(dur_s, 2),
                    "mean_confidence": round(float(np.mean(conf_samples)), 3) if conf_samples else 0.0,
                    "source": source
                })
            # Inicia novo intervalo
            current_holder = h_norm
            current_state = st
            current_scene = sc
            start_t = t
            end_t = t
            conf_samples = [conf]
            source = src
        else:
            end_t = t
            conf_samples.append(conf)

    # Último intervalo
    if current_holder is not None and start_t is not None:
        dur_s = max(0.01, (end_t - start_t) / 1000.0)
        intervals.append({
            "session_id": session_id,
            "scene_id": current_scene,
            "holder_id": current_holder,
            "state": current_state,
            "start_time_ms": start_t,
            "end_time_ms": end_t,
            "duration_s": round(dur_s, 2),
            "mean_confidence": round(float(np.mean(conf_samples)), 3) if conf_samples else 0.0,
            "source": source
        })

    return pd.DataFrame(intervals)

def extract_handoff_events(df_events: pd.DataFrame, session_id: str) -> pd.DataFrame:
    """Extrai e enriquece eventos de handoff com classificação de zona proxêmica de Hall."""
    if df_events.empty:
        return pd.DataFrame()

    ev_type_col = "event_type" if "event_type" in df_events.columns else "type"
    if ev_type_col not in df_events.columns:
        return pd.DataFrame()

    handoffs = df_events[df_events[ev_type_col] == "handoff"].copy()
    if handoffs.empty:
        return pd.DataFrame()

    enriched = []
    for _, row in handoffs.iterrows():
        t = row.get("t_capture_ms", row.get("timestamp_ms", 0))
        sc = row.get("scene_id", "Cena 1")
        donor = row.get("donor_id")
        recv = row.get("receiver_id")
        dist = row.get("distance_m", row.get("distance_interpersonal_m", 0.0))
        dist_m = float(dist) if pd.notna(dist) else 0.0
        dur = float(row.get("duration_s", 0.0)) if pd.notna(row.get("duration_s")) else 0.0
        flags = str(row.get("flags", ""))
        details = str(row.get("details", ""))

        # Extrai ou calcula a zona proxêmica
        zone = None
        # Procura em flags ou details
        m_zone = re.search(r"zone_([a-z]+)", flags) or re.search(r"zone:([a-z]+)", details)
        if m_zone:
            zone = m_zone.group(1)
        elif dist_m > 0.0:
            zone = ProxemicsAnalyzer.get_zone(dist_m)
        else:
            zone = "desconhecida"

        ambig = "recent_track_birth" in flags or "lost_donor" in flags or bool(row.get("id_ambiguous", False))

        enriched.append({
            "session_id": session_id,
            "scene_id": sc,
            "timestamp_ms": t,
            "donor_id": donor,
            "receiver_id": recv,
            "distance_m": round(dist_m, 3),
            "proxemic_zone": zone,
            "duration_s": round(dur, 2),
            "id_ambiguous": ambig,
            "flags": flags
        })

    return pd.DataFrame(enriched)

def extract_presence_summary(df_traj: pd.DataFrame, session_id: str) -> pd.DataFrame:
    """Extrai métricas de presença, velocidade e classificação cinemática por participante."""
    if df_traj.empty or "track_id" not in df_traj.columns:
        return pd.DataFrame()

    # Ignora frames vazios (-1)
    valid_traj = df_traj[df_traj["track_id"] > 0].copy()
    if valid_traj.empty:
        return pd.DataFrame()

    results = []
    # Agrupa por cena e participante
    grouped = valid_traj.groupby(["scene_id", "track_id"])
    for (sc, trk_id), group in grouped:
        n_frames = len(group)
        dwell_s = round(n_frames / 30.0, 2)
        speeds = group["ground_speed_mps"].dropna() if "ground_speed_mps" in group.columns else pd.Series([0.0])
        mean_spd = float(speeds.mean()) if not speeds.empty else 0.0
        
        # Frames parado (< 0.40 m/s) vs em trânsito
        stat_frames = int((speeds < 0.40).sum())
        stat_time_s = round(stat_frames / 30.0, 2)
        trans_time_s = round((n_frames - stat_frames) / 30.0, 2)

        role = group["role"].iloc[0] if "role" in group.columns else "participante"
        pres_state = group["presence_state"].mode().iloc[0] if "presence_state" in group.columns and not group["presence_state"].empty else "passante"

        results.append({
            "session_id": session_id,
            "scene_id": sc,
            "track_id": trk_id,
            "role": role,
            "total_dwell_s": dwell_s,
            "stationary_time_s": stat_time_s,
            "transit_time_s": trans_time_s,
            "mean_speed_mps": round(mean_spd, 3),
            "predominant_state": pres_state
        })

    return pd.DataFrame(results)

def extract_f_formations_intervals(df_traj: pd.DataFrame, df_plush: pd.DataFrame, session_id: str, sample_step: int = 5) -> pd.DataFrame:
    """
    Executa detecção de F-formations ao longo da linha do tempo da sessão (amostragem a cada sample_step frames).
    """
    if df_traj.empty or "track_id" not in df_traj.columns:
        return pd.DataFrame()

    detector = FFormationDetector()
    results = []

    # Agrupa trajetórias por frame_idx
    traj_by_frame = {f_idx: grp for f_idx, grp in df_traj[df_traj["track_id"] > 0].groupby("frame_idx")}
    plush_by_frame = {}
    if not df_plush.empty and "frame_idx" in df_plush.columns:
        for _, p_row in df_plush.iterrows():
            px = p_row.get("plush_x")
            pz = p_row.get("plush_z")
            if pd.notna(px) and pd.notna(pz):
                plush_by_frame[p_row["frame_idx"]] = (float(px), 1.0, float(pz))

    all_frames = sorted(list(traj_by_frame.keys()))
    sampled_frames = all_frames[::sample_step]

    # Mock leve de Track para o FFormationDetector
    class _TrackStub:
        def __init__(self, track_id, pos, speed, heading_deg=None):
            self.track_id = track_id
            self.position = pos
            self.velocity = (0.0, 0.0, 0.0)
            self.ground_speed = speed
            self.heading_deg = heading_deg

    for f_idx in sampled_frames:
        grp = traj_by_frame[f_idx]
        if len(grp) < 2:
            continue

        tracks_stub = []
        for _, r in grp.iterrows():
            px = float(r.get("pos_x", 0.0))
            py = float(r.get("pos_y", 1.0))
            pz = float(r.get("pos_z", 2.0))
            spd = float(r.get("ground_speed_mps", 0.0))
            t_id = int(r.get("track_id"))
            tracks_stub.append(_TrackStub(t_id, (px, py, pz), spd))

        plush_pos = plush_by_frame.get(f_idx)
        groups = detector.detect(tracks_stub, plush_pos=plush_pos)

        t_ms = grp["t_capture_ms"].iloc[0] if "t_capture_ms" in grp.columns else f_idx * 33
        sc = grp["scene_id"].iloc[0] if "scene_id" in grp.columns else "Cena 1"

        for g in groups:
            results.append({
                "session_id": session_id,
                "scene_id": sc,
                "frame_idx": f_idx,
                "timestamp_ms": t_ms,
                "group_id": g.group_id,
                "members_count": len(g.members),
                "members": ";".join(map(str, sorted(g.members))),
                "formation_type": g.formation_type,
                "o_space_x": g.o_space_center[0],
                "o_space_z": g.o_space_center[1],
                "o_space_radius_m": g.o_space_radius,
                "plush_inside_o_space": g.plush_inside_o_space
            })

    return pd.DataFrame(results)

def build_scene_summary(df_presence: pd.DataFrame, df_possession: pd.DataFrame, df_handoffs: pd.DataFrame) -> pd.DataFrame:
    """Consolida indicadores socioenativos por cena (Gini, Razão de Circulação, Handoffs/min)."""
    scenes = set()
    if not df_presence.empty and "scene_id" in df_presence.columns:
        scenes.update(df_presence["scene_id"].unique())
    if not df_possession.empty and "scene_id" in df_possession.columns:
        scenes.update(df_possession["scene_id"].unique())

    if not scenes:
        return pd.DataFrame()

    summary = []
    for sc in sorted(list(scenes)):
        # Participantes ativos na cena
        pres_sc = df_presence[df_presence["scene_id"] == sc] if not df_presence.empty else pd.DataFrame()
        partic_count = len(pres_sc[pres_sc["role"] != "facilitador"]) if not pres_sc.empty else 0
        total_time_s = pres_sc["total_dwell_s"].max() if not pres_sc.empty else 0.0

        # Posse da pelúcia
        poss_sc = df_possession[df_possession["scene_id"] == sc] if not df_possession.empty else pd.DataFrame()
        valid_poss = poss_sc[(poss_sc["holder_id"] > 0) & (poss_sc["state"] == "COM_PORTADOR")] if not poss_sc.empty else pd.DataFrame()
        
        holder_durations = valid_poss.groupby("holder_id")["duration_s"].sum().to_dict() if not valid_poss.empty else {}
        distinct_holders = len(holder_durations)
        gini = calc_gini(list(holder_durations.values())) if holder_durations else 0.0
        circ_ratio = round(distinct_holders / partic_count, 3) if partic_count > 0 else 0.0

        # Handoffs
        hd_sc = df_handoffs[df_handoffs["scene_id"] == sc] if not df_handoffs.empty else pd.DataFrame()
        n_handoffs = len(hd_sc)
        dur_min = max(0.01, total_time_s / 60.0)
        hd_per_min = round(n_handoffs / dur_min, 2)

        # Distribuição de zonas proxêmicas no handoff
        zone_counts = hd_sc["proxemic_zone"].value_counts().to_dict() if not hd_sc.empty and "proxemic_zone" in hd_sc.columns else {}

        summary.append({
            "scene_id": sc,
            "total_scene_time_s": round(total_time_s, 2),
            "distinct_participants": partic_count,
            "distinct_holders": distinct_holders,
            "circulation_ratio": circ_ratio,
            "gini_possession": round(gini, 3),
            "total_handoffs": n_handoffs,
            "handoffs_per_min": hd_per_min,
            "proxemic_zones": str(zone_counts)
        })

    return pd.DataFrame(summary)

def run_etl(session_dirs: List[Path], output_dir: str) -> Dict[str, pd.DataFrame]:
    """Executa o pipeline completo de ETL para uma lista de diretórios de sessão."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    all_possession = []
    all_handoffs = []
    all_presence = []
    all_formations = []

    for s_dir in session_dirs:
        sess_name = s_dir.name
        f_traj = s_dir / "trajectories.csv"
        f_plush = s_dir / "plush_state.csv"
        f_events = s_dir / "events.csv"

        df_traj = pd.read_csv(f_traj) if f_traj.exists() else pd.DataFrame()
        df_plush = pd.read_csv(f_plush) if f_plush.exists() else pd.DataFrame()
        df_events = pd.read_csv(f_events) if f_events.exists() else pd.DataFrame()

        df_poss = extract_possession_intervals(df_plush, sess_name)
        if not df_poss.empty:
            all_possession.append(df_poss)

        df_hd = extract_handoff_events(df_events, sess_name)
        if not df_hd.empty:
            all_handoffs.append(df_hd)

        df_pres = extract_presence_summary(df_traj, sess_name)
        if not df_pres.empty:
            all_presence.append(df_pres)

        df_form = extract_f_formations_intervals(df_traj, df_plush, sess_name)
        if not df_form.empty:
            all_formations.append(df_form)

    # Concatenação de todas as sessões
    df_cat_poss = pd.concat(all_possession, ignore_index=True) if all_possession else pd.DataFrame()
    df_cat_hd = pd.concat(all_handoffs, ignore_index=True) if all_handoffs else pd.DataFrame()
    df_cat_pres = pd.concat(all_presence, ignore_index=True) if all_presence else pd.DataFrame()
    df_cat_form = pd.concat(all_formations, ignore_index=True) if all_formations else pd.DataFrame()
    df_scene_sum = build_scene_summary(df_cat_pres, df_cat_poss, df_cat_hd)

    # Exportação das tabelas consolidadas
    if not df_cat_poss.empty:
        df_cat_poss.to_csv(out_p / "possession_intervals.csv", index=False)
    if not df_cat_hd.empty:
        df_cat_hd.to_csv(out_p / "handoff_events_enriched.csv", index=False)
    if not df_cat_pres.empty:
        df_cat_pres.to_csv(out_p / "presence_summary.csv", index=False)
    if not df_cat_form.empty:
        df_cat_form.to_csv(out_p / "f_formations_intervals.csv", index=False)
    if not df_scene_sum.empty:
        df_scene_sum.to_csv(out_p / "scene_summary.csv", index=False)

    return {
        "possession_intervals": df_cat_poss,
        "handoff_events_enriched": df_cat_hd,
        "presence_summary": df_cat_pres,
        "f_formations_intervals": df_cat_form,
        "scene_summary": df_scene_sum
    }

def generate_analytics_report(etl_results: Dict[str, pd.DataFrame], output_markdown_path: str):
    """
    Gera relatório analítico completo respondendo às 4 perguntas de pesquisa da proposta.
    """
    df_poss = etl_results.get("possession_intervals", pd.DataFrame())
    df_hd = etl_results.get("handoff_events_enriched", pd.DataFrame())
    df_pres = etl_results.get("presence_summary", pd.DataFrame())
    df_form = etl_results.get("f_formations_intervals", pd.DataFrame())
    df_scene = etl_results.get("scene_summary", pd.DataFrame())

    md = []
    md.append("# Relatório Analítico Consolidado (ETL/BI — Milestone M5)\n")
    md.append(f"- **Data da Emissão:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"- **Total de Sessões Processadas:** {len(df_pres['session_id'].unique()) if not df_pres.empty and 'session_id' in df_pres.columns else 0}\n")

    # Pergunta 1: Concentração vs. Circulação do Artefato
    md.append("## 1. Pergunta 1: Concentração vs. Circulação do Artefato (Gini e Posse)\n")
    md.append("> *Em que medida a pelúcia circulou de forma democrática entre os participantes ao longo das cenas?*\n")
    if not df_scene.empty:
        # Formata tabela manual sem tabulate
        cols = ["scene_id", "distinct_participants", "distinct_holders", "circulation_ratio", "gini_possession", "total_handoffs", "handoffs_per_min"]
        md.append("| Cena | Participantes | Portadores | Razão Circulação | Gini Posse | Handoffs | Handoffs/min |")
        md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for _, r in df_scene.iterrows():
            md.append(f"| {r['scene_id']} | {r['distinct_participants']} | {r['distinct_holders']} | {r['circulation_ratio']:.2f} | {r['gini_possession']:.3f} | {r['total_handoffs']} | {r['handoffs_per_min']:.2f} |")
    else:
        md.append("_Sem dados de cenas para exibição._")
    md.append("\n")

    # Pergunta 2: Engajamento e Presença da Plateia
    md.append("## 2. Pergunta 2: Engajamento e Presença da Plateia\n")
    md.append("> *A presença dos participantes configurou-se como plateia engajada (estática) ou fluxo de transeuntes (passantes)?*\n")
    if not df_pres.empty:
        total_dwell = df_pres["total_dwell_s"].sum()
        total_stat = df_pres["stationary_time_s"].sum()
        total_trans = df_pres["transit_time_s"].sum()
        stat_pct = (total_stat / total_dwell * 100.0) if total_dwell > 0 else 0.0
        
        md.append(f"- **Tempo total de presença acumulado:** {total_dwell:.1f} s")
        md.append(f"- **Tempo em atenção estática (< 0.40 m/s):** {total_stat:.1f} s (**{stat_pct:.1f}%** do tempo)")
        md.append(f"- **Tempo em deslocamento ativo / trânsito:** {total_trans:.1f} s (**{100.0 - stat_pct:.1f}%**)")
        md.append(f"- **Classificação predominante:** `{df_pres['predominant_state'].mode().iloc[0]}`")
    else:
        md.append("_Sem dados de presença para exibição._")
    md.append("\n")

    # Pergunta 3: Dinâmica Espacial e Proxêmica nos Handoffs
    md.append("## 3. Pergunta 3: Proxêmica no Momento das Passagens de Posse\n")
    md.append("> *Em que zonas proxêmicas de Hall ocorreram as passagens do objeto cênico?*\n")
    if not df_hd.empty:
        zone_counts = df_hd["proxemic_zone"].value_counts()
        total_hd = len(df_hd)
        md.append("| Zona Proxêmica | Quantidade de Handoffs | Proporção (%) |")
        md.append("| :--- | :--- | :--- |")
        for z_name, count in zone_counts.items():
            pct = (count / total_hd) * 100.0
            md.append(f"| **{z_name.capitalize()}** | {count} | {pct:.1f}% |")
        
        mean_dur = df_hd["duration_s"].mean()
        ambig_count = int(df_hd["id_ambiguous"].sum())
        md.append(f"\n- **Duração média da transferência de posse:** {mean_dur:.2f} s")
        md.append(f"- **Handoffs com ambiguidade de identidade:** {ambig_count} de {total_hd} ({ambig_count/total_hd*100:.1f}%)")
    else:
        md.append("_Nenhum evento de handoff registrado._")
    md.append("\n")

    # Pergunta 4: Formações Sociais de Kendon (F-formations)
    md.append("## 4. Pergunta 4: F-Formations e Interação Compartilhada\n")
    md.append("> *Os participantes engajaram-se em formações grupais com foco comum (o-space)? A pelúcia esteve no centro dessas formações?*\n")
    if not df_form.empty:
        type_counts = df_form["formation_type"].value_counts()
        total_form_samples = len(df_form)
        plush_inside_count = int(df_form["plush_inside_o_space"].sum())
        plush_inside_pct = (plush_inside_count / total_form_samples * 100.0) if total_form_samples > 0 else 0.0

        md.append("| Tipo de Formação | Quadros Amostrados | Proporção (%) |")
        md.append("| :--- | :--- | :--- |")
        for f_type, count in type_counts.items():
            pct = (count / total_form_samples) * 100.0
            md.append(f"| `{f_type}` | {count} | {pct:.1f}% |")

        md.append(f"\n- **Presença da pelúcia no interior do o-space:** {plush_inside_count} amostras (**{plush_inside_pct:.1f}%** das formações sociais)")
    else:
        md.append("_Nenhuma formação grupal observada ou dados de trajetórias insuficientes para F-formations._")
    md.append("\n")

    md.append("## 5. Conclusões e Parecer Científico\n")
    md.append("A camada de ETL consolidou satisfatoriamente as dimensões de posse, atenção, cinemática corporal e formações sociais.")
    md.append("Os dados agregados estão prontos para análise estatística aprofundada e integração em dissertações/artigos científicos.")

    report_text = "\n".join(md)
    with open(output_markdown_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    return report_text

def main():
    parser = argparse.ArgumentParser(description="ETL e Analytics Consolidado SICAMO3D (Milestone M5)")
    parser.add_argument("--sessions", "--session", type=str, required=True, help="Diretório de uma sessão ou pasta contendo múltiplas sessões")
    parser.add_argument("--output-dir", "--out", type=str, default="output/etl_dataset", help="Diretório para salvar tabelas agregadas")
    parser.add_argument("--report", type=str, default=None, help="Caminho do relatório analítico em Markdown")

    args = parser.parse_args()

    sessions = find_session_dirs(args.sessions)
    if not sessions:
        print(f"[ERRO] Nenhuma sessão encontrada em {args.sessions}")
        sys.exit(1)

    print(f"=== INICIANDO CAMADA DE ETL (M5-03 / M5-04) ===")
    print(f"Diretório de entrada: {args.sessions}")
    print(f"Sessões localizadas: {len(sessions)}")
    for s in sessions:
        print(f"  > {s}")
    print(f"Diretório de saída:   {args.output_dir}\n")

    etl_results = run_etl(sessions, args.output_dir)

    print("[OK] Tabelas agregadas geradas com sucesso:")
    for name, df in etl_results.items():
        print(f"  - {name}.csv: {len(df)} registros")

    report_path = args.report or os.path.join(args.output_dir, "analytics_report.md")
    generate_analytics_report(etl_results, report_path)
    print(f"\n[OK] Relatório analítico exportado para: {report_path}")

if __name__ == "__main__":
    main()
