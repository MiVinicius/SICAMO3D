"""
Ferramenta de Reprodução Científica de Sessões Gravadas (Replay).
Lê arquivos gravados no Schema v3 (trajectories.csv, plush_state.csv, events.csv),
reconstrói a linha do tempo da sessão teatral e permite re-analisar métricas de circulação.
"""
import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
import cv2

def replay_session(session_dir: str, fps: float = 30.0):
    traj_path = os.path.join(session_dir, "trajectories.csv")
    plush_path = os.path.join(session_dir, "plush_state.csv")
    events_path = os.path.join(session_dir, "events.csv")

    if not os.path.exists(traj_path):
        print(f"Erro: Arquivo {traj_path} não encontrado!")
        return

    print("=" * 65)
    print(f"REPLAY DE SESSÃO CIENTÍFICA: {session_dir}")
    print("=" * 65)

    df_traj = pd.read_csv(traj_path)
    df_plush = pd.read_csv(plush_path) if os.path.exists(plush_path) else None
    df_events = pd.read_csv(events_path) if os.path.exists(events_path) else None

    # Agrupa por frame_idx
    frames = df_traj["frame_idx"].unique()
    frames.sort()
    print(f"Total de frames registrados: {len(frames)}")

    canvas = np.zeros((800, 800, 3), dtype=np.uint8)
    grid_cx = 400
    grid_origin_y = 700
    ppm = 90.0 # pixels por metro

    cv2.namedWindow("SICAMO3D - Replay Científico", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SICAMO3D - Replay Científico", 800, 800)

    delay_ms = max(1, int(1000.0 / fps))
    paused = False

    for f_idx in frames:
        canvas.fill(18)

        # Grade métrica
        for m in range(1, 6):
            y_m = int(grid_origin_y - m * ppm)
            cv2.line(canvas, (40, y_m), (760, y_m), (40, 40, 40), 1)
            cv2.putText(canvas, f"{m}m", (45, y_m - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1)

        # Desenha pessoas deste frame
        f_tracks = df_traj[df_traj["frame_idx"] == f_idx]
        scene_id = f_tracks.iloc[0]["scene_id"] if len(f_tracks) > 0 else "Cena 1"

        for _, row in f_tracks.iterrows():
            tid = int(row["track_id"])
            if tid == -1:
                continue # Sala vazia
            px = row["pos_x"]
            pz = row["pos_z"]
            if pd.isna(px) or pd.isna(pz):
                continue
            mx = int(grid_cx + px * ppm)
            my = int(grid_origin_y - pz * ppm)

            role = row.get("role", "participante")
            color = (0, 200, 255) if role == "facilitador" else (50, 255, 100)
            cv2.circle(canvas, (mx, my), 12, color, -1)
            cv2.putText(canvas, f"#{tid} ({role[:4]})", (mx + 15, my + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Desenha estado da pelúcia
        holder_txt = "SEM_PORTADOR"
        if df_plush is not None:
            p_rows = df_plush[df_plush["frame_idx"] == f_idx]
            if len(p_rows) > 0:
                p_row = p_rows.iloc[0]
                st = p_row["state"]
                hid = p_row["holder_id"]
                holder_txt = f"{st} (ID #{hid})" if hid != -1 else st
                pl_x = p_row["plush_x"]
                pl_z = p_row["plush_z"]
                if not pd.isna(pl_x) and not pd.isna(pl_z):
                    pmx = int(grid_cx + pl_x * ppm)
                    pmy = int(grid_origin_y - pl_z * ppm)
                    cv2.rectangle(canvas, (pmx - 8, pmy - 8), (pmx + 8, pmy + 8), (0, 255, 255), -1)
                    cv2.putText(canvas, "PELÚCIA", (pmx + 12, pmy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

        # Cabeçalho
        hdr = f"Frame: {f_idx} | {scene_id} | Pelúcia: {holder_txt}"
        cv2.putText(canvas, hdr, (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 180), 2)
        cv2.putText(canvas, "[ESPAÇO] Pausar/Continuar | [Q/ESC] Sair", (25, 770), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)

        cv2.imshow("SICAMO3D - Replay Científico", canvas)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key in [ord('q'), 27]:
            break
        elif key == ord(' '):
            paused = not paused
            while paused:
                k2 = cv2.waitKey(50) & 0xFF
                if k2 in [ord('q'), 27]:
                    break
                elif k2 == ord(' '):
                    paused = False

    cv2.destroyAllWindows()
    print("Replay concluído.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay de Sessão Científica SICAMO3D")
    parser.add_argument("session_dir", type=str, help="Caminho da pasta da sessão (ex: recordings/session_XYZ)")
    parser.add_argument("--fps", type=float, default=30.0, help="Taxa de quadros do replay")
    args = parser.parse_args()
    replay_session(args.session_dir, fps=args.fps)
