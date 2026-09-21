import time
import cv2
import ctypes
import sys
import os
import numpy as np

from src.core.config import config
from src.core.pipeline import Pipeline
from src.network.network_protocol import HubReceiver

def main():
    # Ativa precisão de 1ms no temporizador do Windows kernel (reduz cv2.waitKey de 16.7ms para 1.8ms)
    ctypes.windll.winmm.timeBeginPeriod(1)

    print("=" * 80)
    print(f"SICAMO3D: SISTEMA DE INFERÊNCIA COM CAPTURA DE MOVIMENTO DE OBJETOS 3D (v{config.version})")
    print("=" * 80)

    # 1. Inicializa o Pipeline Central
    print("\n[1/3] Inicializando Pipeline Central (Kinect v2, DirectML, Tracker3D, HolderInference)...")
    pipeline = Pipeline(enable_logger=True)

    # 2. Inicializa receptor de rede local para nó secundário (Notebook / Kinect #2)
    print("\n[2/3] Iniciando Receptor de Rede Local (Hub UDP Port: 5555)...")
    hub_receiver = HubReceiver(host=config.network.hub_ip, port=config.network.hub_port)
    hub_receiver.start()

    # 3. Janela de Exibição Interativa
    print("\n[3/3] Configurando Interface Visual e Modo do Operador (Modo A)...")
    win_name = "SICAMO3D - Rastreamento Socioenativo da Pelúcia Móvel"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1600, 900)

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        active_tracks = pipeline.last_tracks
        if not active_tracks:
            return

        vmode = pipeline.view_mode
        selected = None

        if vmode == 1:
            # Modo Triplo:
            # Câmera: x in [0, 960], y in [0, 540]
            if 0 <= x < 960 and 0 <= y < 540:
                sx = 1920.0 / 960.0
                sy = 1080.0 / 540.0
                cx_orig, cy_orig = x * sx, y * sy
                for trk in active_tracks:
                    if trk.last_bbox is not None:
                        bx1, by1, bx2, by2 = trk.last_bbox
                        if bx1 <= cx_orig <= bx2 and by1 <= cy_orig <= by2:
                            selected = trk.track_id
                            break
            # Planta Baixa: x in [960, 1600], y in [0, 540]
            elif 960 <= x < 1600 and 0 <= y < 540:
                grid_cx = 960 + 320
                grid_origin_y = 500
                ppm = 70.0
                floor_x = (x - grid_cx) / ppm
                floor_z = (grid_origin_y - y) / ppm
                best_dist = 0.80
                for trk in active_tracks:
                    px, _, pz = trk.position
                    d = float(np.hypot(px - floor_x, pz - floor_z))
                    if d < best_dist:
                        best_dist = d
                        selected = trk.track_id

        elif vmode == 2:
            # Planta Baixa Fullscreen (1600 x 900)
            grid_cx = 800
            grid_origin_y = 820
            ppm = 110.0
            floor_x = (x - grid_cx) / ppm
            floor_z = (grid_origin_y - y) / ppm
            best_dist = 0.90
            for trk in active_tracks:
                px, _, pz = trk.position
                d = float(np.hypot(px - floor_x, pz - floor_z))
                if d < best_dist:
                    best_dist = d
                    selected = trk.track_id

        elif vmode == 3:
            # Câmera Fullscreen (1600 x 900)
            sx = 1920.0 / 1600.0
            sy = 1080.0 / 900.0
            cx_orig, cy_orig = x * sx, y * sy
            for trk in active_tracks:
                if trk.last_bbox is not None:
                    bx1, by1, bx2, by2 = trk.last_bbox
                    if bx1 <= cx_orig <= bx2 and by1 <= cy_orig <= by2:
                        selected = trk.track_id
                        break

        if selected is not None:
            pipeline.selected_track_id = selected
            print(f"[OPERADOR] Alvo selecionado por clique: Track #{selected}")

    cv2.setMouseCallback(win_name, on_mouse)

    print("\nSISTEMA PRONTO PARA OPERAÇÃO!")
    print("Atalhos do Operador:")
    print("  [N] Avançar Cena Teatral")
    print("  [TAB / Clique] Selecionar Participante na Sala")
    print("  [P] Atribuir/Alternar Portador do Alvo Selecionado (Modo A)")
    print("  [F] Alternar Papel de Facilitador do Alvo Selecionado")
    print("  [1/2/3] Modos de Tela (Triplo / Mapa 3D / Câmera AR)")
    print("  [T] Trajetórias Neon | [S] Esqueleto | [M] Ajuda | [Q/ESC] Sair\n")

    try:
        while True:
            # Recebe observações remotas do nó sensor secundário se disponível
            remote_dets = hub_receiver.get_latest_detections()

            # Executa passo completo do pipeline
            success, canvas = pipeline.step(remote_detections=remote_dets)
            if not success or canvas is None:
                time.sleep(0.002)
                continue

            cv2.imshow(win_name, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break

            # Controles do Operador Teatral (Modo A)
            elif key in [ord('n'), ord('N')]:
                next_scene = pipeline.advance_scene()
                print(f"[OPERADOR] Avançou para a cena: {next_scene}")

            elif key == 9:  # TAB: Alterna entre tracks ativos
                if pipeline.last_tracks:
                    track_ids = [t.track_id for t in pipeline.last_tracks]
                    if pipeline.selected_track_id in track_ids:
                        cur_idx = track_ids.index(pipeline.selected_track_id)
                        pipeline.selected_track_id = track_ids[(cur_idx + 1) % len(track_ids)]
                    else:
                        pipeline.selected_track_id = track_ids[0]
                    print(f"[OPERADOR] Alvo selecionado via TAB: Track #{pipeline.selected_track_id}")

            elif key in [ord('p'), ord('P')]:
                # Atribui portador ao alvo selecionado ou ao primeiro participante
                target_id = pipeline.selected_track_id
                if target_id is None and pipeline.last_tracks:
                    target_id = pipeline.last_tracks[0].track_id
                if target_id is not None:
                    pipeline.set_manual_holder(target_id, duration_s=6.0)
                    print(f"[OPERADOR] Portador manual definido para o track #{target_id}")

            elif key in [ord('f'), ord('F')]:
                target_id = pipeline.selected_track_id
                if target_id is None and pipeline.last_tracks:
                    target_id = pipeline.last_tracks[0].track_id
                if target_id is not None:
                    pipeline.toggle_facilitator(target_id)
                    print(f"[OPERADOR] Papel do track #{target_id} alterado")

            # Controles de Visualização
            elif key == ord('1'):
                pipeline.view_mode = 1
            elif key == ord('2'):
                pipeline.view_mode = 2
            elif key == ord('3'):
                pipeline.view_mode = 3
            elif key in [ord('t'), ord('T')]:
                pipeline.show_trajectories = not pipeline.show_trajectories
            elif key in [ord('s'), ord('S')]:
                pipeline.skeleton_mode = (pipeline.skeleton_mode + 1) % 3
            elif key in [ord('m'), ord('M')]:
                pipeline.show_hud_help = not pipeline.show_hud_help

    except KeyboardInterrupt:
        print("\nInterrupção manual recebida.")
    finally:
        print("\nEncerrando sistema e gravando descritores de arquivo...")
        hub_receiver.stop()
        pipeline.close()
        cv2.destroyAllWindows()
        ctypes.windll.winmm.timeEndPeriod(1)
        print("Sistema finalizado com sucesso.")

if __name__ == "__main__":
    main()

