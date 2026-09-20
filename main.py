"""
Ponto de entrada principal: Sistema de Rastreamento Socioenativo 3D da Pelúcia Móvel (SICAMO3D v0.2.0)
Nativo para Windows 11 com aceleração DirectML (DirectX 12).
"""
import time
import cv2
import ctypes
import sys
import os

from src.core.config import config
from src.core.pipeline import Pipeline
from src.network.network_protocol import HubReceiver

def main():
    # Ativa precisão de 1ms no temporizador do Windows kernel (reduz cv2.waitKey de 16.7ms para 1.8ms)
    ctypes.windll.winmm.timeBeginPeriod(1)

    print("=" * 75)
    print(f"SICAMO3D: SISTEMA SOCIOENATIVO 3D - PELÚCIA MÓVEL E CENAS (v{config.version})")
    print("=" * 75)

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

    # Estados de visualização do dashboard
    view_mode = 1           # 1: Triplo, 2: Mapa 3D, 3: AR Full
    show_trajectories = True
    skeleton_mode = 2
    show_hud_help = False

    print("\nSISTEMA PRONTO PARA OPERAÇÃO!")
    print("Atalhos do Operador:")
    print("  [N] Próxima Cena Teatral")
    print("  [P] Atribuir/Alternar Portador Manualmente (Modo A)")
    print("  [F] Alternar Papel de Facilitador do 1º Alvo")
    print("  [1/2/3] Modos de Tela (Triplo / Mapa 3D / Câmera AR)")
    print("  [T] Alternar Trajetórias Neon | [S] Alternar Esqueletos | [M] Ajuda | [Q/ESC] Sair\n")

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

            elif key in [ord('p'), ord('P')]:
                # Atribui portador ao primeiro participante ativo
                if pipeline.last_tracks:
                    target_id = pipeline.last_tracks[0].track_id
                    pipeline.set_manual_holder(target_id, duration_s=6.0)
                    print(f"[OPERADOR] Portador manual definido para o track #{target_id}")

            elif key in [ord('f'), ord('F')]:
                if pipeline.last_tracks:
                    target_id = pipeline.last_tracks[0].track_id
                    pipeline.toggle_facilitator(target_id)
                    print(f"[OPERADOR] Papel do track #{target_id} alterado")

            # Controles de Visualização
            elif key == ord('1'):
                pipeline.dashboard.view_mode = 1
            elif key == ord('2'):
                pipeline.dashboard.view_mode = 2
            elif key == ord('3'):
                pipeline.dashboard.view_mode = 3
            elif key in [ord('t'), ord('T')]:
                show_trajectories = not show_trajectories
            elif key in [ord('s'), ord('S')]:
                skeleton_mode = (skeleton_mode + 1) % 3
            elif key in [ord('m'), ord('M')]:
                show_hud_help = not show_hud_help

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
