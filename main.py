"""
Ponto de entrada principal: Sistema de Captura de Movimento em 3D Multi-Alvo (Windows + AMD Radeon RX 6600)
"""
import time
import cv2
import numpy as np
import os
import sys
import ctypes

from src.core.config import config
from src.core.kinect_sensor import KinectSensor
from src.ai.directml_inference import DirectMLInference
from src.tracking.tracker_3d import Tracker3D
from src.socioenative.posture_classifier import PostureClassifier
from src.socioenative.proxemics import ProxemicsAnalyzer
from src.socioenative.toy_interaction import ToyInteractionDetector
from src.socioenative.scientific_logger import ScientificLogger
from src.network.network_protocol import HubReceiver
from src.ai.hand_roi_detector import HandROIDetector
from src.ai.depth_clustering_3d import DepthCluster3D
from src.visualization.dashboard_3d import Dashboard3D

def main():
    # Ativa precisão de 1ms no temporizador do Windows kernel (reduz cv2.waitKey(1) de 16.7ms para 1.8ms!)
    ctypes.windll.winmm.timeBeginPeriod(1)

    print("=" * 70)
    print(f"SISTEMA DE CAPTURA DE MOVIMENTO EM 3D - OPENPTRACK MODERNO (v{config.version})")
    print("=" * 70)

    # 1. Inicializa o sensor Kinect v2
    print("\n[1/6] Inicializando sensor Kinect v2...")
    sensor = KinectSensor()

    # 2. Inicializa os motores de IA com aceleração DirectML na GPU AMD
    print("\n[2/6] Carregando modelos de IA no DirectML (AMD Radeon RX 6600)...")
    pose_engine = DirectMLInference(config.ai.pose_model_path, conf_thresh=config.ai.conf_threshold)
    object_engine = DirectMLInference(config.ai.object_model_path, conf_thresh=config.ai.toy_conf_threshold)
    print(f"  > Modelo de Poses: {config.ai.pose_model_path} (Provedor: {pose_engine.active_provider})")
    print(f"  > Modelo de Objetos: {config.ai.object_model_path} (Provedor: {object_engine.active_provider})")

    # 3. Inicializa o rastreador espacial 3D
    print("\n[3/6] Inicializando Motor de Rastreamento 3D e Filtro de Kalman...")
    tracker = Tracker3D(max_distance_m=config.tracking.max_distance_threshold, 
                        max_lost_frames=config.tracking.max_frames_to_keep_lost)

    # 4. Inicializa os motores socioenativos e gravador científico
    print("\n[4/6] Inicializando Analisadores Socioenativos e Gravador Científico...")
    toy_detector = ToyInteractionDetector(hand_toy_threshold_m=config.tracking.proximity_toy_threshold_m)
    hand_roi_detector = HandROIDetector(physical_roi_size_m=0.55)
    depth_clusterer = DepthCluster3D(roi_radius_m=0.28)
    logger = ScientificLogger(output_dir=config.dataset_output_dir)
    print(f"  > Gravação ativa em: {logger.output_dir}/")

    # 5. Inicializa receptor de rede para nó secundário (Notebook / GTX 1060)
    print("\n[5/6] Iniciando Receptor de Rede Local (Hub Port: 5555)...")
    hub_receiver = HubReceiver(host=config.network.hub_ip, port=config.network.hub_port)
    hub_receiver.start()

    # 6. Inicializa o Dashboard Visual
    print("\n[6/6] Inicializando Dashboard Visual Interativo...")
    dashboard = Dashboard3D(room_width_m=6.0, room_depth_m=6.0)

    print("\nSISTEMA PRONTO! Pressione [Q] ou [ESC] na janela de exibição para encerrar.")
    cv2.namedWindow("Sistema Socioenativo 3D (AMD RX 6600)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sistema Socioenativo 3D (AMD RX 6600)", 1600, 900)

    fps = 30.0
    frame_count = 0
    t_start = time.perf_counter()

    # Estados Visuais Interativos (Fase 4)
    view_mode = 1           # 1: Dashboard Triplo, 2: Mapa 3D Full, 3: Câmera AR Full
    show_trajectories = True # Linhas neon contínuas
    skeleton_mode = 2       # 0: Oculto, 1: Mãos, 2: Completo
    show_hud_help = False   # Menu de atalhos rápido

    cached_global_toys = []
    cached_hand_toys = []
    cached_hand_crops = []
    active_tracks = []

    try:
        while True:
            t_loop_start = time.perf_counter()

            # Captura frames do Kinect (aguarda novo frame de 30 FPS com baixíssimo uso de CPU)
            if not sensor.update():
                time.sleep(0.001)
                continue

            color_bgr = sensor.last_color_bgr
            orig_shape = color_bgr.shape[:2]

            # Preprocessamento para 640x640 letterbox
            blob, ratio, pad = pose_engine.preprocess(color_bgr)

            # Início da medição de IA na GPU AMD
            t_ai0 = time.perf_counter()
            raw_pose = pose_engine.run_raw(blob)

            # Escalonamento Inteligente de Objetos (Mantém exatamente 2 inferências por frame para 30 FPS constantes)
            # Frame ímpar com pessoas ativas: inspeciona 1 mão no recorte 1080p em alta definição
            # Frame par (ou sem pessoas): faz varredura global na sala inteira
            run_hand_roi_now = (frame_count % 2 == 1) and (len(active_tracks) > 0)

            if not run_hand_roi_now:
                raw_obj = object_engine.run_raw(blob)
                obj_dets = object_engine.postprocess_objects(
                    raw_obj, ratio, pad, orig_shape, 
                    config.ai.toy_classes, conf_thresh=config.ai.toy_conf_threshold
                )
                new_global = []
                for obj in obj_dets:
                    bx1, by1, bx2, by2 = obj['bbox']
                    cx = (bx1 + bx2) / 2.0
                    cy = (by1 + by2) / 2.0
                    pt_3d = sensor.get_3d_point_from_color(cx, cy)
                    if pt_3d is not None and pt_3d[2] > 0.35:
                        new_global.append({
                            "class_name": obj['class_name'],
                            "bbox": obj['bbox'],
                            "confidence": obj['confidence'],
                            "pos_3d": pt_3d
                        })
                cached_global_toys = new_global

            # Pós-processamento de Poses e Detecção de Pessoas (Sempre em 30 FPS contínuos)
            pose_dets = pose_engine.postprocess_pose(raw_pose, ratio, pad, orig_shape)

            # Base de brinquedos para este frame
            toys_3d = list(cached_global_toys)

            # Mapeamento 2D -> 3D real das pessoas e filtro de bonecos/brinquedos
            person_detections_3d = []
            for p in pose_dets:
                kpts_2d = p['keypoints'] # (17, 3) em pixels
                kpts_3d = sensor.unproject_keypoints_3d(kpts_2d) # (17, 4) em metros
                
                # Centro 3D anatômico estável (Pelve como âncora constante de massa)
                center_3d = None
                hip_pts = []
                if kpts_3d[11, 3] > 0.20 and kpts_3d[11, 2] > 0.35: hip_pts.append(kpts_3d[11, :3])
                if kpts_3d[12, 3] > 0.20 and kpts_3d[12, 2] > 0.35: hip_pts.append(kpts_3d[12, :3])

                if len(hip_pts) > 0:
                    center_3d = np.mean(hip_pts, axis=0)
                else:
                    # Se quadril estiver ocluso, projeta a partir dos ombros para o nível da pelve
                    sh_pts = []
                    if kpts_3d[5, 3] > 0.20 and kpts_3d[5, 2] > 0.35: sh_pts.append(kpts_3d[5, :3])
                    if kpts_3d[6, 3] > 0.20 and kpts_3d[6, 2] > 0.35: sh_pts.append(kpts_3d[6, :3])
                    if len(sh_pts) > 0:
                        sh_mean = np.mean(sh_pts, axis=0)
                        # No Kinect, +Y é para baixo (chão). Pelve fica ~42cm abaixo dos ombros
                        center_3d = np.array([sh_mean[0], sh_mean[1] + 0.42, sh_mean[2]])
                    else:
                        bx1, by1, bx2, by2 = p['bbox']
                        fb_pt = sensor.get_3d_point_from_color((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)
                        center_3d = np.array(fb_pt) if fb_pt is not None else np.array([0.0, 0.0, 1.8])

                # Diferenciação Métrica 3D: Boneco vs. Pessoa Real
                bx1, by1, bx2, by2 = p['bbox']
                h_pixels = by2 - by1
                center_depth_m = float(center_3d[2]) if center_3d[2] > 0.3 else 1.5
                metric_height_m = (h_pixels * center_depth_m) / 1060.0

                dist_shoulders_m = 999.0
                if kpts_3d[5, 3] > 0.20 and kpts_3d[6, 3] > 0.20:
                    dist_shoulders_m = float(np.linalg.norm(kpts_3d[5, :3] - kpts_3d[6, :3]))

                # Se a altura métrica for menor que 55cm ou os ombros tiverem menos de 20cm: é um BONECO/BRINQUEDO!
                is_doll = (metric_height_m < 0.55) or (0.01 < dist_shoulders_m < 0.20)

                if is_doll:
                    already_toy = False
                    for existing_toy in toys_3d:
                        if np.linalg.norm(np.array(existing_toy['pos_3d']) - center_3d) < 0.30:
                            already_toy = True
                            break
                    if not already_toy:
                        toys_3d.append({
                            "class_name": "boneco_toad",
                            "bbox": p['bbox'],
                            "confidence": p['confidence'],
                            "pos_3d": (float(center_3d[0]), float(center_3d[1]), float(center_3d[2]))
                        })
                else:
                    # Pessoa humana real
                    person_detections_3d.append({
                        "pos_3d": (float(center_3d[0]), float(center_3d[1]), float(center_3d[2])),
                        "keypoints_3d": kpts_3d,
                        "keypoints_2d": kpts_2d,
                        "bbox": p['bbox']
                    })

            # Integração com detecções do nó secundário se disponíveis
            remote_dets = hub_receiver.get_latest_detections()
            if remote_dets:
                person_detections_3d.extend(remote_dets)

            # Atualização do Rastreador 3D (Filtro de Kalman + Algoritmo Húngaro)
            active_tracks = tracker.update(person_detections_3d)

            # FASE 2: Hand-Centric High-Resolution ROI (Executado de forma intercalada a 15 Hz por mão)
            if run_hand_roi_now:
                cached_hand_toys, cached_hand_crops = hand_roi_detector.process(
                    color_1080p=color_bgr,
                    tracks=active_tracks,
                    object_engine=object_engine,
                    sensor=sensor,
                    toy_classes=config.ai.toy_classes,
                    conf_thresh=config.ai.toy_conf_threshold,
                    current_toys=toys_3d
                )

            t_ai1 = time.perf_counter()
            gpu_latency_ms = (t_ai1 - t_ai0) * 1000.0

            # Mescla brinquedos das mãos com a lista 3D geral
            for ht in cached_hand_toys:
                already_in = False
                ht_pos = np.array(ht['pos_3d'])
                for existing in toys_3d:
                    if np.linalg.norm(np.array(existing['pos_3d']) - ht_pos) < 0.25:
                        already_in = True
                        if ht['confidence'] > existing.get('confidence', 0):
                            existing['bbox'] = ht['bbox']
                            existing['confidence'] = ht['confidence']
                            existing['source'] = 'hand_roi'
                        break
                if not already_in:
                    toys_3d.append(ht)

            # Classificação de posturas para cada indivíduo
            for trk in active_tracks:
                trk.posture = PostureClassifier.classify(trk.last_keypoints_3d)

            # Análise de Proxêmica (Distâncias Interpessoais)
            proxemic_events = ProxemicsAnalyzer.calculate_pairwise(active_tracks)

            # FASE 4: Segmentação Geométrica 3D por Nuvem de Pontos (Depth Point Cloud Clustering)
            depth_clusters = depth_clusterer.analyze_hands(sensor.last_depth_mm, active_tracks, toys_3d)

            # Análise de Interação com Brinquedos e Atenção Conjunta
            toy_analysis = toy_detector.analyze(active_tracks, toys_3d)
            joint_events = toy_analysis["joint_attention_events"]

            # Gravação contínua dos dados para pesquisa
            if frame_count % 3 == 0: # Grava a ~10 Hz para economia de disco
                logger.log_trajectories(active_tracks)
                logger.log_interactions(proxemic_events, joint_events)

            # Renderização do Dashboard (Fase 4: Multi-modos + Trajetórias Neon + HUD)
            canvas = dashboard.render(
                color_bgr=color_bgr,
                tracks=active_tracks,
                toys=toys_3d,
                proxemic_events=proxemic_events,
                joint_events=joint_events,
                fps=fps,
                gpu_latency_ms=gpu_latency_ms,
                hand_crops=cached_hand_crops,
                depth_clusters=depth_clusters,
                view_mode=view_mode,
                show_trajectories=show_trajectories,
                skeleton_mode=skeleton_mode,
                show_hud_help=show_hud_help
            )

            cv2.imshow("Sistema Socioenativo 3D (AMD RX 6600)", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'): # ESC ou Q
                break
            elif key == ord('1'):
                view_mode = 1
            elif key == ord('2'):
                view_mode = 2
            elif key == ord('3'):
                view_mode = 3
            elif key == ord('t') or key == ord('T'):
                show_trajectories = not show_trajectories
            elif key == ord('s') or key == ord('S'):
                skeleton_mode = (skeleton_mode + 1) % 3
            elif key == ord('m') or key == ord('M'):
                show_hud_help = not show_hud_help

            frame_count += 1
            t_loop_end = time.perf_counter()
            inst_fps = 1.0 / max(1e-5, (t_loop_end - t_loop_start))
            fps = 0.9 * fps + 0.1 * inst_fps

    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")
    finally:
        print("\nEncerrando recursos...")
        hub_receiver.stop()
        sensor.close()
        logger.close()
        cv2.destroyAllWindows()
        ctypes.windll.winmm.timeEndPeriod(1)
        print(f"Sessão finalizada. Arquivos de dados gravados em: {logger.output_dir}/")

if __name__ == "__main__":
    main()
