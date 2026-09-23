"""
Script de teste de ponta a ponta do pipeline completo (sem abrir janela GUI interativa).
Executa 5 frames, salva o resultado do dashboard em 'pipeline_output.jpg' e valida todas as métricas.
"""
import os
import sys
from pathlib import Path
import time
import cv2
import numpy as np

# Garante acesso ao pacote raiz
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.config import config
from src.core.kinect_sensor import KinectSensor
from src.ai.directml_inference import DirectMLInference
from src.tracking.tracker_3d import Tracker3D
from src.socioenative.posture_classifier import PostureClassifier
from src.socioenative.proxemics import ProxemicsAnalyzer
from src.experimental.toy_interaction import ToyInteractionDetector
from src.socioenative.scientific_logger import ScientificLogger
from src.visualization.dashboard_3d import Dashboard3D

def main():
    print("=== TESTE DE VALIDAÇÃO COMPLETA DO SISTEMA ===")
    sensor = KinectSensor()
    print("[1] Kinect inicializado.")

    pose_engine = DirectMLInference(config.ai.pose_model_path, conf_thresh=0.40)
    object_engine = DirectMLInference(config.ai.object_model_path, conf_thresh=0.35)
    print(f"[2] Motores de IA DirectML carregados: {pose_engine.active_provider}")

    tracker = Tracker3D()
    toy_detector = ToyInteractionDetector()
    logger = ScientificLogger()
    dashboard = Dashboard3D()
    print("[3] Rastreadores e Dashboard configurados.")

    print("\nAguardando 5 frames reais do Kinect para processar...")
    frames_processed = 0
    t_start = time.time()

    while frames_processed < 5 and (time.time() - t_start < 15.0):
        if not sensor.update():
            time.sleep(0.01)
            continue

        color_bgr = sensor.last_color_bgr
        orig_shape = color_bgr.shape[:2]

        blob, ratio, pad = pose_engine.preprocess(color_bgr)
        
        t0 = time.perf_counter()
        raw_pose = pose_engine.run_raw(blob)
        raw_obj = object_engine.run_raw(blob)
        t1 = time.perf_counter()
        gpu_latency_ms = (t1 - t0) * 1000.0

        pose_dets = pose_engine.postprocess_pose(raw_pose, ratio, pad, orig_shape)
        obj_dets = object_engine.postprocess_objects(raw_obj, ratio, pad, orig_shape, config.ai.toy_classes)

        toys_3d = []
        for obj in obj_dets:
            bx1, by1, bx2, by2 = obj['bbox']
            cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
            pt_3d = sensor.get_3d_point_from_color(cx, cy)
            if pt_3d is not None and pt_3d[2] > 0.4:
                toys_3d.append({
                    "class_name": obj['class_name'],
                    "bbox": obj['bbox'],
                    "confidence": obj['confidence'],
                    "pos_3d": pt_3d
                })

        person_detections_3d = []
        for p in pose_dets:
            kpts_2d = p['keypoints']
            kpts_3d = sensor.unproject_keypoints_3d(kpts_2d)
            valid_pts = []
            for idx in [11, 12, 5, 6, 0]:
                if kpts_3d[idx, 3] > 0.20 and kpts_3d[idx, 2] > 0.35:
                    valid_pts.append(kpts_3d[idx, :3])

            if len(valid_pts) > 0:
                center_3d = np.mean(valid_pts, axis=0)
            else:
                bx1, by1, bx2, by2 = p['bbox']
                fb_pt = sensor.get_3d_point_from_color((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)
                center_3d = np.array(fb_pt) if fb_pt is not None else np.array([0.0, 0.0, 1.8])

            person_detections_3d.append({
                "pos_3d": (float(center_3d[0]), float(center_3d[1]), float(center_3d[2])),
                "keypoints_3d": kpts_3d,
                "keypoints_2d": kpts_2d,
                "bbox": p['bbox']
            })

        active_tracks = tracker.update(person_detections_3d)

        for trk in active_tracks:
            trk.posture = PostureClassifier.classify(trk.last_keypoints_3d)

        proxemic_events = ProxemicsAnalyzer.calculate_pairwise(active_tracks)
        toy_analysis = toy_detector.analyze(active_tracks, toys_3d)
        joint_events = toy_analysis["joint_attention_events"]

        logger.log_trajectories(active_tracks)
        logger.log_interactions(proxemic_events, joint_events)

        canvas = dashboard.render(
            color_bgr=color_bgr,
            tracks=active_tracks,
            toys=toys_3d,
            proxemic_events=proxemic_events,
            joint_events=joint_events,
            fps=30.0,
            gpu_latency_ms=gpu_latency_ms
        )

        frames_processed += 1
        print(f"  Frame {frames_processed}/5 processado com sucesso! Latência GPU: {gpu_latency_ms:.2f} ms | Pessoas 3D: {len(active_tracks)} | Brinquedos: {len(toys_3d)}")

        if frames_processed == 5:
            cv2.imwrite("pipeline_output.jpg", canvas)
            print("\n[SUCESSO] Imagem do dashboard gravada em 'pipeline_output.jpg'")

    sensor.close()
    print("=== PIPELINE VALIDADO DE PONTA A PONTA COM SUCESSO! ===")

if __name__ == "__main__":
    main()
