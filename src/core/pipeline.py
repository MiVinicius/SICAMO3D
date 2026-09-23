import os
import json
import time
import numpy as np
from typing import Optional, List, Dict, Tuple, Any

from src.core.config import config
from src.core.kinect_sensor import KinectSensor
from src.core.room_calibration import RoomCalibration
from src.core.zone_manager import ZoneManager
from src.ai.directml_inference import DirectMLInference
from src.tracking.tracker_3d import Tracker3D
from src.socioenative.holder_inference import HolderInference
from src.socioenative.plush_metrics import PlushMetricsAnalyzer
from src.socioenative.scientific_logger import ScientificLogger
from src.socioenative.posture_classifier import PostureClassifier
from src.socioenative.proxemics import ProxemicsAnalyzer
from src.socioenative.f_formations import FFormationDetector
from src.visualization.dashboard_3d import Dashboard3D

class Pipeline:
    def __init__(self, 
                 sensor: Optional[KinectSensor] = None,
                 enable_logger: bool = True,
                 session_name: Optional[str] = None):
        # 1. Sensores e Calibração
        self.sensor = sensor if sensor is not None else KinectSensor()
        self.room_calibration = RoomCalibration()
        self.sensor.set_room_calibration(self.room_calibration)

        # 2. Zonas e Cenas
        self.zone_manager = ZoneManager()

        # 3. Modelos de IA no DirectML
        # Detector de Pose (Obrigatório)
        if os.path.exists(config.ai.pose_model_path):
            self.pose_engine = DirectMLInference(
                config.ai.pose_model_path, 
                conf_thresh=config.ai.conf_threshold,
                device_id=config.ai.device_id
            )
        else:
            print(f"[AVISO] Modelo de Pose não encontrado em {config.ai.pose_model_path}.")
            self.pose_engine = None

        # Detector de Objetos / Pelúcia (Opcional no Modo A com anotação manual)
        if os.path.exists(config.ai.object_model_path):
            self.object_engine = DirectMLInference(
                config.ai.object_model_path, 
                conf_thresh=config.ai.toy_conf_threshold,
                device_id=config.ai.device_id
            )
            print(f"[INFO] Detector de pelúcia carregado: {config.ai.object_model_path}")
        else:
            print(f"[INFO] Detector de pelúcia não encontrado em {config.ai.object_model_path}. Operando no Modo A (anotação manual do portador).")
            self.object_engine = None

        # 4. Rastreamento Espacial 3D
        self.tracker = Tracker3D(
            max_distance_m=config.tracking.max_distance_threshold,
            max_lost_frames=config.tracking.max_frames_to_keep_lost
        )

        # 5. Inferência de Portador e Métricas de Circulação
        self.holder_inference = HolderInference(
            wrist_thresh_m=config.holder.wrist_threshold_m,
            torso_thresh_m=config.holder.torso_threshold_m,
            handoff_window_s=config.holder.handoff_window_s,
            min_margin=config.holder.min_score_margin,
            timeout_s=config.holder.state_timeout_s,
            weights=getattr(config.holder, "weights", (0.35, 0.35, 0.20, 0.10))
        )
        self.plush_metrics = PlushMetricsAnalyzer()
        self.plush_metrics.set_scene(self.zone_manager.current_scene_id)

        # 6. Gravador Científico Schema v3
        self.logger = ScientificLogger(output_dir=config.dataset_output_dir, session_name=session_name) if enable_logger else None

        # 7. Dashboard 3D e estados visuais
        self.dashboard = Dashboard3D(
            room_width_m=self.zone_manager.room_info.get("width_m", 6.0),
            room_depth_m=self.zone_manager.room_info.get("depth_m", 6.0)
        )
        self.view_mode: int = 1
        self.show_trajectories: bool = True
        self.skeleton_mode: int = 2
        self.show_hud_help: bool = False
        self.selected_track_id: Optional[int] = None

        self.frame_idx = 0
        self.last_time: Optional[float] = None
        self.fps_ema: float = 30.0
        self.last_holder_result: Dict[str, Any] = {"state": "INDETERMINADO", "holder_id": None, "confidence": 0.0}
        self.last_tracks: List = []
        self.last_plush_candidates: List[Dict] = []
        self.last_proxemic_events: List[Dict] = []
        self.last_joint_events: List[Dict] = []
        self.last_zone_evaluations: List[Dict] = []
        self.f_formation_detector = FFormationDetector()
        self.last_f_formations: List = []

    def advance_scene(self) -> str:
        """Avança cena no ZoneManager e PlushMetrics (Atalho 'N')."""
        old_scene = self.zone_manager.current_scene_id
        new_scene = self.zone_manager.next_scene()
        self.plush_metrics.set_scene(new_scene)
        if self.logger:
            self.logger.log_event(event_type="scene_change", scene_id=new_scene, details=f"Mudou de {old_scene} para {new_scene}")
        self.export_metrics_summary()
        return new_scene

    def set_manual_holder(self, track_id: Optional[int], duration_s: float = 5.0):
        """Marcação manual de portador pelo operador (Modo A)."""
        self.holder_inference.set_manual_holder(track_id, duration_s=duration_s)
        if self.logger:
            self.logger.log_event(
                event_type="operator_mark_holder", 
                scene_id=self.zone_manager.current_scene_id,
                donor_id=track_id, 
                details=f"Operador atribuiu portador ao track #{track_id}"
            )

    def toggle_facilitator(self, track_id: int):
        """Alterna papel de facilitador para um track_id."""
        for trk in self.tracker.tracks:
            if trk.track_id == track_id:
                trk.role = "facilitador" if trk.role != "facilitador" else "participante"
                if self.logger:
                    self.logger.log_event(
                        event_type="operator_mark_role",
                        scene_id=self.zone_manager.current_scene_id,
                        donor_id=track_id,
                        details=f"Role alterado para {trk.role}"
                    )
                break

    @staticmethod
    def _bbox_iou(box1: List[float], box2: List[float]) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = max(1e-6, (box1[2] - box1[0]) * (box1[3] - box1[1]))
        area2 = max(1e-6, (box2[2] - box2[0]) * (box2[3] - box2[1]))
        return float(inter / (area1 + area2 - inter))

    def _is_plush_ghost_pose(self, pose_det: Dict, plush_candidates: List[Dict], kpts_3d: Optional[np.ndarray]) -> bool:
        """
        Supressão de detecção de pose duplicada na pelúcia:
        Verifica se a pose humana detectada é na verdade a própria pelúcia (boneco/urso)
        com base na sobreposição de bounding box e dimensão métrica ínfima de tronco (< 25 cm).
        """
        p_box = pose_det.get("bbox", [0, 0, 0, 0])
        for pc in plush_candidates:
            c_box = pc.get("bbox", [0, 0, 0, 0])
            iou = self._bbox_iou(p_box, c_box)
            if iou > 0.30:
                # Se houver sobreposição significativa com a pelúcia:
                if kpts_3d is not None:
                    conf = kpts_3d[:, 3]
                    # Se detectou ombro e quadril, avalia altura métrica do tronco
                    if conf[5] > 0.2 and conf[11] > 0.2:
                        torso_h = float(np.linalg.norm(kpts_3d[5, :3] - kpts_3d[11, :3]))
                        if torso_h < 0.25:
                            return True
                        # Se o tronco for de tamanho humano (> 0.25m), é uma pessoa real segurando a pelúcia
                        continue
                    # Se não há pontos anatômicos confiáveis de tronco, é pose espúria na pelúcia
                    return True
                # Sem 3D, se IoU for muito alto (> 0.60), é fantasma
                if iou > 0.60:
                    return True
        return False

    def step(self, remote_detections: Optional[List[Dict]] = None) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Executa um passo completo do pipeline para um novo frame do sensor.
        Retorna (sucesso, canvas_dashboard).
        """
        if not self.sensor.update():
            return False, None

        t_capture = time.time()
        t_capture_ms = int(t_capture * 1000)
        self.frame_idx += 1

        # Cálculo de FPS dinâmico por EMA
        t_now = time.perf_counter()
        if self.last_time is not None:
            dt = t_now - self.last_time
            if dt > 0:
                inst_fps = 1.0 / dt
                self.fps_ema = 0.90 * self.fps_ema + 0.10 * inst_fps
        self.last_time = t_now

        color_bgr = self.sensor.last_color_bgr
        orig_shape = color_bgr.shape[:2]

        # Auto-calibração com plano do chão na primeira detecção estável
        if not self.room_calibration.is_calibrated and self.sensor.floor_clip_plane is not None:
            auto_calib = self.sensor.get_auto_calibration()
            if auto_calib is not None:
                self.room_calibration = auto_calib
                self.sensor.set_room_calibration(self.room_calibration)

        # 1. Inferência de IA DirectML
        t0_ai = time.perf_counter()
        pose_dets = []
        obj_dets = []
        gpu_name = "CPU"

        if self.pose_engine is not None:
            blob, ratio, pad = self.pose_engine.preprocess(color_bgr)
            raw_pose = self.pose_engine.run_raw(blob)
            pose_dets = self.pose_engine.postprocess_pose(raw_pose, ratio, pad, orig_shape)
            gpu_name = self.pose_engine.active_provider

            # Detecção de objetos / pelúcia (se modelo disponível)
            if self.object_engine is not None:
                raw_obj = self.object_engine.run_raw(blob)
                obj_dets = self.object_engine.postprocess_objects(
                    raw_obj, ratio, pad, orig_shape, 
                    config.ai.toy_classes, conf_thresh=config.ai.toy_conf_threshold
                )
        t1_ai = time.perf_counter()
        gpu_latency_ms = (t1_ai - t0_ai) * 1000.0

        # 2. Candidatos da Pelúcia
        plush_candidates = []
        for obj in obj_dets:
            bx1, by1, bx2, by2 = obj['bbox']
            cx = (bx1 + bx2) / 2.0
            cy = (by1 + by2) / 2.0
            pt_3d = self.sensor.get_3d_point_from_color(cx, cy, to_room=True)
            if pt_3d is not None and pt_3d[2] > 0.35:
                plush_candidates.append({
                    "class_name": obj['class_name'],
                    "bbox": obj['bbox'],
                    "confidence": obj['confidence'],
                    "pos_3d": pt_3d
                })
        self.last_plush_candidates = plush_candidates

        # 3. Mapeamento 3D das Pessoas (com supressão de pose duplicada na pelúcia)
        person_detections_3d = []
        for p in pose_dets:
            kpts_2d = p['keypoints']
            kpts_3d = self.sensor.unproject_keypoints_3d(kpts_2d, to_room=True)

            # Supressão de pose duplicada na pelúcia
            if self._is_plush_ghost_pose(p, plush_candidates, kpts_3d):
                continue

            # Âncora anatômica alta e estável: tórax e tronco (ombros 5,6 e quadris 11,12)
            # Decisão de engenharia (Issue M1-06): a cabeça/nariz possui movimentos independentes
            # de alta frequência e frequentes oclusões. Ombros e quadris garantem o centro de massa estável.
            center_3d = None
            torso_pts = []
            for idx in [5, 6, 11, 12]:  # ombros e quadris
                if not np.isnan(kpts_3d[idx, 0]) and kpts_3d[idx, 3] > 0.20 and kpts_3d[idx, 2] > 0.35:
                    torso_pts.append(kpts_3d[idx, :3])

            if len(torso_pts) > 0:
                center_3d = np.mean(torso_pts, axis=0)
            else:
                bx1, by1, bx2, by2 = p['bbox']
                fb_pt = self.sensor.get_3d_point_from_color((bx1 + bx2) / 2.0, (by1 + by2) / 2.0, to_room=True)
                if fb_pt is not None:
                    center_3d = np.array(fb_pt)

            if center_3d is not None:
                person_detections_3d.append({
                    "pos_3d": (float(center_3d[0]), float(center_3d[1]), float(center_3d[2])),
                    "keypoints_3d": kpts_3d,
                    "keypoints_2d": kpts_2d,
                    "bbox": p['bbox'],
                    "role": "participante"
                })

        # Adiciona detecções remotas do 2º sensor se houver
        if remote_detections:
            for rd in remote_detections:
                if "keypoints_3d" in rd and isinstance(rd["keypoints_3d"], list):
                    rd["keypoints_3d"] = np.asarray(rd["keypoints_3d"], dtype=np.float32)
            person_detections_3d.extend(remote_detections)

        # 4. Atualização do Rastreador 3D (com dt dinâmico)
        active_tracks = self.tracker.update(person_detections_3d, timestamp_s=t_capture)
        self.last_tracks = active_tracks

        # 5. Avaliação de Zonas e Papéis
        self.last_zone_evaluations = self.zone_manager.evaluate_tracks(active_tracks)

        # Registro de eventos de zona (enter / exit) em events.csv (Issue M1-01)
        if self.logger:
            for z_ev in self.zone_manager.pending_zone_events:
                self.logger.log_event(
                    event_type=z_ev["type"],
                    scene_id=self.zone_manager.current_scene_id,
                    t_capture_ms=t_capture_ms,
                    donor_id=z_ev.get("track_id"),
                    details=z_ev.get("details", "")
                )

        # 6. Classificação Postural e Proxêmica
        for trk in active_tracks:
            trk.posture = PostureClassifier.classify(trk.last_keypoints_3d)
        proxemic_events = ProxemicsAnalyzer.calculate_pairwise(active_tracks)
        self.last_proxemic_events = proxemic_events

        # 6.1 Detecção de F-Formations (Kendon / Cristani et al.)
        best_cand = plush_candidates[0] if plush_candidates else None
        plush_pos_3d = best_cand.get("pos_3d") if best_cand else None
        f_formations = self.f_formation_detector.detect(active_tracks, plush_pos=plush_pos_3d)
        self.last_f_formations = f_formations

        # 7. Inferência de Portador da Pelúcia (Modo A / Modo B)
        holder_info = self.holder_inference.process(
            tracks=active_tracks,
            plush_candidates=plush_candidates,
            timestamp_s=t_capture
        )
        self.last_holder_result = holder_info

        # 8. Métricas de Circulação da Pelúcia (Issue M1-03)
        # Filtra tracks pelo alcance confiável para não contaminar métricas
        valid_metric_tracks = []
        for trk, ev in zip(active_tracks, self.last_zone_evaluations):
            if ev.get("in_reliable_range", True):
                valid_metric_tracks.append(trk)

        current_scene = self.zone_manager.current_scene_id
        self.plush_metrics.update(
            tracks=valid_metric_tracks,
            holder_info=holder_info,
            scene_id=current_scene,
            timestamp_s=t_capture
        )

        # 9. Gravação Científica Schema v3
        if self.logger:
            raw_obs = {
                "n_persons": len(active_tracks),
                "n_plush_candidates": len(plush_candidates),
                "holder_state": holder_info.get("state"),
                "holder_id": holder_info.get("holder_id"),
                "n_f_formations": len(f_formations)
            }
            self.logger.log_frame(
                t_capture_ms=t_capture_ms,
                frame_idx=self.frame_idx,
                tracks=active_tracks,
                holder_info=holder_info,
                scene_id=current_scene,
                raw_observations=raw_obs
            )

        # 10. Renderização do Dashboard com parâmetros visuais completos
        canvas = self.dashboard.render(
            color_bgr=color_bgr,
            tracks=active_tracks,
            toys=plush_candidates,
            proxemic_events=proxemic_events,
            joint_events=[],
            fps=self.fps_ema,
            gpu_latency_ms=gpu_latency_ms,
            view_mode=self.view_mode,
            show_trajectories=self.show_trajectories,
            skeleton_mode=self.skeleton_mode,
            show_hud_help=self.show_hud_help,
            scene_id=f"{current_scene} - {self.zone_manager.current_scene_name}",
            holder_id=holder_info.get("holder_id"),
            selected_track_id=self.selected_track_id,
            gpu_name=gpu_name,
            zones=self.zone_manager.zones
        )

        return True, canvas

    def export_metrics_summary(self, filepath: Optional[str] = None):
        """Exporta resumo estatístico e métricas de circulação em JSON."""
        all_handoffs = []
        for sc_name, sc_info in self.plush_metrics.scenes_data.items():
            all_handoffs.extend(sc_info.get("handoffs", []))

        scenes_summary = {
            sc_id: self.plush_metrics.get_scene_summary(sc_id)
            for sc_id in self.plush_metrics.scenes_data
        }

        summary = {
            "version": config.version,
            "scenes": scenes_summary,
            "handoff_count_total": len(all_handoffs),
            "handoffs": all_handoffs
        }
        if filepath is None:
            if self.logger and hasattr(self.logger, 'session_dir'):
                filepath = os.path.join(self.logger.session_dir, "metrics_summary.json")
            else:
                filepath = os.path.join(config.dataset_output_dir, "metrics_summary.json")
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"[INFO] Resumo de métricas salvo em: {filepath}")
        except Exception as e:
            print(f"[AVISO] Falha ao exportar metrics_summary: {e}")

    def close(self):
        self.export_metrics_summary()
        if self.sensor:
            self.sensor.close()
        if self.logger:
            self.logger.close()
