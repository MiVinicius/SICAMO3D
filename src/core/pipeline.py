"""
Pipeline Central do Sistema Socioenativo 3D (SICAMO3D v0.2.0).
Organiza o fluxo modular de execução:
Captura -> Inferência DirectML -> Projeção na Sala -> Tracking 3D ->
Zonas/Cenas -> Inferência de Portador -> Métricas de Circulação -> Log v3 -> Renderização.
"""
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
        self.pose_engine = DirectMLInference(
            config.ai.pose_model_path, 
            conf_thresh=config.ai.conf_threshold,
            device_id=config.ai.device_id
        )
        self.object_engine = DirectMLInference(
            config.ai.object_model_path, 
            conf_thresh=config.ai.toy_conf_threshold,
            device_id=config.ai.device_id
        )

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
            timeout_s=config.holder.state_timeout_s
        )
        self.plush_metrics = PlushMetricsAnalyzer()
        self.plush_metrics.set_scene(self.zone_manager.current_scene_id)

        # 6. Gravador Científico Schema v3
        self.logger = ScientificLogger(output_dir=config.dataset_output_dir, session_name=session_name) if enable_logger else None

        # 7. Dashboard 3D
        self.dashboard = Dashboard3D(
            room_width_m=self.zone_manager.room_info.get("width_m", 6.0),
            room_depth_m=self.zone_manager.room_info.get("depth_m", 6.0)
        )

        self.frame_idx = 0
        self.last_holder_result: Dict[str, Any] = {"state": "INDETERMINADO", "holder_id": None, "confidence": 0.0}
        self.last_tracks: List = []
        self.last_plush_candidates: List[Dict] = []
        self.last_proxemic_events: List[Dict] = []
        self.last_joint_events: List[Dict] = []

    def advance_scene(self) -> str:
        """Avança cena no ZoneManager e PlushMetrics (Atalho 'N')."""
        new_scene = self.zone_manager.next_scene()
        self.plush_metrics.set_scene(new_scene)
        if self.logger:
            self.logger.log_event(event_type="scene_change", scene_id=new_scene, details=f"Mudou para {new_scene}")
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
        blob, ratio, pad = self.pose_engine.preprocess(color_bgr)
        raw_pose = self.pose_engine.run_raw(blob)
        pose_dets = self.pose_engine.postprocess_pose(raw_pose, ratio, pad, orig_shape)

        # Detecção de objetos / pelúcia
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

        # 3. Mapeamento 3D das Pessoas (Sem o bloco is_doll!)
        person_detections_3d = []
        for p in pose_dets:
            kpts_2d = p['keypoints']
            kpts_3d = self.sensor.unproject_keypoints_3d(kpts_2d, to_room=True)

            # Âncora anatômica alta e estável: tórax ou ombros
            center_3d = None
            torso_pts = []
            for idx in [5, 6, 11, 12]: # ombros e quadris
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
            person_detections_3d.extend(remote_detections)

        # 4. Atualização do Rastreador 3D (com dt dinâmico)
        active_tracks = self.tracker.update(person_detections_3d, timestamp_s=t_capture)
        self.last_tracks = active_tracks

        # 5. Avaliação de Zonas e Papéis
        self.zone_manager.evaluate_tracks(active_tracks)

        # 6. Classificação Postural e Proxêmica
        for trk in active_tracks:
            trk.posture = PostureClassifier.classify(trk.last_keypoints_3d)
        proxemic_events = ProxemicsAnalyzer.calculate_pairwise(active_tracks)
        self.last_proxemic_events = proxemic_events

        # 7. Inferência de Portador da Pelúcia (Modo A / Modo B)
        holder_info = self.holder_inference.process(
            tracks=active_tracks,
            plush_candidates=plush_candidates,
            timestamp_s=t_capture
        )
        self.last_holder_result = holder_info

        # 8. Métricas de Circulação da Pelúcia
        current_scene = self.zone_manager.current_scene_id
        self.plush_metrics.update(
            tracks=active_tracks,
            holder_info=holder_info,
            scene_id=current_scene,
            timestamp_s=t_capture
        )

        # 9. Gravação Científica Schema v3
        if self.logger:
            # Prepara log bruto compacto anonimizado
            raw_obs = {
                "n_persons": len(active_tracks),
                "n_plush_candidates": len(plush_candidates),
                "holder_state": holder_info.get("state"),
                "holder_id": holder_info.get("holder_id")
            }
            self.logger.log_frame(
                t_capture_ms=t_capture_ms,
                frame_idx=self.frame_idx,
                tracks=active_tracks,
                holder_info=holder_info,
                scene_id=current_scene,
                raw_observations=raw_obs
            )

        # 10. Renderização do Dashboard
        fps = 30.0
        gpu_name = self.pose_engine.active_provider
        canvas = self.dashboard.render(
            color_bgr=color_bgr,
            tracks=active_tracks,
            toys=plush_candidates,
            proxemic_events=proxemic_events,
            joint_events=[],
            fps=fps,
            gpu_latency_ms=gpu_latency_ms,
            scene_id=f"{current_scene} - {self.zone_manager.current_scene_name}",
            holder_id=holder_info.get("holder_id"),
            gpu_name=gpu_name,
            zones=self.zone_manager.zones
        )

        return True, canvas

    def close(self):
        if self.sensor:
            self.sensor.close()
        if self.logger:
            self.logger.close()
