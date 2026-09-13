"""
Configurações globais do sistema de rastreamento socioenativo 3D.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

@dataclass
class KinectConfig:
    color_width: int = 1920
    color_height: int = 1080
    depth_width: int = 512
    depth_height: int = 424
    fps: int = 30
    min_depth_mm: int = 500   # 0.5 metros
    max_depth_mm: int = 8000  # 8.0 metros

import os
from pathlib import Path

def resolve_model_path(model_path: str) -> str:
    """
    Resolve o caminho de um modelo ONNX/PyTorch verificando o caminho direto,
    a pasta weights/ e a raiz do projeto.
    """
    if not model_path:
        return model_path
    
    if os.path.exists(model_path):
        return model_path
    
    project_root = Path(__file__).resolve().parent.parent.parent
    candidate_weights = project_root / "weights" / os.path.basename(model_path)
    if candidate_weights.exists():
        return str(candidate_weights)
        
    candidate_root = project_root / os.path.basename(model_path)
    if candidate_root.exists():
        return str(candidate_root)
        
    return model_path

@dataclass
class AIConfig:
    pose_model_path: str = "weights/yolo11s-pose.onnx"
    object_model_path: str = "weights/yolov8m-worldv2.onnx" # Modelo YOLO-World Medium (Fase 3: Backbone Aprimorado)
    img_size: int = 640
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    toy_conf_threshold: float = 0.08 # Limiar permissivo para capturar Toad e brinquedos

    def __post_init__(self):
        self.pose_model_path = resolve_model_path(self.pose_model_path)
        self.object_model_path = resolve_model_path(self.object_model_path)
    
    # Classes ricas do YOLO-World Medium (Fase 3)
    toy_classes: Dict[int, str] = field(default_factory=lambda: {
        0: "boneco_toad",    # 'toad doll'
        1: "boneco_toad",    # 'toad plush'
        2: "pelucia",        # 'plush doll'
        3: "pelucia",        # 'stuffed animal'
        4: "action_figure",  # 'action figure'
        5: "boneco",         # 'doll'
        6: "brinquedo",      # 'toy'
        7: "carrinho",       # 'toy car'
        8: "bola"            # 'ball'
    })

@dataclass
class TrackingConfig:
    max_distance_threshold: float = 0.85  # Metros para associação de pessoa no 3D
    max_frames_to_keep_lost: int = 30     # 1 segundo a 30 FPS para manter track perdido
    proximity_toy_threshold_m: float = 0.40 # 40 cm da mão para considerar posse de brinquedo
    keypoint_smoothing_alpha: float = 0.65  # Suavização temporal (elimina trepidação)

@dataclass
class NetworkConfig:
    hub_ip: str = "0.0.0.0"
    hub_port: int = 5555
    edge_remote_ip: str = "127.0.0.1"
    edge_remote_port: int = 5555
    osc_ip: str = "127.0.0.1"
    osc_port: int = 9000

@dataclass
class SystemConfig:
    version: str = "0.1.0"
    kinect: KinectConfig = field(default_factory=KinectConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    dataset_output_dir: str = "recordings"

config = SystemConfig()
