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
    object_model_path: str = "weights/yolo11s_plush.onnx" # Modelo de classe única para a pelúcia móvel
    device_id: int = 0 # Dispositivo DirectML explícito (0 = AMD RX 6600 primária)
    img_size: int = 640
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    toy_conf_threshold: float = 0.25 # Limiar robusto para detector ajustado (classe única)

    def __post_init__(self):
        self.pose_model_path = resolve_model_path(self.pose_model_path)
        self.object_model_path = resolve_model_path(self.object_model_path)
    
    # Vocabulário unificado: apenas o artefato móvel do espetáculo
    toy_classes: Dict[int, str] = field(default_factory=lambda: {
        0: "pelucia"
    })

@dataclass
class HolderInferenceConfig:
    wrist_threshold_m: float = 0.45 # Distância máxima punho-pelúcia para considerar pegada
    torso_threshold_m: float = 0.55 # Distância máxima ao eixo do tronco (cobre abraço)
    handoff_window_s: float = 2.0   # Janela máxima para completar passagem de posse
    min_score_margin: float = 0.12  # Margem mínima entre 1º e 2º colocado para exclusividade
    state_timeout_s: float = 1.5    # Timeout sem detecção para transitar para INDETERMINADO

@dataclass
class ProxemicsConfig:
    # Calibrado para crianças e adolescentes (espaços de interação mais compactos)
    intimate_threshold_m: float = 0.35
    personal_threshold_m: float = 1.00
    social_threshold_m: float = 2.50

@dataclass
class TrackingConfig:
    max_distance_threshold: float = 0.85  # Metros para associação de pessoa no 3D
    max_frames_to_keep_lost: int = 30     # 1 segundo a 30 FPS para manter track perdido
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
    version: str = "0.3.0"
    kinect: KinectConfig = field(default_factory=KinectConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    holder: HolderInferenceConfig = field(default_factory=HolderInferenceConfig)
    proxemics: ProxemicsConfig = field(default_factory=ProxemicsConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    dataset_output_dir: str = "recordings"

config = SystemConfig()

