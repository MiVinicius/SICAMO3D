"""
Utilitário para download e exportação dos modelos ONNX necessários para o sistema.
Execução recomendada após clonar o repositório:
    python tools/download_weights.py
"""
import os
import sys
from pathlib import Path

# Raiz do projeto
ROOT_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT_DIR / "weights"

def prepare_weights():
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(str(WEIGHTS_DIR))
    
    print("=== PREPARAÇÃO DE MODELOS DE REDE NEURAL ===")
    print(f"Diretório de destino: {WEIGHTS_DIR}\n")
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERRO] A biblioteca 'ultralytics' não está instalada.")
        print("Execute: pip install ultralytics")
        sys.exit(1)
        
    # 1. YOLO11s Pose (17 Articulações)
    pose_onnx = WEIGHTS_DIR / "yolo11s-pose.onnx"
    if not pose_onnx.exists():
        print("[1/2] Baixando e exportando YOLO11s-Pose para ONNX...")
        try:
            model_pose = YOLO("yolo11s-pose.pt")
            model_pose.export(format="onnx", imgsz=640, simplify=True)
            print(f" -> Concluído: {pose_onnx.name}\n")
        except Exception as e:
            print(f"[ERRO] Falha ao exportar YOLO11s-Pose: {e}\n")
    else:
        print(f"[1/2] YOLO11s-Pose já presente: {pose_onnx.name}")

    # 2. YOLO11s Custom Plush Detector (Single-Class: pelucia)
    plush_onnx = WEIGHTS_DIR / "yolo11s_plush.onnx"
    if not plush_onnx.exists():
        print("[2/3] Modelo customizado da pelúcia (yolo11s_plush.onnx) não encontrado localmente.")
        print("      -> Para treinar e exportar a partir do seu dataset rotulado:")
        print("         python tools/train_plush_detector.py --data datasets/plush_dataset.yaml --epochs 50")
        print("      -> Ou copie os pesos pré-treinados 'yolo11s_plush.onnx' diretamente para esta pasta.\n")
    else:
        print(f"[2/3] YOLO11s-Plush (custom) já presente: {plush_onnx.name}")

    # 3. YOLOv8m World v2 (Objetos / Brinquedos em Vocabulário Aberto - Fallback)
    world_onnx = WEIGHTS_DIR / "yolov8m-worldv2.onnx"
    if not world_onnx.exists():
        print("[3/3] Baixando e exportando YOLOv8m-Worldv2 para ONNX com vocabulário customizado...")
        try:
            model_world = YOLO("yolov8m-worldv2.pt")
            # Vocabulário científico para IHC e análise socioenativa
            classes = [
                "toad doll",
                "toad plush",
                "plush doll",
                "stuffed animal",
                "action figure",
                "doll",
                "toy",
                "toy car",
                "ball"
            ]
            model_world.set_classes(classes)
            model_world.export(format="onnx", imgsz=640, simplify=True)
            print(f" -> Concluído: {world_onnx.name}\n")
        except Exception as e:
            print(f"[ERRO] Falha ao exportar YOLOv8m-Worldv2: {e}\n")
    else:
        print(f"[3/3] YOLOv8m-Worldv2 já presente: {world_onnx.name}")

    print("=== TODOS OS MODELOS FORAM VERIFICADOS COM SUCESSO! ===")

if __name__ == "__main__":
    prepare_weights()
