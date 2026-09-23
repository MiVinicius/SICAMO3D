"""
Script de Treinamento e Exportação do Detector de Pelúcia de Classe Única (Issue M2-02).
Projetado para rodar em GPU NVIDIA (Local, Google Colab ou Kaggle) e gerar o modelo
otimizado em formato ONNX para execução em tempo real via DirectML no SICAMO3D.

Uso:
    python tools/train_plush_detector.py --data dataset_plush/dataset.yaml --epochs 50 --model yolo11s.pt
"""
import os
import sys
import argparse
from pathlib import Path

def train_and_export(data_yaml: str, 
                     model_name: str = "yolo11s.pt", 
                     epochs: int = 50, 
                     imgsz: int = 640, 
                     batch_size: int = 16,
                     output_onnx: str = "weights/yolo11s_plush.onnx"):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERRO] Biblioteca 'ultralytics' não encontrada.")
        print("Instale com: pip install ultralytics")
        sys.exit(1)

    print("=" * 70)
    print("TREINAMENTO DO DETECTOR DE CLASSE ÚNICA — SICAMO3D")
    print(f"Modelo base: {model_name}")
    print(f"Configuração do dataset: {data_yaml}")
    print(f"Épocas: {epochs} | Tamanho da Imagem: {imgsz} | Batch Size: {batch_size}")
    print(f"Destino ONNX: {output_onnx}")
    print("=" * 70)

    # 1. Carrega modelo base pré-treinado
    model = YOLO(model_name)

    # 2. Treinamento com congelamento de backbone inicial ou fine-tuning completo
    print("\nIniciando treinamento com fine-tuning...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        single_cls=True,  # Força tratamento estrito como classe única
        project="runs_plush",
        name="plush_detector",
        save=True,
        plots=True
    )

    # 3. Exportação para formato ONNX com simplificação
    print("\nExportando modelo para formato ONNX com DirectML opset 17...")
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = Path(model_name)

    exported_path = model.export(
        format="onnx",
        imgsz=imgsz,
        simplify=True,
        opset=17,
        dynamic=False
    )
    print(f"Modelo exportado em: {exported_path}")

    # 4. Copia para o diretório de pesos oficial do projeto
    dest_path = Path(output_onnx)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(exported_path, str(dest_path))
    print(f"\n[SUCESSO] Modelo salvo com sucesso em: {dest_path.resolve()}")
    print("O SICAMO3D agora está pronto para carregar e operar no Modo B!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina e exporta o detector da pelúcia (YOLO11 ONNX)")
    parser.add_argument("--data", type=str, default="dataset_plush/dataset.yaml", help="Caminho do dataset.yaml")
    parser.add_argument("--model", type=str, default="yolo11s.pt", help="Modelo base (yolo11n.pt ou yolo11s.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas")
    parser.add_argument("--imgsz", type=int, default=640, help="Resolução das imagens")
    parser.add_argument("--batch", type=int, default=16, help="Tamanho do batch")
    parser.add_argument("--output", type=str, default="weights/yolo11s_plush.onnx", help="Caminho de saída do ONNX")
    args = parser.parse_args()

    train_and_export(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        output_onnx=args.output
    )
