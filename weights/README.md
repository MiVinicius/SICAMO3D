# Modelos Neurais e Pesos (Weights)

Este diretório armazena os modelos de aprendizado profundo utilizados pelo sistema.
Devido ao tamanho dos arquivos binários (alguns excedem 100 MB), eles são ignorados pelo Git (`.gitignore`).

## Modelos Utilizados

| Arquivo | Descrição | Formato | Backend |
| :--- | :--- | :--- | :--- |
| `yolo11s-pose.onnx` | Rastreamento anatômico de 17 keypoints humanos (COCO) | ONNX | DirectML / GPU |
| `yolo11s_plush.onnx` | Detector customizado mono-classe de alta precisão para a pelúcia do projeto | ONNX | DirectML / GPU |
| `yolov8m-worldv2.onnx` | Detecção de brinquedos e objetos com vocabulário aberto customizado (fallback) | ONNX | DirectML / GPU |

## Como Obter / Exportar os Modelos

### 1. Download e exportação automática dos modelos base
Para baixar e exportar automaticamente os modelos base:

```powershell
python tools/download_weights.py
```

### 2. Treinamento e exportação do detector customizado da pelúcia
Para treinar a YOLO11 mono-classe especializada com o dataset local rotulado e exportá-la para `weights/yolo11s_plush.onnx`:

```powershell
python tools/train_plush_detector.py --data datasets/plush_dataset.yaml --epochs 50 --imgsz 640
```

