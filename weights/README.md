# Modelos Neurais e Pesos (Weights)

Este diretório armazena os modelos de aprendizado profundo utilizados pelo sistema.
Devido ao tamanho dos arquivos binários (alguns excedem 100 MB), eles são ignorados pelo Git (`.gitignore`).

## Modelos Utilizados

| Arquivo | Descrição | Formato | Backend |
| :--- | :--- | :--- | :--- |
| `yolo11s-pose.onnx` | Rastreamento anatômico de 17 keypoints humanos (COCO) | ONNX | DirectML / GPU |
| `yolov8m-worldv2.onnx` | Detecção de brinquedos e objetos com vocabulário aberto customizado | ONNX | DirectML / GPU |

## Como Obter / Exportar os Modelos Automaticamente

Para baixar e exportar automaticamente os modelos para este diretório, execute o script utilitário da raiz do projeto:

```powershell
python tools/download_weights.py
```

O script fará o download dos pesos base oficiais e a exportação simplificada para o formato `.onnx`.
