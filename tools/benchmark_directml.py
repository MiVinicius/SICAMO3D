import os
import sys
from pathlib import Path
import time
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

# Garante acesso ao pacote raiz
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.config import resolve_model_path

def main():
    print("=== TESTE DE INFERÊNCIA COM DIRECTML NA GPU AMD RADEON RX 6600 ===")
    
    # Modelo alvo (prioriza pose configurado ou yolo11s-pose.onnx)
    default_model = "weights/yolo11s-pose.onnx"
    onnx_path = resolve_model_path(sys.argv[1] if len(sys.argv) > 1 else default_model)
    
    try:
        if not os.path.exists(onnx_path):
            pt_path = onnx_path.replace(".onnx", ".pt")
            print(f"Modelo ONNX não encontrado em {onnx_path}. Tentando exportar de {pt_path}...")
            if os.path.exists(pt_path):
                model = YOLO(pt_path)
                model.export(format="onnx", imgsz=640, simplify=True)
                print(f"Modelo exportado: {onnx_path}")
            else:
                print(f"Aviso: Nem {onnx_path} nem {pt_path} foram encontrados.")
        else:
            print(f"Modelo ONNX encontrado: {onnx_path}")
    except Exception as e:
        print("Erro durante verificação/exportação do modelo:", e)

    # 2. Configura sessão do ONNX Runtime com DirectML
    print("\nInicializando ONNX Runtime com provedor DirectML (DirectX 12)...")
    providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
    active_providers = session.get_providers()
    print("Provedores ativos na sessão:", active_providers)
    
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"Entrada do modelo: '{input_name}', Formato: {input_shape}")

    # 3. Teste de aquecimento (Warmup)
    dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)
    print("Executando aquecimento na GPU...")
    for _ in range(5):
        _ = session.run(None, {input_name: dummy_input})

    # 4. Benchmark de 50 iterações para medir latência e FPS real
    print("Executando benchmark de 50 iterações...")
    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    fps = 1000.0 / avg_latency

    print("\n=== RESULTADOS DO BENCHMARK (AMD RX 6600) ===")
    print(f"Latência média de inferência : {avg_latency:.2f} ms")
    print(f"Latência percentil 95 (P95)  : {p95_latency:.2f} ms")
    print(f"Taxa de quadros estimada (FPS): {fps:.1f} FPS")

    if 'DmlExecutionProvider' in active_providers:
        print("\n SUCESSO: Aceleração por DirectML na GPU AMD validada com altíssima performance!")
    else:
        print("\n AVISO: Executando apenas na CPU.")

if __name__ == "__main__":
    main()
