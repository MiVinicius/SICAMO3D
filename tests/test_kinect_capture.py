import time
import numpy as np
import cv2
from pykinect2 import PyKinectV2, PyKinectRuntime

def main():
    print("Iniciando sensor Kinect v2...")
    sources = (PyKinectV2.FrameSourceTypes_Color | 
               PyKinectV2.FrameSourceTypes_Depth | 
               PyKinectV2.FrameSourceTypes_Body)
    kinect = PyKinectRuntime.PyKinectRuntime(sources)

    print("Aguardando frames (RGB e Profundidade)...")
    start_time = time.time()
    color_received = False
    depth_received = False
    body_received = False

    # Tenta receber frames por até 10 segundos
    while time.time() - start_time < 10.0:
        if not color_received and kinect.has_new_color_frame():
            frame = kinect.get_last_color_frame()
            if frame is not None and len(frame) > 0:
                # 1920 x 1080 x 4 (RGBA)
                try:
                    color_img = frame.reshape((1080, 1920, 4)).astype(np.uint8)
                    print(f"[SUCESSO] Frame de Cor recebido: shape={color_img.shape}, dtype={color_img.dtype}")
                    # Salva uma amostra para validação
                    bgr_sample = cv2.cvtColor(color_img, cv2.COLOR_RGBA2BGR)
                    cv2.imwrite("sample_kinect_color.jpg", bgr_sample)
                    print(" Amostra salva em sample_kinect_color.jpg")
                    color_received = True
                except Exception as e:
                    print("Erro ao processar frame de cor:", e)

        if not depth_received and kinect.has_new_depth_frame():
            frame = kinect.get_last_depth_frame()
            if frame is not None and len(frame) > 0:
                # 512 x 424 (distância em milímetros, uint16)
                try:
                    depth_img = frame.reshape((424, 512)).astype(np.uint16)
                    print(f"[SUCESSO] Frame de Profundidade recebido: shape={depth_img.shape}, min={depth_img.min()}mm, max={depth_img.max()}mm")
                    # Salva uma visualização normalizada
                    norm_depth = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    colored_depth = cv2.applyColorMap(norm_depth, cv2.COLORMAP_JET)
                    cv2.imwrite("sample_kinect_depth.jpg", colored_depth)
                    print(" Amostra salva em sample_kinect_depth.jpg")
                    depth_received = True
                except Exception as e:
                    print("Erro ao processar frame de profundidade:", e)

        if color_received and depth_received:
            print("\nTODOS OS FLUXOS ESSENCIAIS FORAM RECEBIDOS COM SUCESSO!")
            break

        time.sleep(0.01)

    kinect.close()
    if not (color_received and depth_received):
        print("\nAtenção: Tempo limite atingido. Cor:", color_received, "Profundidade:", depth_received)

if __name__ == "__main__":
    main()
