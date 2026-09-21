"""
Nó Sensor Secundário (Edge Node):
Desenvolvido para rodar no Notebook (ex: com GPU NVIDIA GTX 1060) ou PC secundário com o 2º Kinect v2.
Captura os dados do Kinect, executa inferência local e transmite as coordenadas 3D para o Hub Central.
"""
import sys
import time
import argparse
import numpy as np

from src.core.config import config
from src.core.kinect_sensor import KinectSensor
from src.ai.directml_inference import DirectMLInference
from src.network.network_protocol import EdgeSender

def main():
    parser = argparse.ArgumentParser(description="Nó Sensor Secundário (Kinect #2)")
    parser.add_argument("--hub-ip", type=str, default="192.168.1.100", help="IP do PC Hub na rede local")
    parser.add_argument("--hub-port", type=int, default=5555, help="Porta UDP do PC Hub")
    parser.add_argument("--sensor-id", type=str, default="kinect_node_2", help="Identificador do sensor")
    args = parser.parse_args()

    print("=" * 60)
    print(f"INICIANDO NÓ SENSOR SECUNDÁRIO ({args.sensor_id})")
    print(f"Destino do Hub: {args.hub_ip}:{args.hub_port}")
    print("=" * 60)

    sensor = KinectSensor()
    sender = EdgeSender(hub_ip=args.hub_ip, hub_port=args.hub_port)
    pose_engine = DirectMLInference(config.ai.pose_model_path, conf_thresh=0.40)

    print("Nó ativo e transmitindo coordenadas 3D em tempo real...")
    try:
        while True:
            if not sensor.update():
                time.sleep(0.005)
                continue

            color_bgr = sensor.last_color_bgr
            orig_shape = color_bgr.shape[:2]

            blob, ratio, pad = pose_engine.preprocess(color_bgr)
            raw_pose = pose_engine.run_raw(blob)
            pose_dets = pose_engine.postprocess_pose(raw_pose, ratio, pad, orig_shape)

            person_detections_3d = []
            for p in pose_dets:
                kpts_2d = p['keypoints']
                kpts_3d = sensor.unproject_keypoints_3d(kpts_2d, to_room=True)

                valid_pts = []
                for idx in [5, 6, 11, 12]:  # Ombros e quadris (sem nariz)
                    if kpts_3d[idx, 3] > 0.25 and kpts_3d[idx, 2] > 0.35:
                        valid_pts.append(kpts_3d[idx, :3])

                if len(valid_pts) > 0:
                    center_3d = np.mean(valid_pts, axis=0)
                    person_detections_3d.append({
                        "pos_3d": [float(center_3d[0]), float(center_3d[1]), float(center_3d[2])],
                        "keypoints_3d": kpts_3d.tolist(),
                        "keypoints_2d": kpts_2d,
                        "bbox": p['bbox']
                    })

            sender.send_detections(sensor_id=args.sensor_id, detections=person_detections_3d)

    except KeyboardInterrupt:
        print("\nEncerrando nó sensor...")
    finally:
        sensor.close()
        sender.close()

if __name__ == "__main__":
    main()
