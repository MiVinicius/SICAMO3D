"""
Protocolo e receptor de rede para comunicação entre os nós sensores (Notebook) e o Hub central.
"""
import socket
import json
import threading
from typing import List, Dict, Optional

class HubReceiver:
    def __init__(self, host: str = "0.0.0.0", port: int = 5555):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.host, self.port))
        self.sock.settimeout(0.01) # Non-blocking curto
        
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self._latest_detections: List[Dict] = []
        self._lock = threading.Lock()

    def start(self):
        self.is_running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def _listen_loop(self):
        while self.is_running:
            try:
                data, _ = self.sock.recvfrom(65535)
                packet = json.loads(data.decode('utf-8'))
                dets = packet.get("detections", [])
                with self._lock:
                    self._latest_detections = dets
            except socket.timeout:
                continue
            except Exception:
                continue

    def get_latest_detections(self) -> List[Dict]:
        with self._lock:
            dets = list(self._latest_detections)
            self._latest_detections = [] # Consome o buffer
            return dets

    def stop(self):
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=0.5)
        self.sock.close()


class EdgeSender:
    def __init__(self, hub_ip: str = "127.0.0.1", hub_port: int = 5555):
        self.hub_ip = hub_ip
        self.hub_port = hub_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_detections(self, sensor_id: str, detections: List[Dict]):
        packet = {
            "sensor_id": sensor_id,
            "detections": detections
        }
        data = json.dumps(packet).encode('utf-8')
        try:
            self.sock.sendto(data, (self.hub_ip, self.hub_port))
        except Exception:
            pass

    def close(self):
        self.sock.close()
