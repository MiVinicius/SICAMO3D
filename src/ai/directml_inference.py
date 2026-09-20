"""
Motor de inferência de IA acelerado na GPU AMD com DirectML (DirectX 12).
"""
import time
import numpy as np
import cv2
import onnxruntime as ort
from typing import List, Dict, Tuple, Optional

class DirectMLInference:
    def __init__(self, model_path: str, conf_thresh: float = 0.40, iou_thresh: float = 0.45, device_id: Optional[int] = None):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device_id = device_id if device_id is not None else 0
        
        # Provedor DirectML com device_id explícito (suporta GPU primária ou secundária)
        providers = [('DmlExecutionProvider', {'device_id': self.device_id}), 'CPUExecutionProvider']
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = True
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.intra_op_num_threads = 4
        
        self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
        self.active_provider = self.session.get_providers()[0]

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape # [1, 3, 640, 640]
        self.input_h = self.input_shape[2]
        self.input_w = self.input_shape[3]

        # Buffer reutilizável pré-alocado para evitar alocações de memória repetidas a cada frame
        self.padded_buffer = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)

    def preprocess(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Redimensiona com letterbox para 640x640 de alta performance com C++ SIMD (cv2.dnn).
        Retorna (tensor_chw, ganho, padding (pad_w, pad_h)).
        """
        h0, w0 = img_bgr.shape[:2]
        r = min(self.input_h / h0, self.input_w / w0)
        new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
        pad_w = (self.input_w - new_w) // 2
        pad_h = (self.input_h - new_h) // 2

        # Limpa bordas do buffer apenas se a proporção mudou
        if (new_w, new_h) != (self.input_w, self.input_h):
            self.padded_buffer.fill(114)

        # Redimensionamento direto no buffer pré-alocado
        cv2.resize(img_bgr, (new_w, new_h), dst=self.padded_buffer[pad_h:pad_h + new_h, pad_w:pad_w + new_w], interpolation=cv2.INTER_LINEAR)

        # Conversão C++ SIMD ultra-rápida (BGR -> RGB, CHW, Float32 / 255.0)
        blob = cv2.dnn.blobFromImage(self.padded_buffer, 1.0 / 255.0, (self.input_w, self.input_h), swapRB=True)
        return blob, r, (pad_w, pad_h)

    def run_raw(self, blob: np.ndarray) -> np.ndarray:
        """Executa a inferência direta na GPU AMD."""
        outputs = self.session.run(None, {self.input_name: blob})
        return outputs[0]

    def postprocess_pose(self, raw_output: np.ndarray, ratio: float, pad: Tuple[int, int], orig_shape: Tuple[int, int]) -> List[Dict]:
        """
        Decodifica a saída do YOLO11-Pose: [1, 56, 8400]
        Onde 56 = 4 (bbox: cx, cy, w, h) + 1 (conf pessoa) + 17*3 (17 keypoints: x, y, conf).
        """
        output = raw_output[0] # [56, 8400]
        output = output.transpose(1, 0) # [8400, 56]
        
        pad_w, pad_h = pad
        orig_h, orig_w = orig_shape

        boxes = []
        confidences = []
        keypoints_list = []

        # Filtra por confiança de pessoa
        scores = output[:, 4]
        mask = scores > self.conf_thresh
        filtered = output[mask]

        if len(filtered) == 0:
            return []

        cx = filtered[:, 0]
        cy = filtered[:, 1]
        w = filtered[:, 2]
        h = filtered[:, 3]

        # Converte para x1, y1, w, h na escala original de forma vetorizada
        x1 = (cx - w * 0.5 - pad_w) / ratio
        y1 = (cy - h * 0.5 - pad_h) / ratio
        w_orig = w / ratio
        h_orig = h / ratio

        # Extração vetorizada de keypoints para todos os candidatos
        raw_kpts = filtered[:, 5:].reshape((-1, 17, 3))
        kpts_all = np.zeros_like(raw_kpts)
        kpts_all[:, :, 0] = (raw_kpts[:, :, 0] - pad_w) / ratio
        kpts_all[:, :, 1] = (raw_kpts[:, :, 1] - pad_h) / ratio
        kpts_all[:, :, 2] = raw_kpts[:, :, 2]

        boxes = np.column_stack((x1, y1, w_orig, h_orig)).tolist()
        confidences = filtered[:, 4].tolist()

        # Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thresh, self.iou_thresh)
        results = []
        if len(indices) > 0:
            for idx in indices.flatten():
                bx, by, bw, bh = boxes[idx]
                results.append({
                    "bbox": [bx, by, bx + bw, by + bh],
                    "confidence": confidences[idx],
                    "keypoints": kpts_all[idx] # (17, 3)
                })
        return results

    def postprocess_objects(self, 
                            raw_output: np.ndarray, 
                            ratio: float, 
                            pad: Tuple[int, int], 
                            orig_shape: Tuple[int, int], 
                            target_classes: Dict[int, str],
                            conf_thresh: Optional[float] = None) -> List[Dict]:
        """
        Decodifica a saída do YOLO11-Object / YOLOv8-World: [1, N, 8400]
        com filtragem vetorizada C/NumPy de alto desempenho.
        """
        actual_conf_thresh = conf_thresh if conf_thresh is not None else self.conf_thresh
        output = raw_output[0] # [N, 8400]
        output = output.transpose(1, 0) # [8400, N]

        pad_w, pad_h = pad
        class_scores = output[:, 4:] # [8400, num_classes]
        max_scores = np.max(class_scores, axis=1)
        best_classes = np.argmax(class_scores, axis=1)

        valid_mask = max_scores >= actual_conf_thresh
        if not np.any(valid_mask):
            return []

        v_output = output[valid_mask]
        v_scores = max_scores[valid_mask]
        v_classes = best_classes[valid_mask]

        # Filtra apenas as classes de interesse (definidas em target_classes)
        target_keys = set(target_classes.keys()) if target_classes is not None else None
        if target_keys is not None:
            class_filter = np.array([c in target_keys for c in v_classes], dtype=bool)
            if not np.any(class_filter):
                return []
            v_output = v_output[class_filter]
            v_scores = v_scores[class_filter]
            v_classes = v_classes[class_filter]

        cx = v_output[:, 0]
        cy = v_output[:, 1]
        w = v_output[:, 2]
        h = v_output[:, 3]

        x1 = (cx - w * 0.5 - pad_w) / ratio
        y1 = (cy - h * 0.5 - pad_h) / ratio
        w_orig = w / ratio
        h_orig = h / ratio

        boxes = np.column_stack((x1, y1, w_orig, h_orig)).tolist()
        confidences = v_scores.tolist()
        class_ids = v_classes.tolist()

        indices = cv2.dnn.NMSBoxes(boxes, confidences, actual_conf_thresh, self.iou_thresh)
        results = []
        if len(indices) > 0:
            for idx in indices.flatten():
                bx, by, bw, bh = boxes[idx]
                cid = class_ids[idx]
                name = target_classes.get(cid, f"objeto_{cid}") if target_classes is not None else f"objeto_{cid}"
                results.append({
                    "bbox": [bx, by, bx + bw, by + bh],
                    "confidence": confidences[idx],
                    "class_id": cid,
                    "class_name": name
                })
        return results
