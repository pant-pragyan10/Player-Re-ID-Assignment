import torch
from ultralytics import YOLO
import numpy as np

class Detector:
    def __init__(self, weights_path, device='cpu', threshold=0.2):
        self.model = YOLO(weights_path)
        self.device = device
        self.model.to(device)
        self.threshold = threshold

    def detect(self, frame):
        # Run YOLOv11 inference
        results = self.model(frame)
        detections = []
        for result in results:
            for box in result.boxes:
                # Only keep 'person' class (class 0 in COCO)
                if int(box.cls) == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0].item())
                    if conf >= self.threshold:
                        detections.append([x1, y1, x2, y2, conf])
        return np.array(detections)  # [N, 5] array 