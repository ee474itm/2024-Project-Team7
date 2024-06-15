from ultralytics import YOLO
import torch


class Detector:
    def __init__(self):
        # Load a pretrained YOLOv8n model
        self.model = YOLO("./weights/best.pt")

    @torch.no_grad()
    def run(self, paths):
        if isinstance(paths, str):
            paths = [paths]
        
        results = []
        for path in paths:
            result = self.model.predict(path, verbose=False)
            detected_classes = result[0].boxes.cls
            detected_class_names = [result[0].names[int(cls_id)] for cls_id in detected_classes]
            results.append(detected_class_names)
            # ['Airbag Warning', 'Stability Control Off']
        return results

