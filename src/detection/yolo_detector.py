from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import numpy as np
from typing import Optional, List, Tuple
from torch import Tensor

MODEL_PATH = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection",
                             filename="model.pt",
                             local_dir="data/models",
                             )


class YOLODetector():
    def __init__(
            self,
            model_path: str = MODEL_PATH,
            conf_thresh: float = 0.5
    ):
        '''
        Initialize YOLO face detector

        Args:
            model_path (str): path to the model to be used
            conf_thresh (float): threshold for detection face
        '''
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh

    def detect(self, image: np.ndarray) -> Optional[List[Tuple[Tensor, float]]]:
        '''
        Detect faces in image

        Args:
            image (ndarray): image to detect faces

        Return:
            detections (tuple): (box_tensor, conf)
                    box_tensor (Torch Tensor): Contains face coord in xyxy format
                    conf (float): confidence_threshold
            None: if it couldn't detect a face
        '''
        results = self.model(image, conf=self.conf_thresh)
        if not results:
            return None

        detections = []
        for r in results:  # one per image batch
            for i in range(len(r.boxes)):
                box = r.boxes.xyxy[i]
                conf = r.boxes.conf[i]
                detections.append((box, conf))

        print(f"Found {len(detections)} faces")
        return detections
