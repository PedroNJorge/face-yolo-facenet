from .yolo_detector import YOLODetector
import cv2
from typing import Optional, List, Tuple
import numpy as np
from torch import Tensor


class FaceExtractor():
    def __init__(self):
        '''
        Initialize FaceExtractor with YOLODetector
        '''
        self.detector = YOLODetector()

    def extract_faces(
            self,
            image: np.ndarray,
            required_size: Tuple[int, int] = (160, 160)
    ) -> Optional[List[Tuple[np.ndarray, Tensor, float]]]:
        '''
        Extract faces from image and resizes them for recognition later

        Args:
            image_path (Str): Target image path
            required_size (Tuple(Int)): Size required for Pytorch FaceNet

        Return:
            face_info: List[Tuple(nd.array, Tensor, float)]
                (face, bbox, conf)
        '''
        detections = self.detector.detect(image)
        if detections is None:
            return None

        face_info = []
        for i, detection in enumerate(detections):
            bbox, conf = detection
            x1, y1, x2, y2 = map(int, bbox)
            face = image[y1:y2, x1:x2]
            resized_face = cv2.resize(face, required_size)

            # Save Face
            print("Saving")
            cv2.imshow('test', resized_face)

            cv2.imwrite(f'data/unknown_faces/face_{i}.jpg', resized_face)
            print("Saved")

            face_info.append((resized_face, bbox, conf))
        return face_info
