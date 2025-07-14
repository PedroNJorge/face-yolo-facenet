from .yolo_detector import YOLODetector
import cv2
from typing import Union, List, Tuple
import numpy as np


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
    ) -> Union[List[Tuple[np.ndarray, float]], None]:
        '''
        Extract faces from image and resizes them for recognition later

        Args:
            image_path (Str): Target image path
            required_size (Tuple(Int)): Size required for Pytorch FaceNet

        Return:
            faces: List[Tuple(nd.array, float)]
        '''
        detections = self.detector.detect(image)
        if detections is None:
            return None

        faces = []
        for i, detection in enumerate(detections):
            box_tensor, conf = detection
            x1, y1, x2, y2 = map(int, box_tensor)
            face = image[y1:y2, x1:x2]
            resized_face = cv2.resize(face, required_size)

            # Save Face
            print("Saving")
            cv2.imshow('test', resized_face)

            cv2.imwrite(f'data/unknown_faces/face_{i}.jpg', resized_face)
            print("Saved")

            faces.append((resized_face, conf))
        return faces
