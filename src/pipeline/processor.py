from ..detection import FaceExtractor
from ..recognition import FaceEncoder
from ..database import RedisClient
import numpy as np
from typing import Optional, List, Dict


class Processor:
    def __init__(
            self,
            face_extractor: FaceExtractor,
            face_encoder: FaceEncoder,
            redis_client: RedisClient,
    ):
        self.face_extractor = face_extractor
        self.face_encoder = face_encoder
        self.redis = redis_client

    def process_frame(self, frame: np.ndarray) -> Optional[List[Dict]]:
        """
        Detects + recognises faces (no DB writes are made)

        Args:
            frame: np.ndarray -> given image to process

        Returns:
            people: List[Dict] -> list that contains n elements for the n found faces
                each dict is of the format:
                    {
                        "bbox": (x1, y1, x2, y2),   # Bounding box coordinates
                        "cropped_face": np.ndarray  # Face image
                        "embedding": np.array,      # Face embedding
                        "identity": Optional[str]   # None if unknown, else person_id
                        "confidence": float         # YOLO confidence score
                    }
        """
        detections = self.face_extractor.extract_faces(frame)
        if detections is None:
            print("Couldn't find any faces!")
            return None

        # Extract info
        faces, bboxes, confs = zip(*detections)
        embeddings = self.face_encoder.get_embedding(faces)
        matches = map(self.redis.find_similar, embeddings)

        # Create dict
        keys = ["bbox", "cropped_face", "embedding", "identity", "confidence"]
        lists = [bboxes, faces, embeddings, matches, confs]
        people = [
            dict(zip(keys, values))
            for values in zip(*lists)
        ]

        return people
