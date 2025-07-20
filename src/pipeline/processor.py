from ..detection import FaceExtractor
from ..recognition import FaceEncoder
from ..database import RedisClient
from .snapshot_manager import SnapshotManager
import numpy as np
from torch import Tensor


class Processor:
    def __init__(self):
        self.face_extractor = FaceExtractor()
        self.face_encoder = FaceEncoder()
        self.redis = RedisClient()
        self.snapshot_manager = SnapshotManager()

    def process_frame(self, frame: np.ndarray) -> list[dict] | None:
        """
        Detects + recognises faces (no DB writes are made)

        Args:
            frame: np.ndarray -> given image to process

        Returns:
            people: List[Dict] -> list that contains n elements for the n found faces
                each dict is of the format:
                    {
                        "bbox": (x1, y1, x2, y2),       # Bounding box coordinates
                        "face": np.ndarray      # Face image
                        "embedding": np.array,          # Face embedding
                        "detection_confidence": float   # YOLO confidence score
                        "identity": Optional[str]       # None if unknown, else person_id
                        "profile": FaceProfile          # None if unknown, else FaceProfile obj
                        "match_confidence": float       # Vector search match score
                    }
        """
        detections = self.face_extractor.extract_faces(frame)
        if detections is None:
            print("Couldn't find any faces!")
            return None

        # Extract info
        faces, bboxes, dconfs = zip(*detections)
        embeddings = self.face_encoder.get_embedding(faces)
        matches, mconfs = zip(*map(self.redis.find_similar, embeddings))
        profiles = map(self.redis.read_profile, matches)

        # Create dict
        keys = ["bbox", "face", "embedding", "detection_confidence", "identity", "profile", "match_confidence"]
        lists = [bboxes, faces, embeddings, dconfs, matches, profiles, mconfs]
        people = [
            dict(zip(keys, values))
            for values in zip(*lists)
        ]

        # Update DB with images of known profiles
        for person in people:
            if person["profile"] is not None:
                person["profile"].update_profile(
                        person["identity"],
                        person["face"],
                        person["embedding"],
                        person["detection_confidence"]
                        )

        return people

    def add_person(
            self,
            name: str,
            face: np.ndarray,
            dconf: float,
            bbox: tuple[int, int, int, int],
            embedding: Tensor
            ):
        profile = self.redis.create_profile(name, face, dconf, bbox, embedding)
        self.snapshot_manager

    def delete_person(self):
        pass
