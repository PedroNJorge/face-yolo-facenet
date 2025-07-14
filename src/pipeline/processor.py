from ..detection import FaceExtractor
from ..recognition import FaceEncoder
import numpy as np


class Processor:
    def __init__(
            self,
            face_extractor: FaceExtractor,
            face_encoder: FaceEncoder
    ):
        self.face_extractor = face_extractor
        self.face_encoder = face_encoder

    def process_frame(self, frame: np.ndarray):
        # 1. Detect and extract faces (and confidence) with YOLO
        detections = self.face_extractor.extract_faces(frame)

        if detections is None:
            print("Couldn't find any faces!")
            return

        # 2. For each face, encode embedding
        embedding_conf = self.face_encoder.get_embedding(detections)
        """

        # 3. Match against Redis DB
        match = face_matcher.find_similar(embedding)

        # 4. Return or store results
        yield {"face": face, "identity": match}
        """
