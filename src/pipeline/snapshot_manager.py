from pathlib import Path
from ..utils.hashing import generate_hash
from ..database.schemas.profile_schema import FaceProfile
import numpy as np
import cv2


class SnapshotManager:
    def __init__(self, base_path: str = "data"):
        project_root = Path(__file__).parent.parent.parent
        self.base = project_root / base_path
        self.known = self.base / "known_faces"
        self.unknown = self.base / "unknown_faces"
        self._ensure_dirs()

    def _ensure_dirs(self):
        self.known.mkdir(exist_ok=True, parents=True)
        self.unknown.mkdir(exist_ok=True, parents=True)

    def save_known_face(self, profile: FaceProfile, image: np.ndarray):
        """Save image in data/known_faces/[person_id]"""
        # Generate hash and create person directory
        image_hash = generate_hash(image)
        person_dir = self.known / profile.person_id
        person_dir.mkdir(exist_ok=True, parents=True)

        # Save image only (no metadata.json)
        cv2.imwrite(str(person_dir / f"{image_hash}.jpg"), image)
