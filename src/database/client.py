import redis
import numpy as np

from .schemas.profile_schema import FaceProfile, FaceMetadata
from .utils.config import settings
from ..utils.hashing import generate_hash
from .operations import FaceOperations
from torch import Tensor


class RedisClient:
    _pools = {}  # Class-level pool registry by DB number

    def __init__(
        self,
        host: str = settings.HOST,
        port: int = settings.PORT,
        db: int = settings.DB,
        decode_responses: bool = False,
        max_connections: int = 15
    ):
        pool_key = (host, port, db)
        if pool_key not in self.__class__._pools:
            self.__class__._pools[pool_key] = redis.ConnectionPool(
                host=host,
                port=port,
                db=db,
                decode_responses=decode_responses,
                max_connections=max_connections
            )

        self.redis = redis.Redis(connection_pool=self.__class__._pools[pool_key])
        self._ops = FaceOperations(self.redis)

    @classmethod
    def cleanup(cls):
        """Close all connection pools"""
        for pool in cls._pools.values():
            pool.disconnect()
        cls._pools.clear()

    def find_similar(
            self, query_embedding: Tensor,
            threshold: float = settings.SIMILARITY_THRESHOLD
    ) -> tuple[str, float] | None:
        """
        Find the most similar profile to the query embedding

        Args:
            query_embedding: Face embedding tensor to search with
            threshold: Minimum cosine similarity score (0-1)

        Returns:
            (person_id, match_confidence) or (None, 0.0)
        """
        result = self._ops.find_similar(query_embedding, threshold)
        return result if result is not None else (None, 0.0)

    # --- CRUD for Face Profiles ---
    def create_profile(
            self,
            name: str,
            face: np.ndarray,
            dconf: float,
            bbox: tuple[int, int, int, int],
            embedding: Tensor
            ) -> FaceProfile:
        """Create a new face profile"""
        try:
            new_person_id = self._ops.generate_person_id()
            new_profile = FaceProfile(
                person_id=new_person_id,
                metadata=FaceMetadata(person_id=new_person_id, name=name),
            )

            img_hash = generate_hash(face)
            new_profile.add_embedding(img_hash, embedding, bbox, dconf)
            new_profile.update_main_embedding(embedding, weight=1)

            self._add_embedding_profile(new_profile, face, embedding, bbox, dconf)
            new_profile.save()

            return new_profile
        except Exception:
            return None

    def read_profile(person_id: str | None) -> FaceProfile | None:
        """Retrieve a FaceProfile or return None if not found"""
        try:
            return FaceProfile.get(person_id)
        except Exception:
            return None

    def update_profile(
            self,
            person_id: str | None,
            face: np.ndarray,
            embedding: Tensor,
            bbox: tuple[int, int, int, int],
            dconf: float
            ) -> bool:
        """Update an existing profile"""
        if person_id is None:
            return False

        profile = self.read_profile(person_id)
        if not profile:
            return False

        img_hash = generate_hash(face)
        profile.add_embedding(img_hash, embedding, bbox, dconf)
        profile.update_main_embedding(embedding)
        profile.save()

        return True

    def delete_profile(self, person_id: str):
        """Delete a face profile"""
        FaceProfile.delete(person_id)

    # --- Unknown Faces ---
    '''
    def save_unknown_face(self, unknown: UnknownFace) -> str:
        """Save an unknown face with timestamp-based key"""
        key = f"unknown:{datetime.now().timestamp()}"
        self.redis.set(key, UnknownSerializer.to_json(unknown))
        return key

    def get_unknown_faces(self, limit: int = 100) -> List[UnknownFace]:
        """Retrieve recent unknown faces"""
        keys = sorted(self.redis.keys("unknown:*"), reverse=True)[:limit]
        return [
            UnknownSerializer.from_json(self.redis.get(k).decode('utf-8'))
            for k in keys
        ]
    '''
