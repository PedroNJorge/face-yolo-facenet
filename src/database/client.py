import json
from typing import Optional, List, Dict, Tuple
import redis
from datetime import datetime
import numpy as np
import lz4.frame
import base64

from .schemas import FaceEmbedding, FaceProfile, UnknownFace, FaceMetadata
from .utils.face_matcher import FaceMatcher
from .utils.serialization import ProfileSerializer, UnknownSerializer
from .utils.config import REDIS_HOST, REDIS_PORT, PROD_DB


class RedisClient:
    _pools = {}  # Class-level pool registry by DB number

    def __init__(
        self,
        host: str = REDIS_HOST,
        port: int = REDIS_PORT,
        db: int = PROD_DB,
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

    @classmethod
    def cleanup(cls):
        """Close all connection pools"""
        for pool in cls._pools.values():
            pool.disconnect()
        cls._pools.clear()

    # --- CRUD for Face Profiles ---
    def create_profile(self, profile: FaceProfile) -> bool:
        """Create a new face profile"""
        if self.redis.exists(f"profile:{profile.person_id}"):
            raise ValueError(f"Profile {profile.person_id} already exists")
        return self.save_profile(profile)

    def read_profile(self, person_id: str) -> Optional[FaceProfile]:
        """Retrieve a face profile"""
        profile_data = self.redis.get(f"profile:{person_id}")
        return ProfileSerializer.from_json(profile_data.decode('utf-8')) if profile_data else None

    def update_profile(self, person_id: str, updates: dict) -> bool:
        """Update an existing profile"""
        profile = self.read_profile(person_id)
        if not profile:
            return False

        # Apply updates
        for key, value in updates.items():
            if key == 'metadata':
                profile.metadata = FaceMetadata(**{**profile.metadata.dict(), **value})
            elif hasattr(profile, key):
                setattr(profile, key, value)

        return self.save_profile(profile)

    def delete_profile(self, person_id: str) -> bool:
        """Delete a face profile"""
        return bool(self.redis.delete(f"profile:{person_id}"))

    def save_profile(self, profile: FaceProfile) -> bool:
        """Internal method to save profile"""
        profile_json = ProfileSerializer.to_json(profile)
        return self.redis.set(f"profile:{profile.person_id}", profile_json)

    # --- Embedding Operations ---
    def add_embedding(self, person_id: str, image_hash: str, embedding: np.ndarray) -> bool:
        """Add a new embedding to a profile"""
        profile = self.read_profile(person_id)
        if not profile:
            return False

        new_embedding = FaceEmbedding(embedding=embedding.tolist())
        profile.embeddings[image_hash] = new_embedding

        # Update main embedding (could implement averaging here)
        if not profile.main_embedding:
            profile.main_embedding = new_embedding

        return self.save_profile(profile)

    # --- Unknown Faces ---
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

    # --- Search Operations ---
    def find_similar(self, query_embedding: np.ndarray, threshold: float = 0.5) -> List[Tuple[FaceProfile, float]]:
        """Find profiles with similar embeddings (using FaceMatcher)"""
        matches = []
        for key in self.redis.scan_iter("profile:*"):
            profile = self.read_profile(key.decode('utf-8').split(':')[1])
            if not profile:
                continue

            # Convert profile's main embedding to numpy
            main_embed = np.array(profile.main_embedding.embedding)

            # Calculate similarity (could import your FaceMatcher here)
            score = cosine_similarity(
                query_embedding.reshape(1, -1),
                main_embed.reshape(1, -1)
            ).item()

            if score >= threshold:
                matches.append((profile, score))

        return sorted(matches, key=lambda x: x[1], reverse=True)
