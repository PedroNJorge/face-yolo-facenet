from redis_om import (EmbeddedJsonModel, JsonModel, Field, Migrator)
from pydantic import validator, root_validator
from datetime import datetime
from typing import Optional, Dict, List
from torch import Tensor
import torch
import numpy as np
from ..utils.serialization import SecureTensor


# --- Embedded Models (Stored as JSON in Redis) ---
class FaceEmbedding(EmbeddedJsonModel):
    """Secure embedding storage using only compressed strings"""
    embedding: str  # Base85 + LZ4 + Pickle format

    @validator('embedding')
    def validate_embedding(cls, v):
        try:
            _ = SecureTensor.decode(v)
            return v
        except Exception as e:
            raise ValueError(f"Invalid embedding format: {e}")

    @classmethod
    def from_tensor(cls, tensor: Tensor) -> 'FaceEmbedding':
        return cls(embedding=SecureTensor.encode(tensor))

    def to_tensor(self) -> Tensor:
        return SecureTensor.decode(self.embedding)


class FaceImageMetadata(EmbeddedJsonModel):
    image_hash: str = Field(..., min_length=8, max_length=64)
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    detection_confidence: float = Field(..., ge=0.0, le=1.0)
    face_bbox: tuple[int, int, int, int]


class FaceImageRecord(EmbeddedJsonModel):
    metadata: FaceImageMetadata
    embedding: FaceEmbedding

    def get_embedding_tensor(self) -> Tensor:
        return self.embedding.to_tensor()


class FaceMetadata(EmbeddedJsonModel):
    person_id: str = Field(..., min_length=3, regex=r'^[a-z0-9_-]+$')
    name: Optional[str] = Field(..., min_length=2, max_length=100)
    first_seen: float = Field(default_factory=lambda: datetime.now().timestamp())
    last_seen: float = Field(default_factory=lambda: datetime.now().timestamp())


# --- Main Model (Stored as Redis JSON Document) ---
class FaceProfile(JsonModel):
    """Redis OM model with vector search support"""
    person_id: str = Field(primary_key=True)  # Used in key generation
    metadata: FaceMetadata
    main_embedding: bytes

    embeddings: Dict[str, FaceImageRecord] = Field(default_factory=dict)

    # Customize Redis key format
    def key(self) -> str:
        return f"profile:{self.person_id}"

    # Keep all your existing methods
    @root_validator(pre=True)
    def sync_person_id(cls, values):
        if 'metadata' in values and 'person_id' in values['metadata']:
            values['person_id'] = values['metadata']['person_id']
        return values

    def add_embedding(
            self,
            image_hash: str,
            embedding: Tensor,
            bbox: tuple[int, int, int, int],
            dconf: float
            ):
        self.embeddings[image_hash] = FaceImageRecord(
            metadata=FaceImageMetadata(
                image_hash=image_hash,
                detection_confidence=dconf,
                face_bbox=bbox
                ),
            embedding=FaceEmbedding.from_tensor(embedding)
        )
        self.metadata.source_images.append(image_hash)
        self.metadata.last_seen = datetime.now().timestamp()

    def update_main_embedding(self, new_embedding: Tensor, weight: float = 0.2):
        current_tensor = self.get_main_embedding_tensor()
        updated = weight * new_embedding + (1 - weight) * current_tensor
        self.main_embedding = self._tensor_to_bytes(updated)

    def get_main_embedding_tensor(self) -> Tensor:
        return torch.from_numpy(np.frombuffer(self.main_embedding, dtype=np.float32))

    @staticmethod
    def _tensor_to_bytes(tensor: Tensor) -> bytes:
        return tensor.numpy().astype(np.float32).tobytes()


# Initialize indexes (run once at startup)
Migrator().run()
