from pydantic import BaseModel, Field, validator, root_validator
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Union
import json
from utils.hashing import generate_hash

# ----------------------------
# Core Storage Objects
#
# 1. FaceProfile
# 2. UnknownFace
# ----------------------------

# ----------------------------
# Core Data Type Schemas
# ----------------------------


class FaceEmbedding(BaseModel):
    """Flexible embedding storage supporting both binary and list formats"""
    embedding: Union[bytes, List[float]]  # Binary for Redis, List for RedisJSON

    @validator('embedding')
    def validate_embedding(cls, v):
        if isinstance(v, bytes):
            arr = np.frombuffer(v, dtype=np.float32)
        else:
            arr = np.array(v, dtype=np.float32)

        if len(arr) not in (128, 512):  # Standard FaceNet sizes
            raise ValueError("Embedding must be 128D or 512D")
        return v

    def to_bytes(self):
        """Convert to Redis-compatible binary"""
        if isinstance(self.embedding, bytes):
            return self.embedding
        return np.array(self.embedding, dtype=np.float32).tobytes()

    def to_list(self):
        """Convert to RedisJSON-compatible list"""
        if isinstance(self.embedding, list):
            return self.embedding
        return np.frombuffer(self.embedding, dtype=np.float32).tolist()


class FaceMetadata(BaseModel):
    """Non-vector face attributes"""
    person_id: str = Field(..., min_length=3, regex=r'^[a-z0-9_-]+$')
    name: Optional[str] = Field(None, min_length=2, max_length=100)
    first_seen: float = Field(default_factory=lambda: datetime.now().timestamp())
    last_updated: float = Field(default_factory=lambda: datetime.now().timestamp())
    source_images: List[str] = Field(default_factory=list)  # path to images

    @validator('source_images')
    def validate_source_images(cls, v):
        return list(set(v))

# ----------------------------
# Composite Schemas
# ----------------------------


class FaceProfile(BaseModel):
    """Aggregated person profile with multiple embeddings"""
    person_id: str = Field(..., min_length=3)
    metadata: FaceMetadata
    main_embedding: FaceEmbedding  # average of embeddings
    embeddings: Dict[str, FaceEmbedding] = Field(default_factory=dict)  # {image_hash: embedding}

    @root_validator(pre=True)
    def validate_embeddings(cls, values):
        if not values.get('embeddings') and values.get('main_embedding'):
            # Initialize with main embedding if empty
            values['embeddings'] = {
                "initial": values['main_embedding']
            }
        return values

    def to_redis_mapping(self):
        """Convert to redis mapping structure"""
        return {
            "person_id": self.person_id,
            "main_embedding": self.main_embedding.to_bytes(),
            "embeddings": json.dumps({
                k: v.to_bytes() for k, v in self.embeddings.items()
            }),
            **self.metadata.dict()
        }

    def to_redis_json(self):
        """Convert to RedisJSON-compatible format"""
        return {
            "person_id": self.person_id,
            "main_embedding": self.main_embedding.to_bytes(),
            "embeddings": {
                k: v.to_bytes() for k, v in self.embeddings.items()
            },
            "metadata": self.metadata.dict()
        }

    def add_embedding(self, embedding, image_path):
        """
        Add new embedding and update main embedding

        Args:
            embedding (FaceEmbedding): New embedding to add
            image_path (str): Image path of the embedding
        """
        image_hash = generate_hash(image_path)
        self.embeddings[image_hash] = embedding
        # Simple average update (can be weighted bc people change with time)
        all_embeddings = [e.to_list() for e in self.embeddings.values()]
        avg_embedding = np.mean(all_embeddings, axis=0)
        self.main_embedding = FaceEmbedding(embedding=avg_embedding.tolist())

# ----------------------------
# Temporary Data Schemas
# ----------------------------


class UnknownFace(BaseModel):
    """Schema for unprocessed detections (unchanged)"""
    embedding: bytes
    bbox: tuple[int, int, int, int]
    confidence: float = Field(..., ge=0.5, le=1.0)

    def to_redis_json(self):
        return {
            "embedding": np.frombuffer(self.embedding, dtype=np.float32).tolist(),
            "bbox": list(self.bbox),
            "confidence": self.confidence,
        }
