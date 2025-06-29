from pydantic import BaseModel, validator
import numpy as np
from typing import List, Union
from ..utils import EmbeddingSerializer


class FaceEmbedding(BaseModel):
    """
    Represents a face embedding vector embedding.
    Supports:
    - ndarray (raw)
    - List[float] (JSON)
    - str (Base64+LZ4 for Hashes)
    """
    embedding: Union[np.ndarray, List[float], str]

    @validator('embedding')
    def validate_embedding(cls, v):
        if isinstance(v, np.ndarray):
            if len(v) not in (128, 512):  # Standard FaceNet sizes
                raise ValueError("Embedding must be 128D or 512D")
            return v
        elif isinstance(v, (list, str)):
            return v

        raise TypeError(f"Unsupported type {type(v)}")

    def to_str(self):
        """For Redis Hashes (Base64+LZ4)"""
        if isinstance(self.embedding, str):
            return self.embedding
        return EmbeddingSerializer.to_hash(self._get_array())

    def to_list(self):
        """For RedisJSON (List[float])"""
        if isinstance(self.embedding, list):
            return self.embedding
        return np.frombuffer(self.embedding, dtype=np.float32).tolist()

    @classmethod
    def from_str(cls, data):
        """Create object from data in str format"""
        arr = EmbeddingSerializer.from_hash(data)
        return cls(embedding=arr)

    @classmethod
    def from_list(cls, data):
        """Create object from data in list format"""
        arr = EmbeddingSerializer.from_json(data)
        return cls(embedding=arr)

    def _get_array(self):
        """Returns ndarray regardless of storage format"""
        if isinstance(self.embedding, np.ndarray):
            return self.embedding
        elif isinstance(self.embedding, bytes):
            return EmbeddingSerializer.from_hash(self.embedding)
        return EmbeddingSerializer.from_json(self.embedding)
