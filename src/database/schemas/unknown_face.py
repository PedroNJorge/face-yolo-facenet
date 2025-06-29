from pydantic import BaseModel, Field, validator, root_validator
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Union
import json
from utils.hashing import generate_hash


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
