from pydantic import BaseModel, Field, validator, root_validator
from datetime import datetime
from typing import Optional, List, Dict, Union


class FaceEmbedding(BaseModel):
    """Flexible embedding storage supporting both binary and list formats"""
    embedding: Union[bytes, List[float]]  # Binary for Redis, List for RedisJSON


class FaceMetadata(BaseModel):
    """Non-vector face attributes"""
    person_id: str = Field(..., min_length=3, regex=r'^[a-z0-9_-]+$')
    name: Optional[str] = Field(None, min_length=2, max_length=100)
    first_seen: float = Field(default_factory=lambda: datetime.now().timestamp())
    last_updated: float = Field(default_factory=lambda: datetime.now().timestamp())
    source_images: List[str] = Field(default_factory=list)  # path to images


class FaceProfile(BaseModel):
    """Aggregated person profile with multiple embeddings"""
    person_id: str = Field(..., min_length=3)
    metadata: FaceMetadata
    main_embedding: FaceEmbedding  # average of embeddings
    embeddings: Dict[str, FaceEmbedding] = Field(default_factory=dict)  # {image_hash: embedding}

    @root_validator(pre=True)
    def sync_person_id(cls, values):
        if 'metadata' in values and 'person_id' in values['metadata']:
            values['person_id'] = values['metadata']['person_id']
        return values


class UnknownFace(BaseModel):
    """Schema for unprocessed detections (unchanged)"""
    embedding: bytes
    bbox: tuple[int, int, int, int]
    confidence: float = Field(..., ge=0.5, le=1.0)

    @validator('bbox', pre=True)
    def validate_bbox(cls, v):
        if isinstance(v, (list, tuple)) and len(v) == 4:
            return tuple(v)
        raise ValueError('bbox must be a tuple or list of 4 integers')
