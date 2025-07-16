from pydantic import BaseModel, Field, validator, root_validator
from datetime import datetime
from typing import Optional, List, Dict
from ..utils.serialization import SecureTensor
from torch import Tensor


class FaceEmbedding(BaseModel):
    """Secure embedding storage using only compressed strings"""
    embedding: str  # Base85 + LZ4 + Pickle format only

    @validator('embedding')
    def validate_embedding(cls, v):
        """Verify the string can be decoded"""
        try:
            _ = SecureTensor.decode(v)  # Test decompression
            return v
        except Exception as e:
            raise ValueError(f"Invalid embedding format: {e}")

    @classmethod
    def from_tensor(cls, tensor: Tensor) -> 'FaceEmbedding':
        """Preferred constructor from tensors"""
        return cls(
            embedding=SecureTensor.encode(tensor),
            encoding_version="lz4_b85_v1"
        )

    def to_tensor(self) -> Tensor:
        """Always returns a torch.Tensor"""
        return SecureTensor.decode(self.embedding)


class FaceImageMetadata(BaseModel):
    """Metadata about a specific face image/embedding"""
    image_hash: str = Field(..., min_length=8, max_length=64)
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    detection_confidence: float = Field(..., ge=0.0, le=1.0)
    face_bbox: tuple[int, int, int, int]  # (x, y, w, h)
    source_images: List[str] = Field(default_factory=list)


class FaceImageRecord(BaseModel):
    """Combined storage for image metadata + embedding"""
    metadata: FaceImageMetadata
    embedding: FaceEmbedding

    def get_embedding_tensor(self) -> Tensor:
        return self.embedding.to_tensor()


class FaceMetadata(BaseModel):
    """Non-vector face attributes"""
    person_id: str = Field(..., min_length=3, regex=r'^[a-z0-9_-]+$')
    name: Optional[str] = Field(None, min_length=2, max_length=100)
    first_seen: float = Field(default_factory=lambda: datetime.now().timestamp())
    last_updated: float = Field(default_factory=lambda: datetime.now().timestamp())


class FaceProfile(BaseModel):
    """Aggregated person profile with multiple embeddings"""
    person_id: str = Field(..., min_length=3)
    metadata: FaceMetadata
    main_embedding: FaceEmbedding  # weighted average of embeddings
    embeddings: Dict[str, FaceImageRecord] = Field(default_factory=dict)  # {image_hash: record}

    @root_validator(pre=True)
    def sync_person_id(cls, values):
        if 'metadata' in values and 'person_id' in values['metadata']:
            values['person_id'] = values['metadata']['person_id']
        return values

    def update_main_embedding(self, new_embedding: Tensor, weight: float = 0.2):
        """Weighted average update with tensor support"""
        current = self.main_embedding.to_tensor()
        updated = weight * new_embedding + (1 - weight) * current
        self.main_embedding = FaceEmbedding(embedding=updated)
        self.metadata.last_updated = datetime.now().timestamp()

    def add_embedding(self, image_hash: str, embedding: Tensor):
        """Auto-converts input to secure storage format"""
        self.embeddings[image_hash] = FaceEmbedding(embedding=embedding)
        self.metadata.source_images.append(image_hash)
