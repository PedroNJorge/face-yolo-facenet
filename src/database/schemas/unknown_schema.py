from pydantic import BaseModel, Field, validator


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
