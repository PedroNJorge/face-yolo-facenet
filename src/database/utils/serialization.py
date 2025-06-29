import json
import base64
import lz4.frame
import numpy as np


class EmbeddingSerializer:
    @staticmethod
    def to_json(embedding):
        """
        For RedisJSON

        Args:
            embedding (ndarray): Embedding to convert

        Returns:
            Embedding in list format
        """
        return embedding.tolist()

    @staticmethod
    def to_hash(embedding):
        """
        For Redis Hashes

        Args:
            embedding (ndarray): Embedding in vector form

        Returns:
            LZ4 compressed in bytes in str format
        """
        compressed = lz4.frame.compress(embedding.astype(np.float32).tobytes())
        return base64.b64encode(compressed).decode('ascii')

    @staticmethod
    def from_json(data: list[float]) -> np.ndarray:
        """
        RedisJSON -> numpy

        Args:
            data (list): RedisJSON

        Returns:
            Embedding in ndarray format
        """
        return np.array(data, dtype=np.float32)

    @staticmethod
    def from_hash(data):
        """
        Redis Hash -> numpy

        Args:
            data (str): LZ4 compressed in bytes embedding vector

        Returns:
            Embedding in ndarray format
        """
        decoded_bytes = base64.b64encode(data.encode('ascii'))
        decompressed = lz4.frame.decompress(decoded_bytes)
        return np.frombuffer(decompressed, dtype=np.float32)


class ProfileSerializer:
    """Handles all serialization formats for face data"""

    @staticmethod
    def to_json(profile):
        """
        Convert FaceProfile object for RedisJSON storage

        Args:
            profile (FaceProfile): object to serialize

        Returns:
            Dict ready for RedisJSON storage
        """
        return {
            "person_id": profile.person_id,
            "main_embedding": FaceSerialization.embedding_to_bytes(profile.main_embedding),
            "embeddings": {
                k: v.to_bytes() for k, v in profile.embeddings.items()
            },
            "metadata": profile.metadata.dict()
        }

    @staticmethod
    def profile_from_json(data):
        """
        Reconstruct from RedisJSON

        Args:
            data (RedisJSON): Data to reconstruct

        Returns:
            FaceProfile object
        """
        from .schemas.face_profile import FaceProfile, FaceMetadata  # Avoid circular imports
        from .schemas.face_embedding import FaceEmbedding

        return FaceProfile(
            person_id=data["person_id"],
            metadata=FaceMetadata(**data["metadata"]),
            main_embedding=FaceEmbedding(embedding=np.array(data["main_embedding"], dtype=np.float32)),
            embeddings={
                    img_hash: FaceEmbedding(embedding=np.array(embedding, dtype=np.float32))
                for img_hash, embedding in data["embeddings"].items()
            }
        )


class UnknownSerializer:
    @staticmethod
    def to_hash(face: 'UnknownFace') -> Dict[str, str]:
        """Optimized for Redis Streams"""
        return {
            "embedding": EmbeddingSerializer.to_hash(face.embedding),
            "bbox": ",".join(map(str, face.bbox)),
            "camera": str(face.camera_id),
            "timestamp": face.timestamp.isoformat()
        }


