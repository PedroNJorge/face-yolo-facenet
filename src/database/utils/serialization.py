import json
import base64
import lz4.frame
import numpy as np
from typing import Dict, Union
from face_schema import FaceEmbedding, FaceMetadata, FaceProfile, UnknownFace
from torch import Tensor
import pickle


class SecureTensor:
    """Handles PyTorch tensor serialization/compression"""
    @staticmethod
    def encode(tensor: Tensor) -> str:
        """Tensor → pickle → LZ4 → Base85 (secure compact format)"""
        binary = pickle.dumps(tensor.cpu())
        compressed = lz4.frame.compress(binary)
        return base64.b85encode(compressed).decode('ascii')

    @staticmethod
    def decode(encoded: str) -> Tensor:
        """Base85 → LZ4 → pickle → tensor"""
        compressed = base64.b85decode(encoded.encode('ascii'))
        binary = lz4.frame.decompress(compressed)
        return pickle.loads(binary)


class EmbeddingSerializer:
    @staticmethod
    def to_json(embedding_obj: FaceEmbedding) -> list[float]:
        """
        Converts FaceEmbedding object to a list of floats for RedisJSON.
        """
        if isinstance(embedding_obj.embedding, bytes):
            # If it's bytes, convert to numpy array first, then to list
            # This assumes the bytes represent a numpy array
            decompressed = lz4.frame.decompress(embedding_obj.embedding)
            return np.frombuffer(decompressed, dtype=np.float32).tolist()
        return embedding_obj.embedding

    @staticmethod
    def to_hash(embedding_obj: FaceEmbedding) -> str:
        """
        Converts FaceEmbedding object to LZ4 compressed, base64 encoded string for Redis Hashes.
        """
        if isinstance(embedding_obj.embedding, list):
            # If it's a list, convert to numpy array, then to bytes
            embedding_np = np.array(embedding_obj.embedding, dtype=np.float32)
            compressed = lz4.frame.compress(embedding_np.tobytes())
        elif isinstance(embedding_obj.embedding, bytes):
            # If it's already bytes (e.g., from another source), assume it's already compressed or raw
            # For consistency, we'll ensure it's compressed if not already
            try:
                lz4.frame.decompress(embedding_obj.embedding) # Test if already compressed
                compressed = embedding_obj.embedding
            except Exception:
                # Assume it's raw bytes if decompression fails, then compress
                compressed = lz4.frame.compress(embedding_obj.embedding)
        else:
            raise ValueError("Unsupported embedding format for to_hash")

        return base64.b64encode(compressed).decode("ascii")

    @staticmethod
    def from_json(data: list[float]) -> FaceEmbedding:
        """
        Reconstructs FaceEmbedding object from a list of floats (from RedisJSON).
        """
        return FaceEmbedding(embedding=data)

    @staticmethod
    def from_hash(data: str) -> FaceEmbedding:
        """
        Reconstructs FaceEmbedding object from LZ4 compressed, base64 encoded string (from Redis Hash).
        """
        decoded_bytes = base64.b64decode(data.encode("ascii"))
        # The embedding in FaceEmbedding is Union[bytes, List[float]].
        # When coming from hash, it's typically raw bytes that represent the numpy array.
        # We store the decompressed bytes directly in the FaceEmbedding object.
        decompressed = lz4.frame.decompress(decoded_bytes)
        return FaceEmbedding(embedding=decompressed)


class MetadataSerializer:
    """Handles all serialization formats for metadata"""

    @staticmethod
    def to_json(metadata_obj: FaceMetadata) -> str:
        """
        Convert FaceMetadata object to JSON string for RedisJSON storage.
        """
        return metadata_obj.json()

    @staticmethod
    def from_json(data: str) -> FaceMetadata:
        """
        Reconstruct FaceMetadata object from JSON string.
        """
        return FaceMetadata.parse_raw(data)


class ProfileSerializer:
    """Handles all serialization formats for face data"""

    @staticmethod
    def to_json(profile_obj: FaceProfile) -> str:
        """
        Convert FaceProfile object to JSON string for RedisJSON storage.
        """
        # When serializing for JSON, we want the embeddings as lists of floats
        main_embedding_list = EmbeddingSerializer.to_json(profile_obj.main_embedding)
        embeddings_dict_list = {
            k: EmbeddingSerializer.to_json(v)
            for k, v in profile_obj.embeddings.items()
        }

        return json.dumps({
            "person_id": profile_obj.person_id,
            "metadata": json.loads(MetadataSerializer.to_json(profile_obj.metadata)), # Parse back to dict for embedding into main JSON
            "main_embedding": main_embedding_list,
            "embeddings": embeddings_dict_list
        })

    @staticmethod
    def from_json(data: str) -> FaceProfile:
        """
        Reconstruct FaceProfile object from RedisJSON.
        """
        data_dict = json.loads(data)
        return FaceProfile(
            person_id=data_dict["person_id"],
            metadata=MetadataSerializer.from_json(json.dumps(data_dict["metadata"])),
            main_embedding=EmbeddingSerializer.from_json(data_dict["main_embedding"]),
            embeddings={
                img_hash: EmbeddingSerializer.from_json(embedding_list)
                for img_hash, embedding_list in data_dict["embeddings"].items()
            }
        )


class UnknownSerializer:
    @staticmethod
    def to_json(unknown_face_obj: UnknownFace) -> str:
        """
        Convert UnknownFace object to JSON string for RedisJSON storage.
        """
        # For UnknownFace, embedding is always bytes
        embedding_bytes_b64 = base64.b64encode(unknown_face_obj.embedding).decode("ascii")
        return json.dumps({
            "embedding": embedding_bytes_b64,
            "bbox": list(unknown_face_obj.bbox),
            "confidence": unknown_face_obj.confidence,
        })

    @staticmethod
    def from_json(data: str) -> UnknownFace:
        """
        Reconstruct UnknownFace object from RedisJSON.
        """
        data_dict = json.loads(data)
        embedding_bytes = base64.b64decode(data_dict["embedding"].encode("ascii"))
        return UnknownFace(
            embedding=embedding_bytes,
            bbox=tuple(data_dict["bbox"]),
            confidence=data_dict["confidence"]
        )
