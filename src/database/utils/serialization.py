import base64
import lz4.frame
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
