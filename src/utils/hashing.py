import hashlib
import numpy as np


def generate_hash(image: np.ndarray) -> str:
    """
    Generate hash from image

    Args:
        image (np.ndarray): Image to hash

    Returns:
        SHA-256 hash as hex string (first 12 chars)
    """
    img_bytes = image.tobytes()
    return hashlib.sha256(img_bytes).hexdigest()[:12]
