import hashlib


def generate_hash(input_data):
    """
    Generate hash for image_path

    Args:
        input_data (str): Image path

    Returns:
        SHA-256 hash as hex string (first 12 chars)
    """
    if isinstance(input_data, str):
        return hashlib.sha256(input_data).hexdigest()[:12]

    raise ValueError(f"Unhashable type: {type(input_data)}")
