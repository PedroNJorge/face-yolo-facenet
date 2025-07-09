from torch.nn.functional import cosine_similarity
import numpy as np
import torch


class FaceMatcher:
    @staticmethod
    def calculate_similarity(embedding1, embedding2):
        """
        Calculate cosine similarity between two embeddings

        Args:
            embedding1: Face embedding (numpy array or torch tensor)
            embedding2: Face embedding (numpy array or torch tensor)

        Returns:
            float: Similarity score between -1 and 1
        """
        if isinstance(embedding1, np.ndarray):
            embedding1 = torch.from_numpy(embedding1)
        if isinstance(embedding2, np.ndarray):
            embedding2 = torch.from_numpy(embedding2)

        return cosine_similarity(
            embedding1.reshape(1, -1),
            embedding2.reshape(1, -1)
        ).item()

    @staticmethod
    def is_match(embedding1, embedding2, threshold=0.5):
        """
        Determine if two faces match based on similarity threshold

        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            threshold: Similarity threshold (default: 0.5)

        Returns:
            bool: True if similarity score >= threshold
        """
        score = FaceMatcher.calculate_similarity(embedding1, embedding2)
        return score >= threshold
