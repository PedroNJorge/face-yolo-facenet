from torch.nn.functional import cosine_similarity
from torch import Tensor


class FaceMatcher:
    @staticmethod
    def is_match(embedding1: Tensor, embedding2: Tensor, threshold: float = 0.5) -> bool:
        """
        Determine if two faces match based on similarity threshold

        Args:
            embedding1: Tensor -> First face embedding
            embedding2: Tensor -> Second face embedding
            threshold: float -> Similarity threshold (default: 0.5)

        Returns:
            bool -> True if similarity score >= threshold else False
        """
        score = FaceMatcher._calculate_similarity(embedding1, embedding2)
        return score >= threshold

    @staticmethod
    def _calculate_similarity(embedding1: Tensor, embedding2: Tensor) -> float:
        """
        Calculate cosine similarity between two embeddings

        Args:
            embedding1: Tensor -> First face embedding
            embedding2: Tensor -> Second face embedding

        Returns:
            similarity: float -> Similarity score between -1 and 1
        """
        embedding1 = embedding1.unsqueeze(0)  # [1, 512]
        embedding2 = embedding2.unsqueeze(0)  # [1, 512]

        similarity = cosine_similarity(embedding1, embedding2).item()
        return similarity
