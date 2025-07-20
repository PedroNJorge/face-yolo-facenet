from redis.commands.search.query import Query
from torch import Tensor
import numpy as np
from .utils.config import settings
from .schemas.profile_schema import FaceProfile


class FaceOperations:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.COUNTER_KEY = "face_system:person_counter"

    def generate_person_id(self) -> str:
        """Atomically generates a new person ID"""
        new_id = self.redis.incr(self.COUNTER_KEY)
        return f"{self.PERSON_ID_PREFIX}{new_id}"

    def find_similar(
        self, 
        query_embedding: Tensor,
        threshold: float = settings.SIMILARITY_THRESHOLD
    ) -> tuple[str, float] | None:
        """
        Find the most similar profile to the query embedding

        Args:
            query_embedding: Face embedding tensor to search with
            threshold: Minimum cosine similarity score (0-1)

        Returns:
        """
        query_bytes = query_embedding.numpy().astype(np.float32).tobytes()

        # redis_om query
        results = FaceProfile.find(
            Query("*=>[KNN 1 @main_embedding $vec AS score]")
            .return_fields("person_id", "score")
            .dialect(2),
            query_params={"vec": query_bytes}
        )

        if not results:
            return None

        best_match = results[0]
        similarity = 1 - float(best_match.score)  # Convert distance to similarity
        return (best_match.person_id, similarity) if similarity >= threshold else None
