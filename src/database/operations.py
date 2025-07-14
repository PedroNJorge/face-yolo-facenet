from .client import RedisClient
from torch import Tensor


class FaceOperations:
    PERSON_ID_PREFIX = "person_"

    def __init__(self, redis_client: RedisClient):
        self.redis = redis_client
        self.COUNTER_KEY = "face_system:person_counter"

    def generate_person_id(self) -> str:
        """Atomically generates a new person ID"""
        new_id = self.redis.incr(self.COUNTER_KEY)
        return f"{self.PERSON_ID_PREFIX}{new_id}"

    def find_similar(self):
        pass

    def update_main_embedding(
        self,
        person_id: str,
        new_embedding: Tensor,
        weight: float = 0.2
    ) -> bool:
        """
        Updates a person's main embedding using weighted average.
        Returns success status.
        # 1. Get existing data
        profile_data = self.client.get_json(f"face_profile:{person_id}")
        if not profile_data:
            return False

        # 2. Validate and update
        profile = FaceProfile(**profile_data)
        profile.main_embedding = (
            weight * new_embedding +
            (1 - weight) * profile.main_embedding
        )

        # 3. Store back
        return self.client.set_json(
            f"face_profile:{person_id}",
            profile.dict()
        )
        """
        return False
