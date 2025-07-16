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

    def find_similar(self, embedding: Tensor) -> bool:
        pass
