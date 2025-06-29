import redis
from ...config import REDIS_HOST, REDIS_PORT


class RedisClient:
    _pool = None

    @classmethod
    def get_client(cls, db_num=0):
        if cls._pool is None:
            cls._pool = redis.ConnectionPool(
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    db=db_num,
                    decode_responses=False,  # Enable binary data storage
                    max_connections=15,
                    socket_timeout=5,
                    health_check_interval=30
            )
        return redis.Redis(connection_pool=cls._pool)
