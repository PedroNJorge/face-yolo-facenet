from pydantic_settings import BaseSettings


class RedisSettings(BaseSettings):
    HOST: str = "localhost"
    PORT: int = 6379
    DB: int = 0

    PERSON_ID_PREFIX: str = "person_"

    FACE_INDEX_NAME: str = "faces"
    SIMILARITY_THRESHOLD: float = 0.7
    VECTOR_DIM: int = 512  # FaceNet embedding size


settings = RedisSettings()
