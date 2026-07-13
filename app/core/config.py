from typing import List, Union
from pydantic import BeforeValidator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Annotated
import json

def parse_cors(v: Union[str, List[str]]) -> List[str]:
    if isinstance(v, str) and not v.startswith("["):
        return [i.strip() for i in v.split(",")]
    elif isinstance(v, (list, str)):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                return []
        return v
    return []

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_ignore_empty=True, extra="ignore"
    )
    
    PROJECT_NAME: str = "Clean Architecture FastAPI"
    ENVIRONMENT: str = "local"
    
    BACKEND_CORS_ORIGINS: Annotated[
        List[str], BeforeValidator(parse_cors)
    ] = []
    
    # PostgreSQL Configuration
    POSTGRES_SERVER: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_PORT: int = 5432
    
    # Auth
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    ALGORITHM: str = "HS256"

    # Gemini LLM Configuration
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-flash"

    # Local LLM Configuration (e.g. Ollama)
    LOCAL_LLM_URL: str = "http://localhost:11434"
    LOCAL_LLM_MODEL: str = "llama3"

    # S3 Object Storage Configuration
    S3_BUCKET: str = ""
    S3_ENDPOINT: str = ""
    S3_ACCESS_KEY: str = ""
    S3_SECRET_KEY: str = ""

    # Image Preprocessing & CLIP Filter settings
    IMAGE_SIMILARITY_THRESHOLD: float = 0.70
    IMAGE_BLUR_THRESHOLD: float = 10.0
    IMAGE_NORMALIZED_RESOLUTION: int = 512

    @property
    def async_database_uri(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

settings = Settings()
