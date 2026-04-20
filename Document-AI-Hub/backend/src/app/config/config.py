import logging
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings
from pydantic import AnyUrl, Field, field_validator


class Settings(BaseSettings):
    # allow extra environment variables (don't fail on unrelated .env entries)
    model_config = {"extra": "ignore", "env_file": ".env", "env_file_encoding": "utf-8"}
    APP_NAME: str = "Document-AI-Hub"
    DEBUG: bool = False
    SECRET_KEY: str = "dev-secret-key-0123456789abcdef0123"
    JWT_EXPIRE_MINUTES: int = 60

    @field_validator("SECRET_KEY", mode="before")
    def secret_key_must_be_secure(cls, v) -> str:
        # run before pydantic performs other validation so we can replace
        # short environment values (e.g. from .env) with a safe dev key.
        try:
            if v is None:
                raise ValueError
            sval = str(v)
        except Exception:
            return "dev-secret-key-0123456789abcdef0123456789"
        if len(sval) < 32:
            fallback = "dev-secret-key-0123456789abcdef0123456789"
            logging.getLogger(__name__).warning(
                "SECRET_KEY too short (%d), falling back to dev key",
                len(sval),
            )
            return fallback
        return sval

    # default to an absolute path inside the repo so shells don't need to set it
    _repo_root = Path(__file__).resolve().parents[2]
    _default_db = ( _repo_root / "dev.db" ).resolve()
    DATABASE_URL: str = f"sqlite:///{_default_db}"
    REDIS_URL: str = "redis://localhost:6379/0"

    @field_validator("DATABASE_URL")
    def normalize_sqlite_url(cls, v: str) -> str:
        if isinstance(v, str) and v.startswith("sqlite:///"):
            # strip the 'sqlite:///' prefix (three slashes) so './dev.db'
            # stays relative instead of becoming '/./dev.db' (which is absolute)
            rel = v.replace("sqlite:///", "", 1)
            p = Path(rel)
            if not p.is_absolute():
                repo_root = Path(__file__).resolve().parents[2]
                p = (repo_root / p).resolve()
            return f"sqlite:///{p}"
        return v

    CHROMA_PATH: str = "./chroma_db"
    CHROMA_COLLECTION: str = "document_text_chunks"
    CHROMA_IMAGE_COLLECTION: str = "document_image_chunks"

    GROQ_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "phi3:mini"

    EMBED_MODEL: str = "all-MiniLM-L6-v2"
    EMBED_DIM: int = 384
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_TOP_N: int = 5

    UPLOAD_DIR: str = "./uploads"
    ALLOWED_ORIGINS: List[str] = Field(default_factory=lambda: ["http://localhost:3000", "http://localhost:8501"]) 

    RATE_LIMIT_CHAT: int = 15
    RATE_LIMIT_UPLOAD: int = 5
    RATE_LIMIT_WINDOW_S: int = 60

    DEFAULT_PROVIDER: str = "ollama"
    # RAG/evidence defaults
    DEFAULT_CONTEXT_CHARS: int = 1000
    DEFAULT_MAX_SNIPPETS: int = 6
    # Elasticsearch / MinIO settings
    ELASTICSEARCH_URL: str = "http://localhost:9200"
    MINIO_ENDPOINT: str = "127.0.0.1:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_SECURE: bool = False
    MINIO_BUCKET: str = "rag-storage"

    


@lru_cache()
def get_settings() -> Settings:
    return Settings()  # type: ignore


settings = get_settings()
