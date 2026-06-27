"""Application configuration using Pydantic Settings."""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """Central configuration loaded from environment variables / .env file."""

    # Application
    app_name: str = Field(default="Intelligent Document Q&A Assistant", alias="APP_NAME")
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    debug: bool = Field(default=False, alias="DEBUG")

    # Database
    database_url: str = Field(
        default="postgresql://raguser:ragpassword@localhost:5432/ragdb",
        alias="DATABASE_URL",
    )

    # NVIDIA NIM / LLM
    nvidia_api_key: str = Field(default="", alias="NVIDIA_API_KEY")
    nvidia_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        alias="NVIDIA_BASE_URL",
    )
    llm_model: str = Field(
        default="meta/llama-3.1-8b-instruct",
        alias="LLM_MODEL",
    )
    llm_temperature: float = Field(default=0.1, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=1024, alias="LLM_MAX_TOKENS")

    # Embedding
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )
    embedding_dimension: int = Field(default=384, alias="EMBEDDING_DIMENSION")

    # Chunking
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=64, alias="CHUNK_OVERLAP")

    # Retrieval
    retrieval_top_k: int = Field(default=5, alias="RETRIEVAL_TOP_K")

    # Storage
    faiss_index_path: str = Field(default="faiss_index", alias="FAISS_INDEX_PATH")
    upload_dir: str = Field(default="uploads", alias="UPLOAD_DIR")

    # MLflow
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5001",
        alias="MLFLOW_TRACKING_URI",
    )
    mlflow_experiment_name: str = Field(
        default="rag-document-qa",
        alias="MLFLOW_EXPERIMENT_NAME",
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "populate_by_name": True}


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
