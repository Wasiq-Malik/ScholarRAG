from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the local ScholarRAG production app."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SCHOLARRAG_",
        extra="ignore",
    )

    open_arxiv_dataset: str = "open-index/open-arxiv"
    hf_cache_dir: Path = Path("data/hf_cache")
    hf_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices("SCHOLARRAG_HF_TOKEN", "HF_TOKEN"),
    )

    embedding_model_id: str = "google/embeddinggemma-300m"
    embedding_dimension: int = 768
    embedding_batch_size: int = 64
    embedding_query_prompt: str = "task: fact checking | query: "
    embedding_document_prompt_template: str = "title: {title} | text: "

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "open_arxiv_embeddinggemma_fact_check_768_cosine"

    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: str = "token-abc123"
    vllm_model: str = "Qwen/Qwen3.5-9B"
    vllm_temperature: float = 0.2
    vllm_max_tokens: int = 1024

    sqlite_path: Path = Path("data/scholarrag.sqlite3")

    chunk_chars: int = 1800
    chunk_overlap_chars: int = 250
    ingest_batch_chunks: int = 64

    retrieval_candidate_k: int = 50
    retrieval_context_k: int = 10


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
