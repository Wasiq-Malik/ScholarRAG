from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the local ScholarRAG production app."""

    model_config = SettingsConfigDict(
        env_file=(".env", "scholarrag/.env"),
        env_prefix="SCHOLARRAG_",
        extra="ignore",
        populate_by_name=True,
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

    vector_backend: Literal["qdrant", "faiss"] = "qdrant"
    faiss_index_path: Path = Path("data/faiss/open_arxiv_papers.faiss")
    faiss_sqlite_path: Path = Path("data/faiss/open_arxiv_papers.sqlite")
    faiss_nprobe: int = 32
    faiss_filter_multiplier: int = 5

    gemini_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("SCHOLARRAG_GEMINI_API_KEY", "GEMINI_API_KEY"),
    )
    gemini_model: str = Field(
        default="gemma-4-31b-it",
        validation_alias=AliasChoices("SCHOLARRAG_GEMINI_MODEL", "SCHOLARRAG_LLM_MODEL"),
    )
    gemini_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta",
        validation_alias=AliasChoices(
            "SCHOLARRAG_GEMINI_BASE_URL",
            "SCHOLARRAG_LLM_BASE_URL",
        ),
    )
    gemini_temperature: float = Field(
        default=0.2,
        validation_alias=AliasChoices(
            "SCHOLARRAG_GEMINI_TEMPERATURE",
            "SCHOLARRAG_LLM_TEMPERATURE",
        ),
    )
    gemini_max_tokens: int = Field(
        default=1024,
        validation_alias=AliasChoices(
            "SCHOLARRAG_GEMINI_MAX_TOKENS",
            "SCHOLARRAG_LLM_MAX_TOKENS",
        ),
    )

    sqlite_path: Path = Path("data/scholarrag.sqlite3")

    chunk_chars: int = 1800
    chunk_overlap_chars: int = 250
    ingest_batch_chunks: int = 64

    retrieval_candidate_k: int = 50
    retrieval_context_k: int = 10


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
