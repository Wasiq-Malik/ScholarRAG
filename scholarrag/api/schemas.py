from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)
    filters: dict[str, Any] = Field(default_factory=dict)


class QuerySource(BaseModel):
    point_id: str
    score: float
    paper_id: str | None = None
    arxiv_url: str | None = None
    chunk_id: int | None = None
    title: str | None = None
    categories: list[str] = Field(default_factory=list)
    update_date: str | None = None
    text_preview: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[QuerySource]
    retrieval: dict[str, Any]
    models: dict[str, str]


class IngestOpenArxivRequest(BaseModel):
    limit: int | None = Field(default=None, ge=1)
    categories: list[str] | None = None


class IngestOpenArxivResponse(BaseModel):
    job_id: str
    status: str


class HealthResponse(BaseModel):
    status: str
    vector_backend: str
    qdrant_collection: str
    faiss_index_path: str | None = None
    faiss_sqlite_path: str | None = None
    faiss_index_exists: bool | None = None
    faiss_sqlite_exists: bool | None = None
    embedding_model: str
    embedding_dimension: int
    llm_model: str
