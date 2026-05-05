from __future__ import annotations

from functools import lru_cache
from typing import Any
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException

from scholarrag.api.schemas import (
    HealthResponse,
    IngestOpenArxivRequest,
    IngestOpenArxivResponse,
    QueryRequest,
    QueryResponse,
)
from scholarrag.config import Settings, get_settings
from scholarrag.embeddings import EmbeddingGemmaEmbedder
from scholarrag.ingest import OpenArxivIndexer
from scholarrag.llm import VLLMAnswerGenerator
from scholarrag.retrieval import RetrievalService
from scholarrag.storage import SQLiteStore
from scholarrag.vectorstores.qdrant_store import QdrantVectorStore


def build_vector_store(settings: Settings) -> Any:
    if settings.vector_backend == "faiss":
        from scholarrag.vectorstores.faiss_store import FaissVectorStore

        return FaissVectorStore(settings)
    return QdrantVectorStore(settings)


@lru_cache(maxsize=1)
def get_services() -> tuple[
    Settings,
    SQLiteStore,
    EmbeddingGemmaEmbedder,
    Any,
    RetrievalService,
    VLLMAnswerGenerator,
]:
    settings = get_settings()
    store = SQLiteStore(settings.sqlite_path)
    store.init_db()
    embedder = EmbeddingGemmaEmbedder(settings)
    vector_store = build_vector_store(settings)
    retriever = RetrievalService(
        settings=settings,
        embedder=embedder,
        vector_store=vector_store,
    )
    generator = VLLMAnswerGenerator(settings)
    return settings, store, embedder, vector_store, retriever, generator


def create_app() -> FastAPI:
    app = FastAPI(title="ScholarRAG", version="0.1.0")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        settings = get_settings()
        faiss_index_exists = None
        faiss_sqlite_exists = None
        faiss_index_path = None
        faiss_sqlite_path = None
        if settings.vector_backend == "faiss":
            faiss_index_path = str(settings.faiss_index_path)
            faiss_sqlite_path = str(settings.faiss_sqlite_path)
            faiss_index_exists = settings.faiss_index_path.exists()
            faiss_sqlite_exists = settings.faiss_sqlite_path.exists()
        return HealthResponse(
            status="ok",
            vector_backend=settings.vector_backend,
            qdrant_collection=settings.qdrant_collection,
            faiss_index_path=faiss_index_path,
            faiss_sqlite_path=faiss_sqlite_path,
            faiss_index_exists=faiss_index_exists,
            faiss_sqlite_exists=faiss_sqlite_exists,
            embedding_model=settings.embedding_model_id,
            embedding_dimension=settings.embedding_dimension,
            vllm_model=settings.vllm_model,
        )

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        settings, store, _embedder, _vector_store, retriever, generator = get_services()
        chunks, retrieval_info = retriever.retrieve(
            request.question,
            top_k=request.top_k,
            filters=request.filters,
        )
        answer = generator.answer(question=request.question, chunks=chunks)
        sources = [chunk.source_dict() for chunk in chunks]
        store.log_query(
            query_id=str(uuid4()),
            question=request.question,
            answer=answer,
            sources=sources,
            model=settings.vllm_model,
        )
        return QueryResponse(
            answer=answer,
            sources=sources,
            retrieval=retrieval_info,
            models={
                "embedding": settings.embedding_model_id,
                "llm": settings.vllm_model,
            },
        )

    @app.post("/ingest/open-arxiv", response_model=IngestOpenArxivResponse)
    def ingest_open_arxiv(
        request: IngestOpenArxivRequest,
        background_tasks: BackgroundTasks,
    ) -> IngestOpenArxivResponse:
        settings, store, embedder, vector_store, _retriever, _generator = get_services()
        if settings.vector_backend != "qdrant":
            raise HTTPException(
                status_code=400,
                detail=(
                    "OpenArXiv API ingestion currently writes to Qdrant only. "
                    "Use the Colab notebook to build FAISS artifacts."
                ),
            )

        job_id = str(uuid4())
        store.create_ingestion_job(
            job_id=job_id,
            limit=request.limit,
            categories=request.categories,
            status="queued",
        )

        indexer = OpenArxivIndexer(
            settings=settings,
            store=store,
            embedder=embedder,
            vector_store=vector_store,
        )
        background_tasks.add_task(
            indexer.index,
            limit=request.limit,
            categories=request.categories,
            job_id=job_id,
        )
        return IngestOpenArxivResponse(job_id=job_id, status="queued")

    return app


app = create_app()
