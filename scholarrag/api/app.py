from __future__ import annotations

from functools import lru_cache
import json
import traceback
from typing import Any
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from scholarrag.api.schemas import (
    AnswerStreamRequest,
    HealthResponse,
    IngestOpenArxivRequest,
    IngestOpenArxivResponse,
    QueryRequest,
    QueryResponse,
    QuerySource,
    RetrieveRequest,
    RetrieveResponse,
)
from scholarrag.config import Settings, get_settings
from scholarrag.embeddings import EmbeddingGemmaEmbedder
from scholarrag.ingest import OpenArxivIndexer
from scholarrag.llm import GeminiAnswerGenerator
from scholarrag.retrieval import RetrievalService, RetrievedChunk
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
    GeminiAnswerGenerator,
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
    generator = GeminiAnswerGenerator(settings)
    return settings, store, embedder, vector_store, retriever, generator


def create_app() -> FastAPI:
    app = FastAPI(title="ScholarRAG", version="0.1.0")

    def build_retrieved_chunks(
        *,
        retriever: RetrievalService,
        question: str,
        top_k: int,
        filters: dict[str, Any],
    ) -> tuple[list[Any], dict[str, Any]]:
        return retriever.retrieve(question, top_k=top_k, filters=filters)

    def response_models(settings: Settings) -> dict[str, str]:
        return {
            "embedding": settings.embedding_model_id,
            "llm": settings.gemini_model,
        }

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
            llm_model=settings.gemini_model,
        )

    @app.post("/retrieve", response_model=RetrieveResponse)
    def retrieve(request: RetrieveRequest) -> RetrieveResponse:
        settings, _store, _embedder, _vector_store, retriever, _generator = get_services()
        try:
            chunks, retrieval_info = build_retrieved_chunks(
                retriever=retriever,
                question=request.question,
                top_k=request.top_k,
                filters=request.filters,
            )
            return RetrieveResponse(
                sources=[chunk.source_dict() for chunk in chunks],
                retrieval=retrieval_info,
                models=response_models(settings),
            )
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    @app.post("/answer/stream")
    def answer_stream(request: AnswerStreamRequest) -> StreamingResponse:
        settings, store, _embedder, _vector_store, _retriever, generator = get_services()

        chunks = [
            RetrievedChunk(
                point_id=source.point_id,
                score=source.score,
                text=source.text or source.text_preview,
                metadata={
                    "paper_id": source.paper_id,
                    "chunk_id": source.chunk_id,
                    "title": source.title,
                    "categories": source.categories,
                    "update_date": source.update_date,
                },
            )
            for source in request.sources
        ]

        def event_stream():
            try:
                accumulated = ""
                for piece in generator.stream_answer(question=request.question, chunks=chunks):
                    accumulated += piece
                    yield f"event: chunk\ndata: {json.dumps({'text': piece})}\n\n"
                store.log_query(
                    query_id=str(uuid4()),
                    question=request.question,
                    answer=accumulated,
                    sources=[source.model_dump() for source in request.sources],
                    model=settings.gemini_model,
                )
                yield "event: done\ndata: {}\n\n"
            except Exception as exc:
                traceback.print_exc()
                error_payload = {"detail": f"{type(exc).__name__}: {exc}"}
                yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        settings, store, _embedder, _vector_store, retriever, generator = get_services()
        try:
            chunks, retrieval_info = build_retrieved_chunks(
                retriever=retriever,
                question=request.question,
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
                model=settings.gemini_model,
            )
            return QueryResponse(
                answer=answer,
                sources=sources,
                retrieval=retrieval_info,
                models=response_models(settings),
            )
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

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
