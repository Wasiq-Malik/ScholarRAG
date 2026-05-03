from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from scholarrag.chunking import DocumentChunk, chunk_paper
from scholarrag.config import Settings
from scholarrag.datasets.open_arxiv import iter_open_arxiv
from scholarrag.embeddings import EmbeddingGemmaEmbedder
from scholarrag.storage import SQLiteStore
from scholarrag.vectorstores.qdrant_store import QdrantVectorStore


@dataclass(frozen=True)
class IngestionResult:
    job_id: str
    status: str
    indexed_documents: int
    indexed_chunks: int
    error: str | None = None


class OpenArxivIndexer:
    def __init__(
        self,
        *,
        settings: Settings,
        store: SQLiteStore,
        embedder: EmbeddingGemmaEmbedder,
        vector_store: QdrantVectorStore,
    ) -> None:
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.vector_store = vector_store

    def index(
        self,
        *,
        limit: int | None,
        categories: list[str] | None,
        job_id: str | None = None,
    ) -> IngestionResult:
        effective_job_id = job_id or str(uuid4())
        indexed_documents = 0
        indexed_chunks = 0
        pending_chunks: list[DocumentChunk] = []

        self.store.init_db()
        self.store.create_ingestion_job(
            job_id=effective_job_id,
            limit=limit,
            categories=categories,
            status="running",
        )
        self.vector_store.ensure_collection()
        self.store.update_ingestion_job(
            job_id=effective_job_id,
            status="running",
            indexed_documents=0,
            indexed_chunks=0,
        )

        try:
            for paper in iter_open_arxiv(
                dataset_name=self.settings.open_arxiv_dataset,
                cache_dir=self.settings.hf_cache_dir,
                limit=limit,
                categories=categories,
            ):
                self.store.upsert_document(paper)
                indexed_documents += 1
                pending_chunks.extend(
                    chunk_paper(
                        paper,
                        chunk_chars=self.settings.chunk_chars,
                        overlap_chars=self.settings.chunk_overlap_chars,
                    )
                )

                if len(pending_chunks) >= self.settings.ingest_batch_chunks:
                    indexed_chunks += self._flush_chunks(pending_chunks)
                    pending_chunks = []
                    self.store.update_ingestion_job(
                        job_id=effective_job_id,
                        status="running",
                        indexed_documents=indexed_documents,
                        indexed_chunks=indexed_chunks,
                    )

            indexed_chunks += self._flush_chunks(pending_chunks)
            self.store.update_ingestion_job(
                job_id=effective_job_id,
                status="complete",
                indexed_documents=indexed_documents,
                indexed_chunks=indexed_chunks,
            )
            return IngestionResult(
                job_id=effective_job_id,
                status="complete",
                indexed_documents=indexed_documents,
                indexed_chunks=indexed_chunks,
            )
        except Exception as exc:
            self.store.update_ingestion_job(
                job_id=effective_job_id,
                status="failed",
                indexed_documents=indexed_documents,
                indexed_chunks=indexed_chunks,
                error=str(exc),
            )
            return IngestionResult(
                job_id=effective_job_id,
                status="failed",
                indexed_documents=indexed_documents,
                indexed_chunks=indexed_chunks,
                error=str(exc),
            )

    def _flush_chunks(self, chunks: list[DocumentChunk]) -> int:
        if not chunks:
            return 0
        documents = [(chunk.text, chunk.title) for chunk in chunks]
        vectors = self.embedder.embed_documents(documents)
        self.vector_store.upsert_chunks(chunks, vectors)
        self.store.upsert_chunks(chunks)
        return len(chunks)
