from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scholarrag.config import Settings
from scholarrag.embeddings import EmbeddingGemmaEmbedder
from scholarrag.vectorstores.qdrant_store import QdrantVectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    point_id: str
    score: float
    text: str
    metadata: dict[str, Any]

    def source_dict(self) -> dict[str, Any]:
        return {
            "point_id": self.point_id,
            "score": self.score,
            "paper_id": self.metadata.get("paper_id"),
            "chunk_id": self.metadata.get("chunk_id"),
            "title": self.metadata.get("title"),
            "categories": self.metadata.get("categories", []),
            "update_date": self.metadata.get("update_date"),
            "text_preview": self.text[:500],
        }


class RetrievalService:
    def __init__(
        self,
        *,
        settings: Settings,
        embedder: EmbeddingGemmaEmbedder,
        vector_store: QdrantVectorStore,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(
        self,
        question: str,
        *,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[RetrievedChunk], dict[str, Any]]:
        final_k = top_k or self.settings.retrieval_context_k
        candidate_k = max(self.settings.retrieval_candidate_k, final_k)
        query_vector = self.embedder.embed_query(question)
        points = self.vector_store.search(
            query_vector,
            limit=candidate_k,
            filters=filters,
        )
        chunks = [
            RetrievedChunk(
                point_id=point.point_id,
                score=point.score,
                text=str(point.payload.get("text") or ""),
                metadata=point.payload,
            )
            for point in points
        ]
        retrieval_info = {
            "candidate_k": candidate_k,
            "returned_k": min(final_k, len(chunks)),
            "collection": self.vector_store.collection_name,
            "reranker": None,
        }
        return chunks[:final_k], retrieval_info
