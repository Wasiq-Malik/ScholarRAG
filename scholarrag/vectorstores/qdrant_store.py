from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scholarrag.chunking import DocumentChunk
from scholarrag.config import Settings


@dataclass(frozen=True)
class RetrievedPoint:
    point_id: str
    score: float
    payload: dict[str, Any]


def build_payload_filter(filters: dict[str, Any] | None) -> Any | None:
    if not filters:
        return None

    from qdrant_client.http import models

    conditions = []
    for key, value in filters.items():
        if value is None:
            continue
        if isinstance(value, list):
            match = models.MatchAny(any=value)
        else:
            match = models.MatchValue(value=value)
        conditions.append(models.FieldCondition(key=key, match=match))

    if not conditions:
        return None
    return models.Filter(must=conditions)


class QdrantVectorStore:
    backend = "qdrant"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: Any | None = None

    @property
    def collection_name(self) -> str:
        return self.settings.qdrant_collection

    def _get_client(self) -> Any:
        if self._client is None:
            from qdrant_client import QdrantClient

            kwargs: dict[str, Any] = {"url": self.settings.qdrant_url}
            if self.settings.qdrant_api_key:
                kwargs["api_key"] = self.settings.qdrant_api_key
            self._client = QdrantClient(**kwargs)
        return self._client

    def ensure_collection(self) -> None:
        from qdrant_client.http import models

        client = self._get_client()
        exists = False
        if hasattr(client, "collection_exists"):
            exists = bool(client.collection_exists(self.collection_name))
        else:
            try:
                client.get_collection(self.collection_name)
                exists = True
            except Exception:
                exists = False

        if exists:
            return

        client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.settings.embedding_dimension,
                distance=models.Distance.COSINE,
            ),
        )

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return

        from qdrant_client.http import models

        client = self._get_client()
        points = [
            models.PointStruct(
                id=chunk.point_id,
                vector=vector,
                payload=chunk.payload(),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        client.upsert(collection_name=self.collection_name, points=points)

    def search(
        self,
        vector: list[float],
        *,
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedPoint]:
        client = self._get_client()
        query_filter = build_payload_filter(filters)

        if hasattr(client, "query_points"):
            response = client.query_points(
                collection_name=self.collection_name,
                query=vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
            points = response.points
        else:
            points = client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )

        return [
            RetrievedPoint(
                point_id=str(point.id),
                score=float(point.score),
                payload=dict(point.payload or {}),
            )
            for point in points
        ]
