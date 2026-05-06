from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scholarrag.config import Settings
from scholarrag.vectorstores.qdrant_store import RetrievedPoint


@dataclass(frozen=True)
class FaissArtifactStatus:
    index_path: Path
    sqlite_path: Path
    index_exists: bool
    sqlite_exists: bool


def _json_list(value: str | None) -> list[Any]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _matches_filters(payload: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True

    for key, expected in filters.items():
        if expected is None:
            continue

        actual = payload.get(key)
        if isinstance(actual, list):
            expected_values = expected if isinstance(expected, list) else [expected]
            if not set(actual).intersection(expected_values):
                return False
            continue

        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False

    return True


class FaissVectorStore:
    """FAISS + SQLite vector store for Colab-built OpenArXiv artifacts."""

    backend = "faiss"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._index: Any | None = None

    @property
    def collection_name(self) -> str:
        return f"faiss:{self.settings.faiss_index_path.name}"

    def artifact_status(self) -> FaissArtifactStatus:
        return FaissArtifactStatus(
            index_path=self.settings.faiss_index_path,
            sqlite_path=self.settings.faiss_sqlite_path,
            index_exists=self.settings.faiss_index_path.exists(),
            sqlite_exists=self.settings.faiss_sqlite_path.exists(),
        )

    def _get_index(self) -> Any:
        if self._index is None:
            if not self.settings.faiss_index_path.exists():
                raise FileNotFoundError(
                    f"FAISS index does not exist: {self.settings.faiss_index_path}"
                )

            import faiss

            self._index = faiss.read_index(str(self.settings.faiss_index_path))
            self._set_nprobe(self._index)
        return self._index

    def _set_nprobe(self, index: Any) -> None:
        base_index = getattr(index, "index", index)
        if hasattr(base_index, "nprobe"):
            nlist = int(getattr(base_index, "nlist", self.settings.faiss_nprobe))
            base_index.nprobe = min(self.settings.faiss_nprobe, nlist)

    def _connect(self) -> sqlite3.Connection:
        if not self.settings.faiss_sqlite_path.exists():
            raise FileNotFoundError(
                f"FAISS metadata SQLite DB does not exist: {self.settings.faiss_sqlite_path}"
            )
        connection = sqlite3.connect(self.settings.faiss_sqlite_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _metadata_table(self, connection: sqlite3.Connection) -> str:
        rows = connection.execute(
            "select name from sqlite_master where type = 'table' and name in ('papers', 'chunks')"
        ).fetchall()
        table_names = {str(row["name"]) for row in rows}
        if "papers" in table_names:
            return "papers"
        if "chunks" in table_names:
            return "chunks"
        raise RuntimeError("FAISS metadata DB must contain a papers or chunks table")

    def _payloads_by_vector_id(self, vector_ids: list[int]) -> dict[int, dict[str, Any]]:
        if not vector_ids:
            return {}

        placeholders = ",".join("?" for _ in vector_ids)
        with self._connect() as connection:
            table = self._metadata_table(connection)
            rows = connection.execute(
                f"select * from {table} where vector_id in ({placeholders})",
                vector_ids,
            ).fetchall()

        payloads: dict[int, dict[str, Any]] = {}
        for row in rows:
            raw = dict(row)
            vector_id = int(raw.pop("vector_id"))
            categories = _json_list(raw.pop("categories_json", None))
            authors = _json_list(raw.pop("authors_json", None))
            chunk_id = raw.get("chunk_id")
            payloads[vector_id] = {
                "point_id": str(raw.get("point_id") or vector_id),
                "paper_id": raw.get("paper_id"),
                "chunk_id": int(chunk_id) if chunk_id is not None else 0,
                "title": raw.get("title"),
                "text": raw.get("text") or "",
                "categories": categories,
                "update_date": raw.get("update_date"),
                "authors": authors,
            }
        return payloads

    def search(
        self,
        vector: list[float],
        *,
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedPoint]:
        index = self._get_index()
        if index.ntotal <= 0:
            return []

        query_vector = np.asarray([vector], dtype=np.float32)
        if query_vector.ndim != 2 or query_vector.shape[1] != self.settings.embedding_dimension:
            raise ValueError(
                "Query vector dimension mismatch: "
                f"expected {self.settings.embedding_dimension}, got {query_vector.shape}"
            )

        search_limit = limit
        if filters:
            search_limit = max(limit, limit * self.settings.faiss_filter_multiplier)
        search_limit = min(search_limit, int(index.ntotal))

        scores, ids = index.search(query_vector, search_limit)
        scored_ids = [
            (float(score), int(vector_id))
            for score, vector_id in zip(scores[0].tolist(), ids[0].tolist(), strict=False)
            if int(vector_id) >= 0
        ]
        vector_ids = [vector_id for _score, vector_id in scored_ids]
        payloads = self._payloads_by_vector_id(vector_ids)

        points: list[RetrievedPoint] = []
        for score, vector_id in scored_ids:
            payload = payloads.get(vector_id)
            if payload is None or not _matches_filters(payload, filters):
                continue

            point_id = str(payload.get("point_id") or vector_id)
            points.append(
                RetrievedPoint(
                    point_id=point_id,
                    score=float(score),
                    payload=payload,
                )
            )
            if len(points) >= limit:
                break

        return points
