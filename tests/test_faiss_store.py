import json
import sqlite3

import numpy as np
import pytest

pytest.importorskip("faiss")
pytest.importorskip("pydantic_settings")

import faiss  # noqa: E402

from scholarrag.config import Settings  # noqa: E402
from scholarrag.vectorstores.faiss_store import FaissVectorStore  # noqa: E402


def _write_test_index(path, vectors: np.ndarray, ids: np.ndarray) -> None:
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(vectors.shape[1]))
    index.add_with_ids(vectors.astype(np.float32), ids.astype(np.int64))
    faiss.write_index(index, str(path))


def _write_test_metadata(path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            create table chunks (
                vector_id integer primary key,
                point_id text unique not null,
                paper_id text not null,
                chunk_id integer not null,
                title text not null,
                text text not null,
                categories_json text not null,
                update_date text,
                authors_json text not null,
                created_at text not null
            );
            """
        )
        connection.executemany(
            """
            insert into chunks (
                vector_id, point_id, paper_id, chunk_id, title, text,
                categories_json, update_date, authors_json, created_at
            )
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    10,
                    "point-10",
                    "paper-10",
                    0,
                    "Neural Retrieval",
                    "retrieval evidence",
                    json.dumps(["cs.CL"]),
                    "2026-01-01",
                    json.dumps(["Jane Doe"]),
                    "now",
                ),
                (
                    20,
                    "point-20",
                    "paper-20",
                    0,
                    "Protein Folding",
                    "protein evidence",
                    json.dumps(["q-bio.BM"]),
                    "2026-01-02",
                    json.dumps([]),
                    "now",
                ),
            ],
        )


def test_faiss_store_searches_index_and_returns_sqlite_payload(tmp_path) -> None:
    index_path = tmp_path / "index.faiss"
    sqlite_path = tmp_path / "chunks.sqlite"
    vectors = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    ids = np.asarray([10, 20], dtype=np.int64)
    _write_test_index(index_path, vectors, ids)
    _write_test_metadata(sqlite_path)

    store = FaissVectorStore(
        Settings(
            embedding_dimension=3,
            faiss_index_path=index_path,
            faiss_sqlite_path=sqlite_path,
        )
    )

    results = store.search([1.0, 0.0, 0.0], limit=2)

    assert results[0].point_id == "point-10"
    assert results[0].payload["paper_id"] == "paper-10"
    assert results[0].payload["categories"] == ["cs.CL"]
    assert results[0].payload["text"] == "retrieval evidence"


def test_faiss_store_applies_metadata_filters(tmp_path) -> None:
    index_path = tmp_path / "index.faiss"
    sqlite_path = tmp_path / "chunks.sqlite"
    vectors = np.asarray([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]], dtype=np.float32)
    ids = np.asarray([10, 20], dtype=np.int64)
    _write_test_index(index_path, vectors, ids)
    _write_test_metadata(sqlite_path)

    store = FaissVectorStore(
        Settings(
            embedding_dimension=3,
            faiss_index_path=index_path,
            faiss_sqlite_path=sqlite_path,
        )
    )

    results = store.search([1.0, 0.0, 0.0], limit=2, filters={"categories": ["q-bio.BM"]})

    assert [result.point_id for result in results] == ["point-20"]
