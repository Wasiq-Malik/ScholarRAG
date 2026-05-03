from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from scholarrag.chunking import DocumentChunk, OpenArxivPaper


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def init_db(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                create table if not exists documents (
                    paper_id text primary key,
                    title text not null,
                    abstract text not null,
                    categories_json text not null,
                    update_date text,
                    authors_json text not null,
                    created_at text not null
                );

                create table if not exists chunks (
                    point_id text primary key,
                    paper_id text not null,
                    chunk_index integer not null,
                    title text not null,
                    text text not null,
                    categories_json text not null,
                    update_date text,
                    created_at text not null
                );

                create table if not exists ingestion_jobs (
                    job_id text primary key,
                    status text not null,
                    limit_value integer,
                    categories_json text not null,
                    indexed_documents integer not null default 0,
                    indexed_chunks integer not null default 0,
                    error text,
                    created_at text not null,
                    updated_at text not null
                );

                create table if not exists query_logs (
                    query_id text primary key,
                    question text not null,
                    answer text not null,
                    sources_json text not null,
                    model text not null,
                    created_at text not null
                );
                """
            )

    def upsert_document(self, paper: OpenArxivPaper) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                insert into documents (
                    paper_id, title, abstract, categories_json, update_date,
                    authors_json, created_at
                )
                values (?, ?, ?, ?, ?, ?, ?)
                on conflict(paper_id) do update set
                    title=excluded.title,
                    abstract=excluded.abstract,
                    categories_json=excluded.categories_json,
                    update_date=excluded.update_date,
                    authors_json=excluded.authors_json
                """,
                (
                    paper.paper_id,
                    paper.title,
                    paper.abstract,
                    json.dumps(paper.categories),
                    paper.update_date,
                    json.dumps(paper.authors or []),
                    utc_now(),
                ),
            )

    def upsert_chunks(self, chunks: Iterable[DocumentChunk]) -> None:
        rows = [
            (
                chunk.point_id,
                chunk.paper_id,
                chunk.chunk_index,
                chunk.title,
                chunk.text,
                json.dumps(chunk.categories),
                chunk.update_date,
                utc_now(),
            )
            for chunk in chunks
        ]
        if not rows:
            return
        with self.connect() as connection:
            connection.executemany(
                """
                insert into chunks (
                    point_id, paper_id, chunk_index, title, text,
                    categories_json, update_date, created_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(point_id) do update set
                    paper_id=excluded.paper_id,
                    chunk_index=excluded.chunk_index,
                    title=excluded.title,
                    text=excluded.text,
                    categories_json=excluded.categories_json,
                    update_date=excluded.update_date
                """,
                rows,
            )

    def create_ingestion_job(
        self,
        *,
        job_id: str,
        limit: int | None,
        categories: list[str] | None,
        status: str = "queued",
    ) -> None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """
                insert into ingestion_jobs (
                    job_id, status, limit_value, categories_json,
                    created_at, updated_at
                )
                values (?, ?, ?, ?, ?, ?)
                on conflict(job_id) do update set
                    status=excluded.status,
                    limit_value=excluded.limit_value,
                    categories_json=excluded.categories_json,
                    updated_at=excluded.updated_at,
                    error=null
                """,
                (job_id, status, limit, json.dumps(categories or []), now, now),
            )

    def update_ingestion_job(
        self,
        *,
        job_id: str,
        status: str,
        indexed_documents: int,
        indexed_chunks: int,
        error: str | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                update ingestion_jobs
                set status = ?,
                    indexed_documents = ?,
                    indexed_chunks = ?,
                    error = ?,
                    updated_at = ?
                where job_id = ?
                """,
                (status, indexed_documents, indexed_chunks, error, utc_now(), job_id),
            )

    def log_query(
        self,
        *,
        query_id: str,
        question: str,
        answer: str,
        sources: list[dict[str, object]],
        model: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                insert into query_logs (
                    query_id, question, answer, sources_json, model, created_at
                )
                values (?, ?, ?, ?, ?, ?)
                """,
                (query_id, question, answer, json.dumps(sources), model, utc_now()),
            )
