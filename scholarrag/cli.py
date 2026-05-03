from __future__ import annotations

import argparse
from uuid import uuid4

from scholarrag.config import get_settings
from scholarrag.datasets.open_arxiv import inspect_open_arxiv
from scholarrag.embeddings import EmbeddingGemmaEmbedder
from scholarrag.ingest import OpenArxivIndexer
from scholarrag.storage import SQLiteStore
from scholarrag.vectorstores.qdrant_store import QdrantVectorStore


def _cmd_inspect_open_arxiv(args: argparse.Namespace) -> int:
    settings = get_settings()
    papers = inspect_open_arxiv(
        dataset_name=settings.open_arxiv_dataset,
        cache_dir=settings.hf_cache_dir,
        rows=args.rows,
    )
    for index, paper in enumerate(papers, start=1):
        print(f"\n[{index}] id={paper.paper_id}")
        print(f"title: {paper.title}")
        print(f"categories: {' '.join(paper.categories)}")
        print(f"updated: {paper.update_date or 'unknown'}")
        print(f"abstract: {paper.abstract[:500]}")
    return 0


def _cmd_index_open_arxiv(args: argparse.Namespace) -> int:
    settings = get_settings()
    store = SQLiteStore(settings.sqlite_path)
    store.init_db()
    job_id = args.job_id or str(uuid4())
    store.create_ingestion_job(
        job_id=job_id,
        limit=args.limit,
        categories=args.category,
        status="queued",
    )
    indexer = OpenArxivIndexer(
        settings=settings,
        store=store,
        embedder=EmbeddingGemmaEmbedder(settings),
        vector_store=QdrantVectorStore(settings),
    )
    result = indexer.index(limit=args.limit, categories=args.category, job_id=job_id)
    print(
        f"job_id={result.job_id} status={result.status} "
        f"documents={result.indexed_documents} chunks={result.indexed_chunks}"
    )
    if result.error:
        print(f"error={result.error}")
        return 1
    return 0


def _cmd_serve_api(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "scholarrag.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scholarrag")
    subcommands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subcommands.add_parser(
        "inspect-open-arxiv",
        help="Stream and print sample OpenArXiv records.",
    )
    inspect_parser.add_argument("--rows", type=int, default=3)
    inspect_parser.set_defaults(func=_cmd_inspect_open_arxiv)

    index_parser = subcommands.add_parser(
        "index-open-arxiv",
        help="Index OpenArXiv abstracts into Qdrant using EmbeddingGemma.",
    )
    index_parser.add_argument("--limit", type=int, default=None)
    index_parser.add_argument("--category", action="append", default=None)
    index_parser.add_argument("--job-id", default=None)
    index_parser.set_defaults(func=_cmd_index_open_arxiv)

    serve_parser = subcommands.add_parser("serve-api", help="Run the ScholarRAG API.")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8080)
    serve_parser.add_argument("--reload", action="store_true")
    serve_parser.set_defaults(func=_cmd_serve_api)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
