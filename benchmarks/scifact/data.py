from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "mteb/scifact"


@dataclass
class SciFactEvalData:
    query_ids: list[str]
    query_texts: list[str]
    corpus_ids: list[str]
    corpus_texts: list[str]
    positives_by_query: dict[str, dict[str, float]]
    split: str
    document_mode: str


def _build_document_text(title: str, abstract: str, document_mode: str) -> str:
    title = (title or "").strip()
    abstract = (abstract or "").strip()

    if document_mode == "abstract":
        return abstract
    if document_mode == "title_abstract":
        if title and abstract:
            return f"{title}\n\n{abstract}"
        return title or abstract
    raise ValueError(f"Unsupported document_mode: {document_mode}")


def load_scifact(
    *,
    split: str = "test",
    document_mode: str = "abstract",
    cache_dir: Path | None = None,
    max_queries: int | None = None,
) -> SciFactEvalData:
    cache_dir_str = str(cache_dir) if cache_dir is not None else None

    corpus = load_dataset(
        DATASET_NAME,
        "corpus",
        split="corpus",
        cache_dir=cache_dir_str,
    )
    queries = load_dataset(
        DATASET_NAME,
        "queries",
        split="queries",
        cache_dir=cache_dir_str,
    )
    qrels = load_dataset(
        DATASET_NAME,
        split=split,
        cache_dir=cache_dir_str,
    )

    positives_by_query: dict[str, dict[str, float]] = defaultdict(dict)
    ordered_query_ids: list[str] = []
    seen_query_ids: set[str] = set()

    for row in qrels:
        query_id = row["query-id"]
        corpus_id = row["corpus-id"]
        positives_by_query[query_id][corpus_id] = float(row["score"])
        if query_id not in seen_query_ids:
            ordered_query_ids.append(query_id)
            seen_query_ids.add(query_id)

    if max_queries is not None:
        ordered_query_ids = ordered_query_ids[:max_queries]
        positives_by_query = {
            query_id: positives_by_query[query_id] for query_id in ordered_query_ids
        }

    query_text_lookup = {row["_id"]: row["text"].strip() for row in queries}
    query_texts = [query_text_lookup[query_id] for query_id in ordered_query_ids]

    corpus_ids: list[str] = []
    corpus_texts: list[str] = []
    for row in corpus:
        corpus_ids.append(row["_id"])
        corpus_texts.append(
            _build_document_text(
                title=row["title"],
                abstract=row["text"],
                document_mode=document_mode,
            )
        )

    return SciFactEvalData(
        query_ids=ordered_query_ids,
        query_texts=query_texts,
        corpus_ids=corpus_ids,
        corpus_texts=corpus_texts,
        positives_by_query=positives_by_query,
        split=split,
        document_mode=document_mode,
    )
