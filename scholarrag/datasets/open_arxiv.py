from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from scholarrag.chunking import OpenArxivPaper, clean_text


DATASET_NAME = "open-index/open-arxiv"


def _loads_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def normalize_open_arxiv_record(record: dict[str, Any]) -> OpenArxivPaper:
    authors_parsed = _loads_json(record.get("authors_parsed"), [])
    authors: list[str] = []
    for author in authors_parsed:
        if not isinstance(author, list) or len(author) < 2:
            continue
        last = str(author[0] or "").strip()
        first = str(author[1] or "").strip()
        name = " ".join(part for part in [first, last] if part)
        if name:
            authors.append(name)

    categories = [
        category.strip()
        for category in str(record.get("categories") or "").split()
        if category.strip()
    ]

    return OpenArxivPaper(
        paper_id=str(record.get("id") or "").strip(),
        title=clean_text(record.get("title")),
        abstract=clean_text(record.get("abstract")),
        categories=categories,
        update_date=str(record.get("update_date") or "").strip() or None,
        authors=authors,
    )


def iter_open_arxiv(
    *,
    dataset_name: str = DATASET_NAME,
    cache_dir: Path | None = None,
    limit: int | None = None,
    categories: list[str] | None = None,
) -> Iterator[OpenArxivPaper]:
    from datasets import load_dataset

    category_filter = set(categories or [])
    cache_dir_str = str(cache_dir) if cache_dir is not None else None
    dataset = load_dataset(dataset_name, split="train", streaming=True, cache_dir=cache_dir_str)

    yielded = 0
    for record in dataset:
        paper = normalize_open_arxiv_record(record)
        if not paper.paper_id or not paper.abstract:
            continue
        if category_filter and category_filter.isdisjoint(paper.categories):
            continue
        yield paper
        yielded += 1
        if limit is not None and yielded >= limit:
            break


def inspect_open_arxiv(
    *,
    dataset_name: str = DATASET_NAME,
    cache_dir: Path | None = None,
    rows: int = 3,
) -> list[OpenArxivPaper]:
    return list(
        iter_open_arxiv(
            dataset_name=dataset_name,
            cache_dir=cache_dir,
            limit=rows,
        )
    )
