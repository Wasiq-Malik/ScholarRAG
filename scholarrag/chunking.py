from __future__ import annotations

from dataclasses import dataclass
from uuid import NAMESPACE_URL, uuid5


@dataclass(frozen=True)
class OpenArxivPaper:
    paper_id: str
    title: str
    abstract: str
    categories: list[str]
    update_date: str | None = None
    authors: list[str] | None = None


@dataclass(frozen=True)
class DocumentChunk:
    point_id: str
    paper_id: str
    chunk_index: int
    title: str
    text: str
    categories: list[str]
    update_date: str | None = None

    def payload(self) -> dict[str, object]:
        return {
            "paper_id": self.paper_id,
            "chunk_id": self.chunk_index,
            "title": self.title,
            "text": self.text,
            "categories": self.categories,
            "update_date": self.update_date,
        }


def clean_text(value: str | None) -> str:
    return " ".join((value or "").split())


def build_chunk_text(paper: OpenArxivPaper) -> str:
    title = clean_text(paper.title)
    abstract = clean_text(paper.abstract)
    if title and abstract:
        return f"{title}\n\n{abstract}"
    return title or abstract


def split_text(text: str, *, chunk_chars: int, overlap_chars: int) -> list[str]:
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be non-negative")
    if overlap_chars >= chunk_chars:
        raise ValueError("overlap_chars must be smaller than chunk_chars")

    text = clean_text(text)
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    step = chunk_chars - overlap_chars
    while start < len(text):
        raw_end = min(start + chunk_chars, len(text))
        end = raw_end
        if raw_end < len(text):
            whitespace = text.rfind(" ", start + max(chunk_chars // 2, 1), raw_end)
            if whitespace > start:
                end = whitespace

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if raw_end >= len(text):
            break
        start = max(end - overlap_chars, start + step)

    return chunks


def stable_point_id(paper_id: str, chunk_index: int) -> str:
    return str(uuid5(NAMESPACE_URL, f"scholarrag:open-arxiv:{paper_id}:{chunk_index}"))


def chunk_paper(
    paper: OpenArxivPaper,
    *,
    chunk_chars: int,
    overlap_chars: int,
) -> list[DocumentChunk]:
    text = build_chunk_text(paper)
    chunks = split_text(text, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
    return [
        DocumentChunk(
            point_id=stable_point_id(paper.paper_id, index),
            paper_id=paper.paper_id,
            chunk_index=index,
            title=clean_text(paper.title),
            text=chunk,
            categories=paper.categories,
            update_date=paper.update_date,
        )
        for index, chunk in enumerate(chunks)
    ]
