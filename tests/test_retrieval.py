import pytest

pytest.importorskip("pydantic_settings")

from scholarrag.config import Settings  # noqa: E402
from scholarrag.retrieval import RetrievalService  # noqa: E402
from scholarrag.vectorstores.qdrant_store import RetrievedPoint  # noqa: E402


class FakeEmbedder:
    def embed_query(self, question: str) -> list[float]:
        assert question == "scientific claim"
        return [0.1, 0.2, 0.3]


class FakeVectorStore:
    collection_name = "fake_collection"

    def search(self, vector, *, limit: int, filters=None):
        assert vector == [0.1, 0.2, 0.3]
        assert limit == 50
        assert filters == {"categories": ["cs.CL"]}
        return [
            RetrievedPoint(
                point_id="point-1",
                score=0.9,
                payload={
                    "paper_id": "paper-1",
                    "chunk_id": 0,
                    "text": "evidence",
                    "title": "Title",
                },
            )
        ]


def test_retrieval_uses_candidate_k_and_returns_sources() -> None:
    service = RetrievalService(
        settings=Settings(),
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
    )

    chunks, info = service.retrieve(
        "scientific claim",
        top_k=10,
        filters={"categories": ["cs.CL"]},
    )

    assert info["candidate_k"] == 50
    assert info["collection"] == "fake_collection"
    assert chunks[0].source_dict()["paper_id"] == "paper-1"
