import pytest

pytest.importorskip("pydantic_settings")

from scholarrag.llm import build_answer_messages  # noqa: E402
from scholarrag.retrieval import RetrievedChunk  # noqa: E402


def test_answer_prompt_contains_context_and_citation_instruction() -> None:
    chunk = RetrievedChunk(
        point_id="point-1",
        score=0.91,
        text="A retrieved scientific abstract.",
        metadata={"paper_id": "paper-1", "title": "Paper One", "categories": ["cs.CL"]},
    )

    messages = build_answer_messages("What does the paper say?", [chunk])

    assert messages[0]["role"] == "system"
    assert "Cite sources" in messages[0]["content"]
    assert "Paper One" in messages[1]["content"]
    assert "A retrieved scientific abstract." in messages[1]["content"]
