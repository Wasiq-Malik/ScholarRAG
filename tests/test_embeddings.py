import pytest

from scholarrag.embeddings import (
    EmbeddingGemmaEmbedder,
    format_document_for_embedding,
    format_fact_check_query,
)
from scholarrag.config import Settings


def test_fact_check_query_prompt() -> None:
    assert (
        format_fact_check_query("Does aspirin reduce cardiovascular risk?")
        == "task: fact checking | query: Does aspirin reduce cardiovascular risk?"
    )


def test_document_prompt_uses_title_or_none() -> None:
    assert (
        format_document_for_embedding("Abstract text", title="Paper Title")
        == "title: Paper Title | text: Abstract text"
    )
    assert (
        format_document_for_embedding("Abstract text", title="")
        == "title: none | text: Abstract text"
    )


def test_local_hash_embedding_is_normalized_and_deterministic() -> None:
    embedder = EmbeddingGemmaEmbedder(
        Settings(embedding_model_id="local/hash-embedding", embedding_dimension=16)
    )

    first = embedder.embed_query("retrieval augmented generation")
    second = embedder.embed_query("retrieval augmented generation")

    assert first == second
    assert len(first) == 16
    assert sum(value * value for value in first) == pytest.approx(1.0)
