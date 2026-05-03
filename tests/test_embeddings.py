from scholarrag.embeddings import (
    format_document_for_embedding,
    format_fact_check_query,
)


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
