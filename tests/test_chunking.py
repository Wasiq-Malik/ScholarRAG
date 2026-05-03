from scholarrag.chunking import OpenArxivPaper, chunk_paper, split_text


def test_split_text_preserves_overlap_without_empty_chunks() -> None:
    text = " ".join(f"token{i}" for i in range(80))

    chunks = split_text(text, chunk_chars=120, overlap_chars=20)

    assert len(chunks) > 1
    assert all(chunks)
    assert all(len(chunk) <= 120 for chunk in chunks)


def test_chunk_paper_adds_stable_metadata() -> None:
    paper = OpenArxivPaper(
        paper_id="2401.12345",
        title=" A Useful Paper ",
        abstract=" This paper studies scientific retrieval. ",
        categories=["cs.CL", "cs.IR"],
        update_date="2026-01-01",
    )

    chunks = chunk_paper(paper, chunk_chars=1000, overlap_chars=100)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.paper_id == "2401.12345"
    assert chunk.chunk_index == 0
    assert chunk.title == "A Useful Paper"
    assert chunk.categories == ["cs.CL", "cs.IR"]
    assert chunk.payload()["paper_id"] == "2401.12345"
    assert chunk.payload()["text"]
