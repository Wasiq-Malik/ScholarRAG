from scholarrag.datasets.open_arxiv import normalize_open_arxiv_record


def test_normalize_open_arxiv_record() -> None:
    paper = normalize_open_arxiv_record(
        {
            "id": "2401.12345",
            "title": "  Test   Paper ",
            "abstract": "  A   useful abstract. ",
            "categories": "cs.CL cs.IR",
            "update_date": "2026-01-01",
            "authors_parsed": '[["Doe", "Jane", ""]]',
        }
    )

    assert paper.paper_id == "2401.12345"
    assert paper.title == "Test Paper"
    assert paper.abstract == "A useful abstract."
    assert paper.categories == ["cs.CL", "cs.IR"]
    assert paper.authors == ["Jane Doe"]
