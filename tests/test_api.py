import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pydantic_settings")

from fastapi.testclient import TestClient  # noqa: E402

from scholarrag.api import app as app_module  # noqa: E402
from scholarrag.config import Settings  # noqa: E402
from scholarrag.retrieval import RetrievedChunk  # noqa: E402


class FakeStore:
    def log_query(self, **kwargs) -> None:
        self.logged = kwargs


def fake_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        point_id="point-1",
        score=0.8,
        text="retrieved evidence",
        metadata={"paper_id": "paper-1", "chunk_id": 0, "title": "Paper"},
    )


class FakeRetriever:
    def retrieve(self, question: str, *, top_k: int, filters: dict):
        assert question == "What is retrieved?"
        assert top_k == 1
        assert filters == {}
        return ([fake_chunk()], {"candidate_k": 50, "returned_k": 1, "collection": "fake", "reranker": None})


class FakeGenerator:
    def answer(self, *, question: str, chunks: list[RetrievedChunk]) -> str:
        return "Grounded answer [1]."

    def stream_answer(self, *, question: str, chunks: list[RetrievedChunk]):
        yield "Grounded "
        yield "answer [1]."


def test_query_endpoint_returns_answer_sources_and_models(monkeypatch) -> None:
    settings = Settings()
    fake_store = FakeStore()

    def fake_services():
        return (
            settings,
            fake_store,
            object(),
            object(),
            FakeRetriever(),
            FakeGenerator(),
        )

    monkeypatch.setattr(app_module, "get_services", fake_services)
    client = TestClient(app_module.create_app())

    response = client.post("/query", json={"question": "What is retrieved?", "top_k": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Grounded answer [1]."
    assert body["sources"][0]["paper_id"] == "paper-1"
    assert body["sources"][0]["arxiv_url"] == "https://arxiv.org/abs/paper-1"
    assert body["sources"][0]["confidence_score"] == 0.8
    assert body["sources"][0]["text"] == "retrieved evidence"
    assert body["models"]["embedding"] == settings.embedding_model_id
    assert body["models"]["llm"] == settings.gemini_model


def test_retrieve_endpoint_returns_sources_without_answer(monkeypatch) -> None:
    settings = Settings()

    def fake_services():
        return (
            settings,
            FakeStore(),
            object(),
            object(),
            FakeRetriever(),
            FakeGenerator(),
        )

    monkeypatch.setattr(app_module, "get_services", fake_services)
    client = TestClient(app_module.create_app())

    response = client.post("/retrieve", json={"question": "What is retrieved?", "top_k": 1})

    assert response.status_code == 200
    body = response.json()
    assert "answer" not in body
    assert body["sources"][0]["paper_id"] == "paper-1"


def test_answer_stream_endpoint_streams_sse(monkeypatch) -> None:
    settings = Settings()
    fake_store = FakeStore()

    def fake_services():
        return (
            settings,
            fake_store,
            object(),
            object(),
            FakeRetriever(),
            FakeGenerator(),
        )

    monkeypatch.setattr(app_module, "get_services", fake_services)
    client = TestClient(app_module.create_app())

    response = client.post(
        "/answer/stream",
        json={
            "question": "What is retrieved?",
            "sources": [
                {
                    "point_id": "point-1",
                    "score": 0.8,
                    "confidence_score": 0.8,
                    "paper_id": "paper-1",
                    "arxiv_url": "https://arxiv.org/abs/paper-1",
                    "chunk_id": 0,
                    "title": "Paper",
                    "categories": [],
                    "update_date": None,
                    "text": "retrieved evidence",
                    "text_preview": "retrieved evidence",
                }
            ],
        },
    )

    assert response.status_code == 200
    assert "event: chunk" in response.text
    assert "Grounded " in response.text
    assert "event: done" in response.text


def test_health_reports_faiss_backend(monkeypatch, tmp_path) -> None:
    index_path = tmp_path / "index.faiss"
    sqlite_path = tmp_path / "chunks.sqlite"
    index_path.write_bytes(b"index")
    sqlite_path.write_bytes(b"sqlite")
    settings = Settings(
        vector_backend="faiss",
        faiss_index_path=index_path,
        faiss_sqlite_path=sqlite_path,
    )

    monkeypatch.setattr(app_module, "get_settings", lambda: settings)
    client = TestClient(app_module.create_app())

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["vector_backend"] == "faiss"
    assert body["faiss_index_path"] == str(index_path)
    assert body["faiss_sqlite_path"] == str(sqlite_path)
    assert body["faiss_index_exists"] is True
    assert body["faiss_sqlite_exists"] is True


def test_faiss_backend_rejects_api_ingestion(monkeypatch) -> None:
    settings = Settings(vector_backend="faiss")

    def fake_services():
        return (
            settings,
            FakeStore(),
            object(),
            object(),
            FakeRetriever(),
            FakeGenerator(),
        )

    monkeypatch.setattr(app_module, "get_services", fake_services)
    client = TestClient(app_module.create_app())

    response = client.post("/ingest/open-arxiv", json={"limit": 1})

    assert response.status_code == 400
    assert "Qdrant only" in response.json()["detail"]
