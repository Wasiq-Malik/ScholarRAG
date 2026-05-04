import pytest

pytest.importorskip("pydantic_settings")

from scholarrag.config import Settings  # noqa: E402


def test_llm_env_aliases(monkeypatch) -> None:
    monkeypatch.setenv("SCHOLARRAG_LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("SCHOLARRAG_LLM_API_KEY", "test-key")
    monkeypatch.setenv("SCHOLARRAG_LLM_MODEL", "test-model")

    settings = Settings()

    assert settings.vllm_base_url == "https://llm.example/v1"
    assert settings.vllm_api_key == "test-key"
    assert settings.vllm_model == "test-model"


def test_vllm_env_aliases_still_work(monkeypatch) -> None:
    monkeypatch.setenv("SCHOLARRAG_VLLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("SCHOLARRAG_VLLM_API_KEY", "vllm-key")
    monkeypatch.setenv("SCHOLARRAG_VLLM_MODEL", "Qwen/test")

    settings = Settings()

    assert settings.vllm_base_url == "http://localhost:8000/v1"
    assert settings.vllm_api_key == "vllm-key"
    assert settings.vllm_model == "Qwen/test"
