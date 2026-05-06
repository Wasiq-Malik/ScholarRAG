import pytest

pytest.importorskip("pydantic_settings")

from scholarrag.config import Settings  # noqa: E402


def test_llm_env_aliases(monkeypatch) -> None:
    monkeypatch.setenv("SCHOLARRAG_LLM_BASE_URL", "https://generativelanguage.example/v1beta")
    monkeypatch.setenv("SCHOLARRAG_LLM_MODEL", "test-model")
    monkeypatch.setenv("SCHOLARRAG_LLM_TEMPERATURE", "0.1")
    monkeypatch.setenv("SCHOLARRAG_LLM_MAX_TOKENS", "123")

    settings = Settings()

    assert settings.gemini_base_url == "https://generativelanguage.example/v1beta"
    assert settings.gemini_model == "test-model"
    assert settings.gemini_temperature == 0.1
    assert settings.gemini_max_tokens == 123


def test_gemini_env_aliases(monkeypatch) -> None:
    monkeypatch.setenv("SCHOLARRAG_GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("SCHOLARRAG_GEMINI_BASE_URL", "https://generativelanguage.example/v1beta")
    monkeypatch.setenv("SCHOLARRAG_GEMINI_MODEL", "gemma-test")

    settings = Settings()

    assert settings.gemini_api_key == "gemini-key"
    assert settings.gemini_base_url == "https://generativelanguage.example/v1beta"
    assert settings.gemini_model == "gemma-test"
