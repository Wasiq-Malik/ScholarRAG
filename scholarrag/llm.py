from __future__ import annotations

from collections.abc import Sequence
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from scholarrag.config import Settings
from scholarrag.retrieval import RetrievedChunk


SYSTEM_PROMPT = """You are ScholarRAG, a scientific research assistant.
Answer only from the provided retrieved context. Cite sources with bracketed
numbers like [1] and [2]. If the context does not contain enough evidence,
say that the retrieved evidence is insufficient. Return only the final answer.
Do not show reasoning steps, analysis notes, bullet plans, or restate these
instructions."""


def build_context(chunks: Sequence[RetrievedChunk]) -> str:
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        title = chunk.metadata.get("title") or "Untitled"
        paper_id = chunk.metadata.get("paper_id") or "unknown"
        categories = ", ".join(chunk.metadata.get("categories") or [])
        blocks.append(
            "\n".join(
                [
                    f"[{index}] paper_id={paper_id}",
                    f"title={title}",
                    f"categories={categories}",
                    f"text={chunk.text}",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_answer_messages(question: str, chunks: Sequence[RetrievedChunk]) -> list[dict[str, str]]:
    context = build_context(chunks)
    user_prompt = (
        "Question:\n"
        f"{question.strip()}\n\n"
        "Retrieved context:\n"
        f"{context if context else 'No context was retrieved.'}\n\n"
        "Write a concise grounded answer with source citations. Output only the "
        "answer text. Do not include markdown bullets, labels, analysis, or a "
        "restatement of the question/context."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


class GeminiAnswerGenerator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def model(self) -> str:
        return self.settings.gemini_model

    def answer(self, *, question: str, chunks: Sequence[RetrievedChunk]) -> str:
        if not chunks:
            return "The retrieved evidence is insufficient to answer this question."
        if not self.settings.gemini_api_key:
            return (
                "Answer generation is unavailable, but retrieval succeeded. "
                "Set GEMINI_API_KEY or SCHOLARRAG_GEMINI_API_KEY to generate answers."
            )

        system_message, user_message = build_answer_messages(question, chunks)
        try:
            response = self._generate(
                system_prompt=system_message["content"],
                user_prompt=user_message["content"],
            )
        except RuntimeError as exc:
            return (
                "Answer generation is unavailable, but retrieval succeeded. "
                f"Gemini API could not generate an answer. ({exc})"
            )
        return response.strip()

    def _generate(self, *, system_prompt: str, user_prompt: str) -> str:
        model = self.settings.gemini_model.removeprefix("models/")
        url = (
            f"{self.settings.gemini_base_url.rstrip('/')}/models/"
            f"{model}:generateContent"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": self.settings.gemini_temperature,
                "maxOutputTokens": self.settings.gemini_max_tokens,
            },
        }
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.settings.gemini_api_key or "",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini API request failed: HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Gemini API request failed: {exc.reason}") from exc
        except Exception as exc:
            raise RuntimeError(f"Gemini API request failed: {exc}") from exc

        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts") or []
        return "".join(str(part.get("text") or "") for part in parts)
