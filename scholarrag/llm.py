from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from scholarrag.config import Settings
from scholarrag.retrieval import RetrievedChunk


SYSTEM_PROMPT = """You are ScholarRAG, a scientific research assistant.
Answer only from the provided retrieved context. Cite sources with bracketed
numbers like [1] and [2]. If the context does not contain enough evidence,
say that the retrieved evidence is insufficient."""


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
        "Write a concise grounded answer with source citations."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


class VLLMAnswerGenerator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: Any | None = None

    @property
    def model(self) -> str:
        return self.settings.vllm_model

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.settings.vllm_base_url,
                api_key=self.settings.vllm_api_key,
            )
        return self._client

    def answer(self, *, question: str, chunks: Sequence[RetrievedChunk]) -> str:
        if not chunks:
            return "The retrieved evidence is insufficient to answer this question."

        try:
            response = self._get_client().chat.completions.create(
                model=self.settings.vllm_model,
                messages=build_answer_messages(question, chunks),
                temperature=self.settings.vllm_temperature,
                max_tokens=self.settings.vllm_max_tokens,
            )
        except Exception as exc:
            return (
                "Answer generation is unavailable, but retrieval succeeded. "
                f"Configure an OpenAI-compatible LLM endpoint to generate answers. ({exc})"
            )
        return str(response.choices[0].message.content or "").strip()
