from __future__ import annotations

from collections.abc import Sequence
import json
from typing import Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from scholarrag.config import Settings
from scholarrag.retrieval import RetrievedChunk


SYSTEM_PROMPT = """You are ScholarRAG, a scientific research assistant.
Treat the user's query as a scientific claim-checking or prior-work lookup task.
Answer using only the provided retrieved context. When the query asks whether a
method, result, or system exists, answer by stating whether the existing literature
 provides evidence that it has been done, and name the relevant papers.
Support claims with inline citations that use the provided source numbers in
square brackets, for example [1] or [2]. Use a direct, confident tone grounded
in the literature. If the papers clearly show the work has been done, say so
plainly. If the papers do not show it, say that no evidence for it was found.
Do not hedge with meta phrasing such as "the retrieved evidence suggests" or
"based on the provided context" unless truly necessary."""


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
        "Write a concise grounded answer for the question above. Use only the "
        "retrieved context. If the query is asking whether prior scientific work "
        "exists, answer that directly and identify the papers that support the "
        "answer. If none of the retrieved papers actually show the requested work, "
        "say clearly that no evidence for it was found instead of generalizing from "
        "nearby topics. Cite supporting sources inline with exact bracketed "
        "references like [1] and [4]. Make sure you provide a clear, direct answer, do not try to give extra information that the user did not ask for."
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

    def stream_answer(
        self,
        *,
        question: str,
        chunks: Sequence[RetrievedChunk],
    ) -> Iterator[str]:
        if not chunks:
            yield "The retrieved evidence is insufficient to answer this question."
            return
        if not self.settings.gemini_api_key:
            yield (
                "Answer generation is unavailable, but retrieval succeeded. "
                "Set GEMINI_API_KEY or SCHOLARRAG_GEMINI_API_KEY to generate answers."
            )
            return

        system_message, user_message = build_answer_messages(question, chunks)
        try:
            emitted = False
            for piece in self._stream_generate(
                system_prompt=system_message["content"],
                user_prompt=user_message["content"],
            ):
                if piece:
                    emitted = True
                    yield piece
            if not emitted:
                return
        except RuntimeError as exc:
            yield (
                "Answer generation is unavailable, but retrieval succeeded. "
                f"Gemini API could not generate an answer. ({exc})"
            )

    def _generate(self, *, system_prompt: str, user_prompt: str) -> str:
        model = self.settings.gemini_model.removeprefix("models/")
        url = (
            f"{self.settings.gemini_base_url.rstrip('/')}/models/"
            f"{model}:generateContent"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": self._generation_config(),
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

    def _stream_generate(self, *, system_prompt: str, user_prompt: str) -> Iterator[str]:
        model = self.settings.gemini_model.removeprefix("models/")
        url = (
            f"{self.settings.gemini_base_url.rstrip('/')}/models/"
            f"{model}:streamGenerateContent?alt=sse"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": self._generation_config(),
        }
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "x-goog-api-key": self.settings.gemini_api_key or "",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=120) as response:
                data_lines: list[str] = []
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                    if not line:
                        if data_lines:
                            payload_text = "\n".join(data_lines)
                            data_lines = []
                            if payload_text == "[DONE]":
                                return
                            yield from self._extract_stream_text(payload_text)
                        continue
                    if line.startswith("data:"):
                        data_lines.append(line[len("data:") :].lstrip())
                if data_lines:
                    payload_text = "\n".join(data_lines)
                    if payload_text != "[DONE]":
                        yield from self._extract_stream_text(payload_text)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini API request failed: HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Gemini API request failed: {exc.reason}") from exc
        except Exception as exc:
            raise RuntimeError(f"Gemini API request failed: {exc}") from exc

    def _extract_stream_text(self, payload_text: str) -> Iterator[str]:
        try:
            data = json.loads(payload_text)
        except json.JSONDecodeError:
            return

        candidates = data.get("candidates") or []
        for candidate in candidates:
            parts = candidate.get("content", {}).get("parts") or []
            for part in parts:
                text = part.get("text")
                if isinstance(text, str) and text:
                    yield text

    def _generation_config(self) -> dict[str, object]:
        config: dict[str, object] = {
            "temperature": self.settings.gemini_temperature,
            "maxOutputTokens": self.settings.gemini_max_tokens,
        }
        model = self.settings.gemini_model.removeprefix("models/")
        if model.startswith("gemini-3") and self.settings.gemini_thinking_level:
            config["thinkingConfig"] = {
                "thinkingLevel": self.settings.gemini_thinking_level,
            }
        return config
