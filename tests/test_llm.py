from __future__ import annotations

import json

from scholarrag.config import Settings
from scholarrag.llm import GeminiAnswerGenerator


def test_extract_stream_text_yields_candidate_text_parts() -> None:
    generator = GeminiAnswerGenerator(Settings())
    payload = json.dumps(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Grounded "},
                            {"text": "answer [1]."},
                        ]
                    }
                }
            ]
        }
    )

    pieces = list(generator._extract_stream_text(payload))

    assert pieces == ["Grounded ", "answer [1]."]


def test_extract_stream_text_ignores_invalid_json() -> None:
    generator = GeminiAnswerGenerator(Settings())

    pieces = list(generator._extract_stream_text("not-json"))

    assert pieces == []
