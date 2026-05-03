from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scholarrag.config import Settings


FACT_CHECK_QUERY_PROMPT = "task: fact checking | query: "
DOCUMENT_PROMPT_TEMPLATE = "title: {title} | text: "


def format_fact_check_query(query: str, *, prompt: str = FACT_CHECK_QUERY_PROMPT) -> str:
    return f"{prompt}{query.strip()}"


def format_document_for_embedding(
    text: str,
    *,
    title: str | None,
    prompt_template: str = DOCUMENT_PROMPT_TEMPLATE,
) -> str:
    title_value = "none" if not title or not title.strip() else " ".join(title.split())
    return f"{prompt_template.format(title=title_value)}{text.strip()}"


class EmbeddingGemmaEmbedder:
    """Lazy SentenceTransformers wrapper for EmbeddingGemma fact-check retrieval."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: Any | None = None

    @property
    def model_id(self) -> str:
        return self.settings.embedding_model_id

    @property
    def dimension(self) -> int:
        return self.settings.embedding_dimension

    def _load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            model_kwargs: dict[str, object] = {}
            try:
                import torch

                if torch.cuda.is_available():
                    model_kwargs["torch_dtype"] = torch.bfloat16
            except Exception:
                model_kwargs = {}

            self._model = SentenceTransformer(
                self.settings.embedding_model_id,
                token=self.settings.hf_token,
                truncate_dim=self.settings.embedding_dimension,
                model_kwargs=model_kwargs,
            )
        return self._model

    def embed_queries(self, queries: Sequence[str]) -> list[list[float]]:
        formatted = [
            format_fact_check_query(query, prompt=self.settings.embedding_query_prompt)
            for query in queries
        ]
        return self._encode(formatted)

    def embed_query(self, query: str) -> list[float]:
        return self.embed_queries([query])[0]

    def embed_documents(self, documents: Sequence[tuple[str, str | None]]) -> list[list[float]]:
        formatted = [
            format_document_for_embedding(
                text,
                title=title,
                prompt_template=self.settings.embedding_document_prompt_template,
            )
            for text, title in documents
        ]
        return self._encode(formatted)

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        import numpy as np

        model = self._load_model()
        embeddings = model.encode(
            list(texts),
            batch_size=self.settings.embedding_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self.settings.embedding_dimension:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"expected {self.settings.embedding_dimension}, got {array.shape}"
            )
        return array.tolist()
