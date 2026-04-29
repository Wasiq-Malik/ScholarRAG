from __future__ import annotations

import gc
from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


SCIENTIFIC_RERANK_INSTRUCTION = (
    "Given a scientific claim or research question, judge whether the abstract "
    "provides relevant evidence."
)


@dataclass(frozen=True)
class RerankerSpec:
    key: str
    model_id: str
    family: str
    approx_params: str
    max_length: int
    default_batch_size: int
    reranker_kind: str
    notes: str
    trust_remote_code: bool = False
    instruction: str | None = None


RERANKER_SPECS: dict[str, RerankerSpec] = {
    "msmarco_minilm_l12_v2": RerankerSpec(
        key="msmarco_minilm_l12_v2",
        model_id="cross-encoder/ms-marco-MiniLM-L12-v2",
        family="CrossEncoder MiniLM",
        approx_params="120M",
        max_length=512,
        default_batch_size=64,
        reranker_kind="sequence_classification",
        notes="Small, fast English cross-encoder reranker trained on MS MARCO.",
    ),
    "gte_reranker_modernbert_base": RerankerSpec(
        key="gte_reranker_modernbert_base",
        model_id="Alibaba-NLP/gte-reranker-modernbert-base",
        family="GTE ModernBERT",
        approx_params="149M",
        max_length=1024,
        default_batch_size=32,
        reranker_kind="sequence_classification",
        notes="ModernBERT-based reranker with long-context support and strong BEIR results.",
    ),
    "jina_reranker_v2_base_multilingual": RerankerSpec(
        key="jina_reranker_v2_base_multilingual",
        model_id="jinaai/jina-reranker-v2-base-multilingual",
        family="Jina",
        approx_params="278M",
        max_length=1024,
        default_batch_size=16,
        reranker_kind="sequence_classification",
        notes="Classic multilingual cross-encoder reranker from Jina. Open-weight for research/non-commercial use under CC-BY-NC-4.0.",
        trust_remote_code=True,
    ),
    "bge_reranker_v2_m3": RerankerSpec(
        key="bge_reranker_v2_m3",
        model_id="BAAI/bge-reranker-v2-m3",
        family="BGE",
        approx_params="568M",
        max_length=1024,
        default_batch_size=16,
        reranker_kind="sequence_classification",
        notes="Strong multilingual BGE reranker baseline.",
    ),
    "qwen3_reranker_0_6b": RerankerSpec(
        key="qwen3_reranker_0_6b",
        model_id="Qwen/Qwen3-Reranker-0.6B",
        family="Qwen3",
        approx_params="0.6B",
        max_length=2048,
        default_batch_size=8,
        reranker_kind="qwen_cross_encoder",
        notes="Instruction-aware Qwen reranker with strong open multilingual reranking results.",
        instruction=SCIENTIFIC_RERANK_INSTRUCTION,
    ),
}


def available_reranker_keys() -> list[str]:
    return list(RERANKER_SPECS.keys())


def default_reranker_keys() -> list[str]:
    return [
        "msmarco_minilm_l12_v2",
        "gte_reranker_modernbert_base",
        "jina_reranker_v2_base_multilingual",
        "bge_reranker_v2_m3",
        "qwen3_reranker_0_6b",
    ]


def resolve_torch_dtype(dtype_name: str, device: str) -> torch.dtype | None:
    if dtype_name == "auto":
        if device == "cuda":
            return torch.float16
        return None
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


class BaseReranker:
    def __init__(
        self,
        spec: RerankerSpec,
        *,
        device: str,
        batch_size: int,
        dtype_name: str,
    ) -> None:
        self.spec = spec
        self.device = device
        self.batch_size = batch_size
        self.dtype = resolve_torch_dtype(dtype_name, device)

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        raise NotImplementedError

    def unload(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SequenceClassificationReranker(BaseReranker):
    def __init__(
        self,
        spec: RerankerSpec,
        *,
        device: str,
        batch_size: int,
        dtype_name: str,
    ) -> None:
        super().__init__(spec, device=device, batch_size=batch_size, dtype_name=dtype_name)
        tokenizer_kwargs: dict[str, object] = {}
        if spec.trust_remote_code:
            tokenizer_kwargs["trust_remote_code"] = True
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id, **tokenizer_kwargs)

        model_kwargs: dict[str, object] = {}
        if spec.trust_remote_code:
            model_kwargs["trust_remote_code"] = True
        if self.dtype is not None:
            model_kwargs["torch_dtype"] = self.dtype
        self.model = AutoModelForSequenceClassification.from_pretrained(spec.model_id, **model_kwargs)
        self.model.eval()
        self.model.to(self.device)

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start : start + self.batch_size]
                queries = [query for query, _ in batch]
                documents = [document for _, document in batch]
                inputs = self.tokenizer(
                    queries,
                    text_pair=documents,
                    padding=True,
                    truncation=True,
                    max_length=self.spec.max_length,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                logits = self.model(**inputs, return_dict=True).logits.view(-1).float()
                scores.extend(logits.detach().cpu().tolist())
        return scores


class QwenCrossEncoderReranker(BaseReranker):
    def __init__(
        self,
        spec: RerankerSpec,
        *,
        device: str,
        batch_size: int,
        dtype_name: str,
    ) -> None:
        super().__init__(spec, device=device, batch_size=batch_size, dtype_name=dtype_name)
        from sentence_transformers import CrossEncoder

        model_kwargs: dict[str, object] = {}
        if self.dtype is not None:
            model_kwargs["torch_dtype"] = self.dtype

        prompts = {
            "scientific": spec.instruction or SCIENTIFIC_RERANK_INSTRUCTION,
        }
        self.model = CrossEncoder(
            spec.model_id,
            max_length=spec.max_length,
            trust_remote_code=True,
            prompts=prompts,
            default_prompt_name="scientific",
            automodel_args=model_kwargs,
        )
        self.model.model.to(self.device)

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return [float(score) for score in scores]

    def unload(self) -> None:
        del self.model
        super().unload()


def build_reranker(
    reranker_key: str,
    *,
    device: str,
    batch_size: int | None,
    dtype_name: str,
) -> BaseReranker:
    spec = RERANKER_SPECS[reranker_key]
    effective_batch_size = batch_size or spec.default_batch_size
    if spec.reranker_kind == "sequence_classification":
        return SequenceClassificationReranker(
            spec,
            device=device,
            batch_size=effective_batch_size,
            dtype_name=dtype_name,
        )
    if spec.reranker_kind == "qwen_cross_encoder":
        return QwenCrossEncoderReranker(
            spec,
            device=device,
            batch_size=effective_batch_size,
            dtype_name=dtype_name,
        )
    raise ValueError(f"Unsupported reranker_kind: {spec.reranker_kind}")
