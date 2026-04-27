from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


SCIENTIFIC_QUERY_INSTRUCTION = (
    "Given a scientific claim or research question, retrieve paper abstracts "
    "that provide relevant evidence."
)


BGE_EXAMPLES = [
    {
        "instruct": SCIENTIFIC_QUERY_INSTRUCTION,
        "query": "Does aspirin reduce the risk of cardiovascular events in high-risk patients?",
        "response": (
            "Evidence-focused abstracts should discuss aspirin use in high-risk "
            "patients and report clinical outcomes such as myocardial infarction, "
            "stroke, bleeding risk, or overall cardiovascular event reduction."
        ),
    },
    {
        "instruct": SCIENTIFIC_QUERY_INSTRUCTION,
        "query": "Can deep learning improve protein structure prediction?",
        "response": (
            "Relevant abstracts should describe protein structure prediction "
            "methods, especially deep learning architectures, and report "
            "improvements in accuracy, ranking, or structure quality."
        ),
    },
]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    family: str
    approx_params: str
    max_length: int
    encoder_kind: str
    default_batch_size: int
    notes: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "bge_en_icl": ModelSpec(
        key="bge_en_icl",
        model_id="BAAI/bge-en-icl",
        family="BGE",
        approx_params="7B",
        max_length=512,
        encoder_kind="bge_en_icl",
        default_batch_size=8,
        notes="English instruction-aware embedding model with optional few-shot examples.",
    ),
    "qwen3_8b": ModelSpec(
        key="qwen3_8b",
        model_id="Qwen/Qwen3-Embedding-8B",
        family="Qwen3",
        approx_params="8B",
        max_length=2048,
        encoder_kind="qwen_like",
        default_batch_size=8,
        notes="Instruction-aware decoder embedding model with last-token pooling.",
    ),
    "harrier_0_6b": ModelSpec(
        key="harrier_0_6b",
        model_id="microsoft/harrier-oss-v1-0.6b",
        family="Harrier OSS",
        approx_params="0.6B",
        max_length=2048,
        encoder_kind="qwen_like",
        default_batch_size=16,
        notes="Smaller multilingual harrier variant with last-token pooling.",
    ),
    "nv_embed_v2": ModelSpec(
        key="nv_embed_v2",
        model_id="nvidia/NV-Embed-v2",
        family="NV-Embed",
        approx_params="8B",
        max_length=2048,
        encoder_kind="nv_embed",
        default_batch_size=4,
        notes="Instruction-tuned LLM-based embedding model; non-commercial license.",
    ),
    "specter2": ModelSpec(
        key="specter2",
        model_id="allenai/specter2",
        family="SPECTER2",
        approx_params="BERT-base + adapters",
        max_length=512,
        encoder_kind="specter2",
        default_batch_size=32,
        notes="Scientific-domain retrieval model using adhoc query and proximity adapters.",
    ),
}


def available_model_keys() -> list[str]:
    return list(MODEL_SPECS.keys())


def default_model_keys() -> list[str]:
    return ["bge_en_icl", "qwen3_8b", "harrier_0_6b", "nv_embed_v2", "specter2"]


def select_device(explicit_device: str | None) -> str:
    if explicit_device:
        return explicit_device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


def _normalize(embeddings: torch.Tensor) -> np.ndarray:
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.detach().cpu().to(torch.float32).numpy()


def _iter_batches(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padded = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if left_padded:
        return last_hidden_state[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_state.shape[0]
    return last_hidden_state[
        torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths
    ]


class BaseEncoder:
    def __init__(self, spec: ModelSpec, *, device: str, batch_size: int, dtype_name: str) -> None:
        self.spec = spec
        self.device = device
        self.batch_size = batch_size
        self.dtype = resolve_torch_dtype(dtype_name, device)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def unload(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class QwenLikeEncoder(BaseEncoder):
    def __init__(self, spec: ModelSpec, *, device: str, batch_size: int, dtype_name: str) -> None:
        super().__init__(spec, device=device, batch_size=batch_size, dtype_name=dtype_name)
        tokenizer_kwargs = {}
        if "Qwen" in spec.model_id:
            tokenizer_kwargs["padding_side"] = "left"
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id, **tokenizer_kwargs)
        model_kwargs = {}
        if self.dtype is not None:
            model_kwargs["torch_dtype"] = self.dtype
        self.model = AutoModel.from_pretrained(spec.model_id, **model_kwargs)
        self.model.eval()
        self.model.to(self.device)

    def _format_query(self, text: str) -> str:
        return f"Instruct: {SCIENTIFIC_QUERY_INSTRUCTION}\nQuery: {text}"

    def _encode(self, texts: list[str]) -> np.ndarray:
        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for batch in _iter_batches(texts, self.batch_size):
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.spec.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                outputs = self.model(**encoded)
                pooled = _last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
                chunks.append(_normalize(pooled))
        return np.concatenate(chunks, axis=0)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self._format_query(text) for text in texts])

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)


class NVEmbedEncoder(BaseEncoder):
    def __init__(self, spec: ModelSpec, *, device: str, batch_size: int, dtype_name: str) -> None:
        super().__init__(spec, device=device, batch_size=batch_size, dtype_name=dtype_name)
        model_kwargs = {"trust_remote_code": True}
        if self.dtype is not None:
            model_kwargs["torch_dtype"] = self.dtype
        self.model = AutoModel.from_pretrained(spec.model_id, **model_kwargs)
        self.model.eval()
        self.model.to(self.device)

    def _encode(self, texts: list[str], instruction: str) -> np.ndarray:
        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for batch in _iter_batches(texts, self.batch_size):
                embeddings = self.model.encode(
                    batch,
                    instruction=instruction,
                    max_length=self.spec.max_length,
                )
                if not isinstance(embeddings, torch.Tensor):
                    embeddings = torch.tensor(embeddings)
                chunks.append(_normalize(embeddings))
        return np.concatenate(chunks, axis=0)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        query_instruction = f"Instruct: {SCIENTIFIC_QUERY_INSTRUCTION}\nQuery: "
        return self._encode(texts, instruction=query_instruction)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, instruction="")


class Specter2Encoder(BaseEncoder):
    def __init__(self, spec: ModelSpec, *, device: str, batch_size: int, dtype_name: str) -> None:
        super().__init__(spec, device=device, batch_size=batch_size, dtype_name=dtype_name)
        from adapters import AutoAdapterModel

        base_id = "allenai/specter2_base"
        self.tokenizer = AutoTokenizer.from_pretrained(base_id)
        self.query_model = AutoAdapterModel.from_pretrained(base_id)
        query_adapter = self.query_model.load_adapter(
            "allenai/specter2_adhoc_query",
            source="hf",
        )
        self.query_model.set_active_adapters(query_adapter)
        self.query_model.eval()
        self.query_model.to(self.device)

        self.document_model = AutoAdapterModel.from_pretrained(base_id)
        document_adapter = self.document_model.load_adapter(
            "allenai/specter2",
            source="hf",
        )
        self.document_model.set_active_adapters(document_adapter)
        self.document_model.eval()
        self.document_model.to(self.device)

    def _encode(self, texts: list[str], *, model: torch.nn.Module) -> np.ndarray:
        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for batch in _iter_batches(texts, self.batch_size):
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.spec.max_length,
                    return_tensors="pt",
                    return_token_type_ids=False,
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                outputs = model(**encoded)
                pooled = outputs.last_hidden_state[:, 0, :]
                chunks.append(_normalize(pooled))
        return np.concatenate(chunks, axis=0)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, model=self.query_model)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, model=self.document_model)

    def unload(self) -> None:
        del self.query_model
        del self.document_model
        super().unload()


class BGEEnICLEncoder(BaseEncoder):
    def __init__(
        self,
        spec: ModelSpec,
        *,
        device: str,
        batch_size: int,
        dtype_name: str,
        use_examples: bool,
    ) -> None:
        super().__init__(spec, device=device, batch_size=batch_size, dtype_name=dtype_name)
        model_kwargs = {}
        if self.dtype is not None:
            model_kwargs["torch_dtype"] = self.dtype
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        self.model = AutoModel.from_pretrained(spec.model_id, **model_kwargs)
        self.model.eval()
        self.model.to(self.device)
        self.use_examples = use_examples
        self.examples_prefix = self._build_examples_prefix() if use_examples else ""

    def _get_detailed_instruct(self, query: str) -> str:
        return f"<instruct>{SCIENTIFIC_QUERY_INSTRUCTION}\n<query>{query}"

    def _get_detailed_example(self, example: dict[str, str]) -> str:
        return (
            f"<instruct>{example['instruct']}\n<query>{example['query']}\n"
            f"<response>{example['response']}"
        )

    def _build_examples_prefix(self) -> str:
        return "\n\n".join(self._get_detailed_example(example) for example in BGE_EXAMPLES) + "\n\n"

    def _prepare_queries(self, queries: list[str]) -> tuple[int, list[str]]:
        queries = [self._get_detailed_instruct(query) for query in queries]
        if not self.examples_prefix:
            return self.spec.max_length, [query + "\n<response>" for query in queries]

        stripped_queries = []
        max_query_len = self.spec.max_length
        prompt_budget = (
            len(self.tokenizer("<s>", add_special_tokens=False)["input_ids"])
            + len(self.tokenizer("\n<response></s>", add_special_tokens=False)["input_ids"])
        )
        query_encoding = self.tokenizer(
            queries,
            max_length=max_query_len - prompt_budget,
            truncation=True,
            return_token_type_ids=False,
            return_tensors=None,
            add_special_tokens=False,
        )
        stripped_queries = self.tokenizer.batch_decode(query_encoding["input_ids"])

        prefix_ids = self.tokenizer(self.examples_prefix, add_special_tokens=False)["input_ids"]
        suffix_ids = self.tokenizer("\n<response>", add_special_tokens=False)["input_ids"]
        new_max_length = (len(prefix_ids) + len(suffix_ids) + max_query_len + 8) // 8 * 8 + 8

        return (
            new_max_length,
            [self.examples_prefix + query + "\n<response>" for query in stripped_queries],
        )

    def _encode(self, texts: list[str], *, max_length: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for batch in _iter_batches(texts, self.batch_size):
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                outputs = self.model(**encoded)
                pooled = _last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
                chunks.append(_normalize(pooled))
        return np.concatenate(chunks, axis=0)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        max_length, prepared_queries = self._prepare_queries(texts)
        return self._encode(prepared_queries, max_length=max_length)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, max_length=self.spec.max_length)


def build_encoder(
    model_key: str,
    *,
    device: str,
    batch_size: int | None,
    dtype_name: str,
    bge_use_examples: bool,
) -> BaseEncoder:
    spec = MODEL_SPECS[model_key]
    effective_batch_size = batch_size or spec.default_batch_size

    if spec.encoder_kind == "qwen_like":
        return QwenLikeEncoder(
            spec,
            device=device,
            batch_size=effective_batch_size,
            dtype_name=dtype_name,
        )
    if spec.encoder_kind == "nv_embed":
        return NVEmbedEncoder(
            spec,
            device=device,
            batch_size=effective_batch_size,
            dtype_name=dtype_name,
        )
    if spec.encoder_kind == "specter2":
        return Specter2Encoder(
            spec,
            device=device,
            batch_size=effective_batch_size,
            dtype_name=dtype_name,
        )
    if spec.encoder_kind == "bge_en_icl":
        return BGEEnICLEncoder(
            spec,
            device=device,
            batch_size=effective_batch_size,
            dtype_name=dtype_name,
            use_examples=bge_use_examples,
        )
    raise ValueError(f"Unsupported encoder_kind: {spec.encoder_kind}")
