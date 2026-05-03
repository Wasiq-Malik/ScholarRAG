# ScholarRAG

Scientific retrieval augmented generation over `open-index/open-arxiv`.

The repo now has two tracks:

- `benchmarks/scifact/`: retrieval and reranker benchmarking.
- `scholarrag/`: production RAG app code for ingestion, retrieval, and answer generation.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
scholarrag inspect-open-arxiv
```

## Production architecture

```text
OpenArXiv dataset
  -> normalize title/abstract metadata
  -> chunk abstracts
  -> google/embeddinggemma-300m document embeddings
  -> Qdrant collection, cosine distance, 768 dimensions

User query
  -> EmbeddingGemma fact-check query prompt
  -> Qdrant dense retrieval
  -> Qwen/Qwen3.5-9B through vLLM
  -> grounded answer with source citations
```

The production retriever is fixed to `google/embeddinggemma-300m` with
fact-check query formatting:

```text
task: fact checking | query: <question>
```

Documents are embedded as:

```text
title: <title-or-none> | text: <chunk text>
```

## Local services

Start Qdrant separately:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Start vLLM separately. The default final-answer model is `Qwen/Qwen3.5-9B`,
chosen for the Colab/L4 or small-GPU target because it is stronger than the
older 8B choice while still being realistic with a constrained context length.

```bash
vllm serve Qwen/Qwen3.5-9B \
  --port 8000 \
  --max-model-len 32768 \
  --dtype auto \
  --gpu-memory-utilization 0.90 \
  --language-model-only
```

Fallback choices:

- `Qwen/Qwen3-8B`: use if Qwen3.5 compatibility or memory is a problem.
- `Qwen/Qwen3-14B`: use only if you can accept lower throughput or have more GPU headroom.

## Production commands

Inspect streamed OpenArXiv rows:

```bash
scholarrag inspect-open-arxiv --rows 3
```

Index a small OpenArXiv sample into Qdrant:

```bash
scholarrag index-open-arxiv --limit 1000
```

Filter indexing to one or more arXiv categories:

```bash
scholarrag index-open-arxiv --limit 1000 --category cs.CL --category cs.IR
```

Run the FastAPI app:

```bash
scholarrag serve-api --host 127.0.0.1 --port 8080
```

Query the API:

```bash
curl -sS http://127.0.0.1:8080/query \
  -H 'content-type: application/json' \
  -d '{"question":"What evidence exists for retrieval augmented generation in scientific QA?","top_k":10}'
```

## Settings

Settings are read from environment variables with the `SCHOLARRAG_` prefix.
Useful defaults:

```text
SCHOLARRAG_QDRANT_URL=http://localhost:6333
SCHOLARRAG_QDRANT_COLLECTION=open_arxiv_embeddinggemma_fact_check_768_cosine
SCHOLARRAG_VLLM_BASE_URL=http://localhost:8000/v1
SCHOLARRAG_VLLM_MODEL=Qwen/Qwen3.5-9B
SCHOLARRAG_SQLITE_PATH=data/scholarrag.sqlite3
HF_TOKEN=<token for gated Hugging Face models, if needed>
```

## API

```text
GET /health
POST /ingest/open-arxiv
POST /query
```

`POST /query` request:

```json
{
  "question": "Does the evidence support this scientific claim?",
  "top_k": 10,
  "filters": {}
}
```

`POST /query` response:

```json
{
  "answer": "...",
  "sources": [
    {
      "point_id": "...",
      "score": 0.82,
      "paper_id": "...",
      "chunk_id": 0,
      "title": "...",
      "categories": ["cs.CL"],
      "update_date": "2026-01-01",
      "text_preview": "..."
    }
  ],
  "retrieval": {
    "candidate_k": 50,
    "returned_k": 10,
    "collection": "open_arxiv_embeddinggemma_fact_check_768_cosine",
    "reranker": null
  },
  "models": {
    "embedding": "google/embeddinggemma-300m",
    "llm": "Qwen/Qwen3.5-9B"
  }
}
```

## Dataset inspection

The legacy script still works:

```bash
python scripts/inspect_open_arxiv.py
```

It:

- Prints dataset metadata, including row count and reported size
- Streams the first few records so you can inspect titles and abstracts
- Optionally downloads the dataset locally into `data/hf_cache`

## Optional local download

```bash
python scripts/inspect_open_arxiv.py --download-local
```

## SciFact benchmark

A dedicated benchmark harness lives under
[benchmarks/scifact/README.md](benchmarks/scifact/README.md) for comparing
bi-encoder retrieval models on `mteb/scifact`.
