# ScholarRAG

Scientific retrieval augmented generation over `open-index/open-arxiv`.

The repo now has three tracks:

- `benchmarks/scifact/`: retrieval and reranker benchmarking.
- `experiments/`: Colab notebooks for larger OpenArXiv indexing and follow-on model work.
- `scholarrag/`: app code for ingestion, retrieval, vector-store access, and answer generation.

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
  -> embed title + full abstract with google/embeddinggemma-300m
  -> Qdrant or FAISS vector index, cosine/inner-product over normalized 768-d vectors

User query
  -> EmbeddingGemma fact-check query prompt
  -> dense retrieval from Qdrant or FAISS
  -> hosted Gemma through the Gemini API
  -> grounded answer with source citations
  -> ranked source list with arXiv links
```

The production retriever is fixed to `google/embeddinggemma-300m` with
fact-check query formatting:

```text
task: fact checking | query: <question>
```

Documents are embedded as:

```text
title: <title-or-none> | text: <full abstract>
```

## Module map

- `scholarrag/config.py`: runtime settings and environment variable aliases.
- `scholarrag/datasets/open_arxiv.py`: OpenArXiv streaming and record normalization.
- `scholarrag/chunking.py`: legacy Qdrant abstract/title chunk creation and stable chunk IDs.
- `scholarrag/embeddings.py`: EmbeddingGemma query/document formatting and encoding.
- `scholarrag/vectorstores/qdrant_store.py`: Qdrant collection creation, upsert, and search.
- `scholarrag/vectorstores/faiss_store.py`: FAISS index loading plus SQLite metadata lookup.
- `scholarrag/ingest.py`: Qdrant-backed OpenArXiv ingestion.
- `scholarrag/retrieval.py`: query embedding, vector search, source formatting, arXiv URLs.
- `scholarrag/llm.py`: Gemini/Gemma answer generation over retrieved context.
- `scholarrag/api/app.py`: FastAPI app with health, ingestion, and query endpoints.

## Local services

Qdrant is the service-backed vector DB path. Start it separately:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Qdrant is the better long-term app backend when you want service semantics,
incremental upserts, payload filtering, and snapshots. The local API ingestion
endpoint currently writes to Qdrant only.

FAISS is the current Colab artifact path. It is the fastest way to build and
reload a 100k-paper experiment from Google Drive without operating a database
service inside every Colab session.

For a FAISS-backed local test using Colab-generated artifacts, copy these files
from Google Drive to your machine:

```text
open_arxiv_papers.faiss
open_arxiv_papers.sqlite
manifest.json
```

Then start the API with FAISS settings:

```bash
export SCHOLARRAG_VECTOR_BACKEND=faiss
export SCHOLARRAG_FAISS_INDEX_PATH=/absolute/path/to/open_arxiv_papers.faiss
export SCHOLARRAG_FAISS_SQLITE_PATH=/absolute/path/to/open_arxiv_papers.sqlite
export GEMINI_API_KEY=<your-gemini-api-key>
export SCHOLARRAG_GEMINI_MODEL=gemma-4-31b-it
scholarrag serve-api --host 127.0.0.1 --port 8080
```

If no reachable LLM endpoint is configured, `/query` still returns retrieved
sources and an answer-generation-unavailable message.

Answer generation uses the hosted Gemini API. Set `GEMINI_API_KEY` or
`SCHOLARRAG_GEMINI_API_KEY` in `.env`; the default answer model is
`gemma-4-31b-it`.

## OpenArXiv FAISS experiment

The current large-corpus notebook is:

`experiments/open_arxiv_faiss_colab.ipynb`

It defaults to `RUN_MODE = "cs_100k"` and indexes up to 100,000 recent
OpenArXiv papers with at least one category beginning with `cs.`. The default
slice uses `MIN_UPDATE_DATE = "2020-01-01"` and sorts by `update_date`
descending before selecting the 100k papers. Each paper gets one vector from
`title + full abstract`; abstracts are not chunked in this
experiment. `dry_run` uses exact `IndexIDMap2(IndexFlatIP)`. `cs_100k` defaults
to approximate `IndexIDMap2(IndexIVFFlat)` so you can test faster retrieval;
set `FAISS_INDEX_KIND = "flat"` for an exact baseline. For `full`, it switches
to compressed approximate `IndexIDMap2(IndexIVFPQ)`. SQLite stores abstract text
and metadata.

The notebook uses L4-oriented embedding batch defaults: 512 for `dry_run` and
`cs_100k`, 384 for `full`. OpenArXiv does not provide a ready category index,
so the first CS-only run still scans the dataset once; the filtered subset is
then saved to Drive and reused on later Colab reconnects.

The notebook writes durable artifacts to Google Drive under:

`/content/drive/MyDrive/scholarrag/open_arxiv_embeddinggemma_fact_check_cs_100k_from_2020_recent_ivfflat/`

Use `RUN_MODE = "dry_run"` for a 1,000-paper validation run before spending L4
time on the 100k experiment.

## Colab API launcher

After the FAISS notebook has written `open_arxiv_papers.faiss` and
`open_arxiv_papers.sqlite` to Drive, you can serve the API from the Colab VM
without reindexing:

```bash
cd /content/ScholarRAG
pip install -e .
python scripts/colab_launch_api.py --print-env
```

The launcher defaults to the current 100k recent-CS artifact folder:

```text
/content/drive/MyDrive/scholarrag/open_arxiv_embeddinggemma_fact_check_cs_100k_from_2020_recent_ivfflat/
```

For a public URL, set an ngrok token in the Colab terminal first:

```bash
export NGROK_AUTHTOKEN=<your-ngrok-token>
python scripts/colab_launch_api.py --install-ngrok
```

To reuse a fixed ngrok domain assigned to your account:

```bash
export NGROK_DOMAIN=complete-jay-strictly.ngrok-free.app
python scripts/colab_launch_api.py --install-ngrok
```

It prints `/health` and public tunnel URLs when available. If you use a
different FAISS run folder, pass `--run-dir /content/drive/MyDrive/...`.

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
SCHOLARRAG_VECTOR_BACKEND=qdrant
SCHOLARRAG_FAISS_INDEX_PATH=data/faiss/open_arxiv_papers.faiss
SCHOLARRAG_FAISS_SQLITE_PATH=data/faiss/open_arxiv_papers.sqlite
SCHOLARRAG_GEMINI_MODEL=gemma-4-31b-it
SCHOLARRAG_LLM_MODEL=<Gemini model, optional alias>
GEMINI_API_KEY=<Gemini API key for hosted answer generation>
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
      "arxiv_url": "https://arxiv.org/abs/...",
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
    "llm": "gemma-4-31b-it"
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
