# Experiments

This directory holds Colab-oriented experiments that are larger or more
exploratory than the local app path.

## OpenArXiv FAISS

`open_arxiv_faiss_colab.ipynb` builds a persistent OpenArXiv vector index using:

- `google/embeddinggemma-300m` with the fact-check query prompt
- one vector per paper using title + full abstract
- FAISS `IndexIDMap2(IndexFlatIP)` for exact search in `dry_run`
- FAISS `IndexIDMap2(IndexIVFFlat)` for fast approximate search in `cs_100k`
- FAISS `IndexIDMap2(IndexIVFPQ)` for compressed approximate search in `full`
- SQLite for abstract text, categories, authors, update dates, and arXiv IDs
- Google Drive for durable FAISS/SQLite artifacts and checkpoints
- Gemini API / hosted Gemma for optional RAG answer generation

The default run mode is `cs_100k`, which indexes up to 100,000 recent papers
with at least one arXiv category beginning with `cs.`. It uses
`MIN_UPDATE_DATE = "2020-01-01"` and sorts by `update_date` descending before
selecting the 100k papers. Use `dry_run` for a 1,000-paper smoke test first.

The L4-oriented default embedding batch size is 512 for `dry_run` and
`cs_100k`, and 384 for `full`. Lower it to 256 or 128 if Colab runs out of GPU
memory.

OpenArXiv is stored as a corpus table, not a searchable category index, so the
first CS-only run still has to scan the dataset once. The notebook saves the
filtered subset to Drive under the run folder, so later Colab reconnects can
reload the CS subset directly before resuming the FAISS/SQLite artifacts.

To compare exact and approximate search on the same 100k corpus, set
`FAISS_INDEX_KIND = "flat"` for an exact baseline or keep the default
`"ivfflat"` for faster approximate search. The Drive run folder includes the
index kind, so both artifacts can coexist. The `cs_100k` IVFFlat default trains
on 20,000 papers with `nlist = 512` and `nprobe = 32`; raise `nprobe` for
higher recall or lower it for faster search.

The notebook is standalone: it does not require cloning or installing this repo
inside Colab for indexing/retrieval. It intentionally duplicates a small amount
of package logic so long indexing runs are reproducible from the notebook alone.
Clone/install the repo in Colab only if you want to run the FastAPI app there
after the FAISS artifacts are built.

To launch that app from a Colab terminal against the saved Drive artifacts:

```bash
cd /content/ScholarRAG
pip install -e .
python scripts/colab_launch_api.py --print-env
```

For an internet-accessible tunnel, set `NGROK_AUTHTOKEN` and run:

```bash
python scripts/colab_launch_api.py --install-ngrok
```

For a fixed ngrok domain assigned to your account:

```bash
export NGROK_DOMAIN=complete-jay-strictly.ngrok-free.app
python scripts/colab_launch_api.py --install-ngrok
```

## SciFact

`scifact/` contains the reranker fine-tuning notebook and related follow-on
work. The stable benchmark harness remains under `benchmarks/scifact/`.
