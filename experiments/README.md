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

The default run mode is `cs_100k`, which indexes up to 100,000 papers with at
least one arXiv category beginning with `cs.`. Use `dry_run` for a 1,000-paper
smoke test first.

The L4-oriented default embedding batch size is 256 for `dry_run` and
`cs_100k`, and 192 for `full`. Lower it to 128 if Colab runs out of GPU memory.

OpenArXiv is stored as a corpus table, not a searchable category index, so the
first CS-only run still has to scan the dataset once. The notebook saves the
filtered subset to Drive under the run folder, so later Colab reconnects can
reload the CS subset directly before resuming the FAISS/SQLite artifacts.

To compare exact and approximate search on the same 100k corpus, set
`FAISS_INDEX_KIND = "flat"` for an exact baseline or keep the default
`"ivfflat"` for faster approximate search. The Drive run folder includes the
index kind, so both artifacts can coexist.

The notebook is standalone: it does not require cloning or installing this repo
inside Colab for indexing/retrieval. It intentionally duplicates a small amount
of package logic so long indexing runs are reproducible from the notebook alone.
Clone/install the repo in Colab only if you want to run the FastAPI app there
after the FAISS artifacts are built.

## SciFact

`scifact/` contains the reranker fine-tuning notebook and related follow-on
work. The stable benchmark harness remains under `benchmarks/scifact/`.
