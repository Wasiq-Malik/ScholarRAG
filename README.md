# ScholarRAG

Initial dataset inspection setup for `open-index/open-arxiv`.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/inspect_open_arxiv.py
```

## What the script does

- Prints dataset metadata, including row count and reported size
- Streams the first few records so you can inspect titles and abstracts
- Optionally downloads the dataset locally into `data/hf_cache`

## Optional local download

```bash
python scripts/inspect_open_arxiv.py --download-local
```

## SciFact benchmark

A dedicated benchmark harness now lives under [benchmarks/scifact/README.md](/Users/wasiqmalik/Public/ScholarRAG/benchmarks/scifact/README.md) for comparing bi-encoder retrieval models on `mteb/scifact`.
