# SciFact Benchmark

This directory contains a focused benchmark harness for `query -> abstract`
retrieval on the `mteb/scifact` dataset.

## What it does

- Loads SciFact queries, corpus, and qrels from Hugging Face
- Benchmarks a shortlist of bi-encoder retrieval models
- Saves per-model rankings and aggregate metrics
- Writes per-model status files and failure metadata
- Generates plots and a markdown report for comparison

## Default model shortlist

- `bge_en_icl`
- `qwen3_8b`
- `harrier_0_6b`
- `nv_embed_v2`
- `specter2`

## Quick start

```bash
source .venv/bin/activate
pip install -e .
python benchmarks/scifact/run_benchmark.py
```

## Common runs

Benchmark the default five models on SciFact test:

```bash
python benchmarks/scifact/run_benchmark.py
```

Colab notebook entry point:

`benchmarks/scifact/scifact_benchmark_colab.ipynb`

The notebook runs one model per subprocess, writes per-model logs, and
aggregates successful runs so Colab OOMs do not discard all results.

Benchmark a smaller subset first:

```bash
python benchmarks/scifact/run_benchmark.py --models specter2 harrier_0_6b --max-queries 50
```

Switch the document representation from abstract-only to title plus abstract:

```bash
python benchmarks/scifact/run_benchmark.py --document-mode title_abstract
```

Regenerate the report from an existing run:

```bash
python benchmarks/scifact/generate_report.py --run-dir benchmarks/scifact/results/<run_name>
```

## Outputs

Each run writes to `benchmarks/scifact/results/<timestamp>/`:

- `summary.csv`: one row per model
- `summary.json`: same metrics in JSON form
- `rankings/<model>.csv`: top-k ranked documents per query
- `plots/*.png`: metric comparison charts
- `REPORT.md`: generated benchmark summary

## Notes

- The default setup evaluates `query -> abstract`, because that is closest to
  your current retrieval stage.
- `SciFact` is still only a proxy benchmark. Use it to narrow the field before
  benchmarking on `open-arxiv`.
