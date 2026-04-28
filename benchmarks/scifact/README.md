# SciFact Benchmark

This directory contains a focused benchmark harness for `query -> abstract`
retrieval on the `mteb/scifact` dataset.

## What it does

- Loads SciFact queries, corpus, and qrels from Hugging Face
- Benchmarks a shortlist of bi-encoder retrieval models
- Saves per-model rankings and aggregate metrics
- Writes per-model status files and failure metadata
- Generates plots and a markdown report for comparison
- Reports both offline indexing cost and query-time latency metrics

## Default model shortlist

- `bge_en_icl_plain`
- `bge_en_icl_examples`
- `qwen3_0_6b`
- `qwen3_4b`
- `qwen3_8b`
- `harrier_0_6b`
- `embeddinggemma_300m`
- `embeddinggemma_fact_check`
- `embeddinggemma_qa`
- `specter2`

## Quick start

```bash
source .venv/bin/activate
pip install -e .
python benchmarks/scifact/run_benchmark.py
```

If you want to include gated Hugging Face models such as `embeddinggemma_300m`,
export `HF_TOKEN` in your shell or Colab session before running the benchmark.

## Common runs

Benchmark the default ten-model mix on SciFact test:

```bash
python benchmarks/scifact/run_benchmark.py
```

Colab notebook entry point:

`benchmarks/scifact/scifact_benchmark_colab.ipynb`

The notebook runs one model per subprocess, writes per-model logs, and
aggregates successful runs so Colab OOMs do not discard all results.
It now defaults to an L4-oriented inference preset, with one lower-memory
fallback preset you can switch to in one config cell.

`bge_en_icl` is benchmarked twice by default:

- `bge_en_icl_plain`: instruction/query formatting only
- `bge_en_icl_examples`: instruction/query formatting plus few-shot examples

`EmbeddingGemma` is benchmarked three ways by default:

- `embeddinggemma_300m`: default query/document prompts
- `embeddinggemma_fact_check`: official fact-checking query prompt
- `embeddinggemma_qa`: official question-answering query prompt

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

Key latency fields in `summary.csv`:

- `document_encoding_seconds`: offline corpus embedding time
- `query_encoding_seconds`: total query embedding time over the evaluation set
- `retrieval_scoring_seconds`: total dense retrieval scoring time over the evaluation set
- `avg_document_latency_ms`: average per-document embedding latency during corpus encoding
- `avg_query_encoding_latency_ms`: average per-query embedding latency
- `avg_retrieval_scoring_latency_ms`: average per-query dense scoring latency
- `avg_query_end_to_end_latency_ms`: average per-query latency for query embedding plus dense retrieval scoring
- `total_runtime_seconds`: end-to-end benchmark runtime for the model

The generated report also includes a direct `nDCG@10` vs query-latency tradeoff plot so you can see which models sit on the useful quality/latency frontier.
It also writes a `MAP@10` comparison plot alongside the `nDCG@10`, `MRR@10`, and latency plots.

## Notes

- The default setup evaluates `query -> abstract`, because that is closest to
  your current retrieval stage.
- `SciFact` is still only a proxy benchmark. Use it to narrow the field before
  benchmarking on `open-arxiv`.
