# SciFact Experiments

This directory holds iterative R&D work for SciFact beyond static benchmarking.

Current notebooks:

- `scifact_gte_reranker_finetune_colab.ipynb`
  Fine-tunes `Alibaba-NLP/gte-reranker-modernbert-base` on SciFact claim-to-abstract relevance.

Notes:

- The existing retrieval and reranker harness under `benchmarks/scifact/` is still the source of truth for evaluation runs.
- This `experiments/` workspace is for follow-on work such as score fusion analysis and model fine-tuning without breaking the benchmark notebooks you already ran.
