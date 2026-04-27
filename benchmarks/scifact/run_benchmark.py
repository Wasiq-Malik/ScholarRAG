from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

from data import load_scifact
from generate_report import build_report
from metrics import score_and_write_rankings
from models import (
    MODEL_SPECS,
    available_model_keys,
    build_encoder,
    default_model_keys,
    select_device,
)


def write_summary_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark bi-encoder retrieval models on SciFact.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=default_model_keys(),
        choices=available_model_keys(),
        help="Model keys to benchmark.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "test"],
        help="SciFact qrels split to evaluate.",
    )
    parser.add_argument(
        "--document-mode",
        default="abstract",
        choices=["abstract", "title_abstract"],
        help="Which corpus text representation to use.",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Execution device. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype to request during model loading.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the per-model default batch size.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional query cap for smoke tests.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="How many ranked documents to write per query.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/hf_cache"),
        help="Hugging Face cache directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Benchmark output directory. Defaults to a timestamped directory under benchmarks/scifact/results.",
    )
    parser.add_argument(
        "--bge-use-examples",
        action="store_true",
        help="Enable few-shot examples for bge-en-icl query formatting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)

    run_dir = args.output_dir
    if run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("benchmarks/scifact/results") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    data = load_scifact(
        split=args.split,
        document_mode=args.document_mode,
        cache_dir=args.cache_dir,
        max_queries=args.max_queries,
    )

    config = {
        "split": args.split,
        "document_mode": args.document_mode,
        "device": device,
        "dtype": args.dtype,
        "models": args.models,
        "query_count": len(data.query_ids),
        "corpus_size": len(data.corpus_ids),
        "top_k": args.top_k,
        "bge_use_examples": args.bge_use_examples,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    summary_rows: list[dict[str, object]] = []
    for model_key in args.models:
        spec = MODEL_SPECS[model_key]
        print(f"Benchmarking {model_key} ({spec.model_id})")

        encoder = build_encoder(
            model_key,
            device=device,
            batch_size=args.batch_size,
            dtype_name=args.dtype,
            bge_use_examples=args.bge_use_examples,
        )
        try:
            start_time = time.perf_counter()
            doc_start = time.perf_counter()
            corpus_embeddings = encoder.encode_documents(data.corpus_texts)
            doc_seconds = time.perf_counter() - doc_start

            query_start = time.perf_counter()
            query_embeddings = encoder.encode_queries(data.query_texts)
            query_seconds = time.perf_counter() - query_start

            rankings_path = run_dir / "rankings" / f"{model_key}.csv"
            metrics = score_and_write_rankings(
                model_name=model_key,
                query_ids=data.query_ids,
                corpus_ids=data.corpus_ids,
                query_embeddings=query_embeddings,
                corpus_embeddings=corpus_embeddings,
                positives_by_query=data.positives_by_query,
                rankings_path=rankings_path,
                top_k_to_write=args.top_k,
            )
            total_seconds = time.perf_counter() - start_time

            summary_rows.append(
                {
                    "model": model_key,
                    "model_id": spec.model_id,
                    "family": spec.family,
                    "approx_params": spec.approx_params,
                    "embedding_dim": corpus_embeddings.shape[1],
                    "query_count": len(data.query_ids),
                    "corpus_size": len(data.corpus_ids),
                    "document_mode": args.document_mode,
                    "ndcg_at_10": metrics.ndcg_at_10,
                    "mrr_at_10": metrics.mrr_at_10,
                    "recall_at_10": metrics.recall_at_10,
                    "recall_at_50": metrics.recall_at_50,
                    "recall_at_100": metrics.recall_at_100,
                    "map_at_10": metrics.map_at_10,
                    "document_encoding_seconds": doc_seconds,
                    "query_encoding_seconds": query_seconds,
                    "total_runtime_seconds": total_seconds,
                    "documents_per_second": len(data.corpus_ids) / max(doc_seconds, 1e-6),
                    "queries_per_second": len(data.query_ids) / max(query_seconds, 1e-6),
                    "notes": spec.notes,
                }
            )
        finally:
            encoder.unload()

    if not summary_rows:
        raise ValueError("No benchmark results were produced.")

    write_summary_csv(summary_rows, run_dir / "summary.csv")
    (run_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2))
    report_path = build_report(run_dir)

    print(f"\nFinished benchmark run.")
    print(f"Summary: {run_dir / 'summary.csv'}")
    print(f"Report:  {report_path}")


if __name__ == "__main__":
    main()
