from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from data import load_scifact
from generate_reranker_report import build_report
from metrics import EvalSummary, score_ranked_lists, write_rankings_csv
from models import available_model_keys, build_encoder, select_device
from reranker_models import (
    RERANKER_SPECS,
    available_reranker_keys,
    build_reranker,
    default_reranker_keys,
)


def write_summary_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def recall_for_candidates(candidate_ids: list[str], relevant_docs: dict[str, float]) -> float:
    if not relevant_docs:
        return 0.0
    hits = sum(1 for corpus_id in candidate_ids if corpus_id in relevant_docs)
    return hits / len(relevant_docs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark rerankers on top of a fixed SciFact bi-encoder retriever.")
    parser.add_argument(
        "--retriever-model",
        default="embeddinggemma_fact_check",
        choices=available_model_keys(),
        help="Bi-encoder retriever used to generate top-k candidates.",
    )
    parser.add_argument(
        "--rerankers",
        nargs="+",
        default=default_reranker_keys(),
        choices=available_reranker_keys(),
        help="Rerankers to evaluate.",
    )
    parser.add_argument(
        "--candidate-ks",
        nargs="+",
        type=int,
        default=[50, 100],
        help="Candidate set sizes retrieved by the bi-encoder before reranking.",
    )
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--document-mode", default="abstract", choices=["abstract", "title_abstract"])
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--retriever-batch-size", type=int, default=None)
    parser.add_argument("--reranker-batch-size", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/hf_cache"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show retriever progress bars.",
    )
    return parser.parse_args()


def summary_row(
    *,
    stage: str,
    candidate_k: int,
    retriever_model: str,
    reranker_model: str,
    reranker_model_id: str,
    reranker_family: str,
    reranker_batch_size: int,
    stage1_recall_at_k: float,
    metrics: EvalSummary,
    retriever_query_stage_seconds: float,
    avg_reranker_latency_ms: float,
    reranker_seconds: float,
    notes: str,
) -> dict[str, object]:
    avg_retriever_latency_ms = retriever_query_stage_seconds * 1000
    return {
        "stage": stage,
        "retriever_model": retriever_model,
        "reranker_model": reranker_model,
        "reranker_model_id": reranker_model_id,
        "reranker_family": reranker_family,
        "candidate_k": candidate_k,
        "stage1_recall_at_k": stage1_recall_at_k,
        "ndcg_at_10": metrics.ndcg_at_10,
        "mrr_at_10": metrics.mrr_at_10,
        "recall_at_10": metrics.recall_at_10,
        "recall_at_50": metrics.recall_at_50,
        "recall_at_100": metrics.recall_at_100,
        "map_at_10": metrics.map_at_10,
        "retriever_query_stage_seconds": retriever_query_stage_seconds,
        "avg_retriever_query_latency_ms": avg_retriever_latency_ms,
        "reranker_seconds": reranker_seconds,
        "avg_reranker_latency_ms": avg_reranker_latency_ms,
        "avg_total_query_latency_ms": avg_retriever_latency_ms + avg_reranker_latency_ms,
        "reranker_batch_size": reranker_batch_size,
        "notes": notes,
    }


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    candidate_ks = sorted(set(args.candidate_ks))
    max_candidate_k = max(candidate_ks)

    run_dir = args.output_dir
    if run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("benchmarks/scifact/results") / f"reranker_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    data = load_scifact(
        split=args.split,
        document_mode=args.document_mode,
        cache_dir=args.cache_dir,
        max_queries=args.max_queries,
    )

    config = {
        "retriever_model": args.retriever_model,
        "rerankers": args.rerankers,
        "candidate_ks": candidate_ks,
        "split": args.split,
        "document_mode": args.document_mode,
        "device": device,
        "dtype": args.dtype,
        "query_count": len(data.query_ids),
        "corpus_size": len(data.corpus_ids),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    retriever = build_encoder(
        args.retriever_model,
        device=device,
        batch_size=args.retriever_batch_size,
        dtype_name=args.dtype,
        bge_use_examples=False,
        show_progress=args.show_progress,
    )
    try:
        doc_start = time.perf_counter()
        corpus_embeddings = retriever.encode_documents(data.corpus_texts)
        doc_seconds = time.perf_counter() - doc_start

        query_start = time.perf_counter()
        query_embeddings = retriever.encode_queries(data.query_texts)
        query_seconds = time.perf_counter() - query_start

        scoring_start = time.perf_counter()
        similarity = query_embeddings @ corpus_embeddings.T
        full_ranked_indices = np.argsort(-similarity, axis=1)
        scoring_seconds = time.perf_counter() - scoring_start
    finally:
        retriever.unload()

    retriever_query_stage_seconds = (query_seconds + scoring_seconds) / len(data.query_ids)

    full_ranked_ids_by_query: dict[str, list[str]] = {}
    full_ranked_scores_by_query: dict[str, list[float]] = {}
    candidate_ids_by_query: dict[int, dict[str, list[str]]] = {k: {} for k in candidate_ks}
    candidate_scores_by_query: dict[int, dict[str, list[float]]] = {k: {} for k in candidate_ks}
    candidate_recall_by_k: dict[int, list[float]] = {k: [] for k in candidate_ks}
    corpus_text_by_id = dict(zip(data.corpus_ids, data.corpus_texts, strict=True))

    for query_index, query_id in enumerate(data.query_ids):
        ranked_indices = full_ranked_indices[query_index]
        ranked_ids = [data.corpus_ids[index] for index in ranked_indices]
        ranked_scores = [float(similarity[query_index, index]) for index in ranked_indices]
        full_ranked_ids_by_query[query_id] = ranked_ids
        full_ranked_scores_by_query[query_id] = ranked_scores

        relevant_docs = data.positives_by_query[query_id]
        for candidate_k in candidate_ks:
            top_ids = ranked_ids[:candidate_k]
            top_scores = ranked_scores[:candidate_k]
            candidate_ids_by_query[candidate_k][query_id] = top_ids
            candidate_scores_by_query[candidate_k][query_id] = top_scores
            candidate_recall_by_k[candidate_k].append(recall_for_candidates(top_ids, relevant_docs))

    summary_rows: list[dict[str, object]] = []

    baseline_metrics = score_ranked_lists(
        query_ids=data.query_ids,
        ranked_corpus_ids_by_query=full_ranked_ids_by_query,
        positives_by_query=data.positives_by_query,
    )
    write_rankings_csv(
        model_name=args.retriever_model,
        query_ids=data.query_ids,
        ranked_corpus_ids_by_query={query_id: ids[:max_candidate_k] for query_id, ids in full_ranked_ids_by_query.items()},
        ranked_scores_by_query={query_id: scores[:max_candidate_k] for query_id, scores in full_ranked_scores_by_query.items()},
        positives_by_query=data.positives_by_query,
        rankings_path=run_dir / "rankings" / f"{args.retriever_model}.csv",
    )
    summary_rows.append(
        summary_row(
            stage="retriever_only",
            candidate_k=max_candidate_k,
            retriever_model=args.retriever_model,
            reranker_model="none",
            reranker_model_id="none",
            reranker_family="none",
            reranker_batch_size=0,
            stage1_recall_at_k=float(np.mean(candidate_recall_by_k[max_candidate_k])),
            metrics=baseline_metrics,
            retriever_query_stage_seconds=retriever_query_stage_seconds,
            avg_reranker_latency_ms=0.0,
            reranker_seconds=0.0,
            notes="Retriever-only baseline.",
        )
    )

    for reranker_key in args.rerankers:
        spec = RERANKER_SPECS[reranker_key]
        print(f"\n=== Benchmarking reranker {reranker_key} ({spec.model_id}) ===", flush=True)
        reranker = build_reranker(
            reranker_key,
            device=device,
            batch_size=args.reranker_batch_size,
            dtype_name=args.dtype,
        )
        try:
            for candidate_k in candidate_ks:
                reranked_ids_by_query: dict[str, list[str]] = {}
                reranked_scores_by_query: dict[str, list[float]] = {}

                rerank_start = time.perf_counter()
                for query_id, query_text in zip(data.query_ids, data.query_texts, strict=True):
                    candidate_ids = candidate_ids_by_query[candidate_k][query_id]
                    candidate_texts = [corpus_text_by_id[corpus_id] for corpus_id in candidate_ids]
                    pairs = list(zip([query_text] * len(candidate_texts), candidate_texts, strict=True))
                    scores = reranker.score_pairs(pairs)
                    order = np.argsort(-np.asarray(scores))
                    reranked_ids_by_query[query_id] = [candidate_ids[index] for index in order]
                    reranked_scores_by_query[query_id] = [float(scores[index]) for index in order]
                reranker_seconds = time.perf_counter() - rerank_start

                metrics = score_ranked_lists(
                    query_ids=data.query_ids,
                    ranked_corpus_ids_by_query=reranked_ids_by_query,
                    positives_by_query=data.positives_by_query,
                )

                write_rankings_csv(
                    model_name=f"{reranker_key}@{candidate_k}",
                    query_ids=data.query_ids,
                    ranked_corpus_ids_by_query=reranked_ids_by_query,
                    ranked_scores_by_query=reranked_scores_by_query,
                    positives_by_query=data.positives_by_query,
                    rankings_path=run_dir / "rankings" / f"{reranker_key}_k{candidate_k}.csv",
                )

                summary_rows.append(
                    summary_row(
                        stage="reranked",
                        candidate_k=candidate_k,
                        retriever_model=args.retriever_model,
                        reranker_model=reranker_key,
                        reranker_model_id=spec.model_id,
                        reranker_family=spec.family,
                        reranker_batch_size=reranker.batch_size,
                        stage1_recall_at_k=float(np.mean(candidate_recall_by_k[candidate_k])),
                        metrics=metrics,
                        retriever_query_stage_seconds=retriever_query_stage_seconds,
                        avg_reranker_latency_ms=(reranker_seconds / len(data.query_ids)) * 1000,
                        reranker_seconds=reranker_seconds,
                        notes=spec.notes,
                    )
                )
                write_summary_csv(summary_rows, run_dir / "summary.csv")
                (run_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2))
        finally:
            reranker.unload()

    write_summary_csv(summary_rows, run_dir / "summary.csv")
    (run_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2))
    report_path = build_report(run_dir)
    print(f"\nFinished reranker benchmark run.")
    print(f"Summary: {run_dir / 'summary.csv'}")
    print(f"Report:  {report_path}")
    print(f"Retriever document encoding seconds: {doc_seconds:.2f}")


if __name__ == "__main__":
    main()
