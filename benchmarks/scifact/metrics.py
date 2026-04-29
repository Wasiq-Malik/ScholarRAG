from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EvalSummary:
    ndcg_at_10: float
    mrr_at_10: float
    recall_at_10: float
    recall_at_50: float
    recall_at_100: float
    map_at_10: float


def _dcg_at_k(relevances: list[float], k: int) -> float:
    score = 0.0
    for index, relevance in enumerate(relevances[:k], start=1):
        score += (2**relevance - 1.0) / math.log2(index + 1.0)
    return score


def _ndcg_at_k(ranked_relevances: list[float], ideal_relevances: list[float], k: int) -> float:
    ideal = _dcg_at_k(sorted(ideal_relevances, reverse=True), k)
    if ideal == 0:
        return 0.0
    return _dcg_at_k(ranked_relevances, k) / ideal


def _mrr_at_k(ranked_binary: list[int], k: int) -> float:
    for index, is_relevant in enumerate(ranked_binary[:k], start=1):
        if is_relevant:
            return 1.0 / index
    return 0.0


def _recall_at_k(ranked_binary: list[int], total_relevant: int, k: int) -> float:
    if total_relevant == 0:
        return 0.0
    return sum(ranked_binary[:k]) / total_relevant


def _average_precision_at_k(ranked_binary: list[int], total_relevant: int, k: int) -> float:
    if total_relevant == 0:
        return 0.0

    num_hits = 0
    precision_sum = 0.0
    for index, is_relevant in enumerate(ranked_binary[:k], start=1):
        if is_relevant:
            num_hits += 1
            precision_sum += num_hits / index
    return precision_sum / min(total_relevant, k)


def score_ranked_lists(
    *,
    query_ids: list[str],
    ranked_corpus_ids_by_query: dict[str, list[str]],
    positives_by_query: dict[str, dict[str, float]],
) -> EvalSummary:
    ndcg_values: list[float] = []
    mrr_values: list[float] = []
    recall10_values: list[float] = []
    recall50_values: list[float] = []
    recall100_values: list[float] = []
    map10_values: list[float] = []

    for query_id in query_ids:
        relevant_docs = positives_by_query[query_id]
        total_relevant = len(relevant_docs)
        ranked_ids = ranked_corpus_ids_by_query[query_id]

        ranked_binary = [1 if corpus_id in relevant_docs else 0 for corpus_id in ranked_ids]
        ranked_relevances = [relevant_docs.get(corpus_id, 0.0) for corpus_id in ranked_ids]
        ideal_relevances = list(relevant_docs.values())

        ndcg_values.append(_ndcg_at_k(ranked_relevances, ideal_relevances, 10))
        mrr_values.append(_mrr_at_k(ranked_binary, 10))
        recall10_values.append(_recall_at_k(ranked_binary, total_relevant, 10))
        recall50_values.append(_recall_at_k(ranked_binary, total_relevant, 50))
        recall100_values.append(_recall_at_k(ranked_binary, total_relevant, 100))
        map10_values.append(_average_precision_at_k(ranked_binary, total_relevant, 10))

    return EvalSummary(
        ndcg_at_10=float(np.mean(ndcg_values)),
        mrr_at_10=float(np.mean(mrr_values)),
        recall_at_10=float(np.mean(recall10_values)),
        recall_at_50=float(np.mean(recall50_values)),
        recall_at_100=float(np.mean(recall100_values)),
        map_at_10=float(np.mean(map10_values)),
    )


def write_rankings_csv(
    *,
    model_name: str,
    query_ids: list[str],
    ranked_corpus_ids_by_query: dict[str, list[str]],
    ranked_scores_by_query: dict[str, list[float]],
    positives_by_query: dict[str, dict[str, float]],
    rankings_path: Path,
) -> None:
    rankings_path.parent.mkdir(parents=True, exist_ok=True)
    with rankings_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "query_id",
                "rank",
                "corpus_id",
                "score",
                "is_relevant",
                "relevance",
            ]
        )

        for query_id in query_ids:
            relevant_docs = positives_by_query[query_id]
            ranked_ids = ranked_corpus_ids_by_query[query_id]
            ranked_scores = ranked_scores_by_query[query_id]

            for rank, (corpus_id, score) in enumerate(
                zip(ranked_ids, ranked_scores, strict=True),
                start=1,
            ):
                relevance = relevant_docs.get(corpus_id, 0.0)
                writer.writerow(
                    [
                        model_name,
                        query_id,
                        rank,
                        corpus_id,
                        float(score),
                        int(relevance > 0),
                        relevance,
                    ]
                )


def score_and_write_rankings(
    *,
    model_name: str,
    query_ids: list[str],
    corpus_ids: list[str],
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    positives_by_query: dict[str, dict[str, float]],
    rankings_path: Path,
    top_k_to_write: int,
) -> EvalSummary:
    similarity = query_embeddings @ corpus_embeddings.T

    ndcg_values: list[float] = []
    mrr_values: list[float] = []
    recall10_values: list[float] = []
    recall50_values: list[float] = []
    recall100_values: list[float] = []
    map10_values: list[float] = []

    rankings_path.parent.mkdir(parents=True, exist_ok=True)
    with rankings_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "query_id",
                "rank",
                "corpus_id",
                "score",
                "is_relevant",
                "relevance",
            ]
        )

        for query_index, query_id in enumerate(query_ids):
            relevant_docs = positives_by_query[query_id]
            total_relevant = len(relevant_docs)

            ranked_indices = np.argsort(-similarity[query_index])
            ranked_ids = [corpus_ids[index] for index in ranked_indices]

            ranked_binary = [1 if corpus_id in relevant_docs else 0 for corpus_id in ranked_ids]
            ranked_relevances = [relevant_docs.get(corpus_id, 0.0) for corpus_id in ranked_ids]
            ideal_relevances = list(relevant_docs.values())

            ndcg_values.append(_ndcg_at_k(ranked_relevances, ideal_relevances, 10))
            mrr_values.append(_mrr_at_k(ranked_binary, 10))
            recall10_values.append(_recall_at_k(ranked_binary, total_relevant, 10))
            recall50_values.append(_recall_at_k(ranked_binary, total_relevant, 50))
            recall100_values.append(_recall_at_k(ranked_binary, total_relevant, 100))
            map10_values.append(_average_precision_at_k(ranked_binary, total_relevant, 10))

            for rank, corpus_index in enumerate(ranked_indices[:top_k_to_write], start=1):
                corpus_id = corpus_ids[corpus_index]
                relevance = relevant_docs.get(corpus_id, 0.0)
                writer.writerow(
                    [
                        model_name,
                        query_id,
                        rank,
                        corpus_id,
                        float(similarity[query_index, corpus_index]),
                        int(relevance > 0),
                        relevance,
                    ]
                )

    return EvalSummary(
        ndcg_at_10=float(np.mean(ndcg_values)),
        mrr_at_10=float(np.mean(mrr_values)),
        recall_at_10=float(np.mean(recall10_values)),
        recall_at_50=float(np.mean(recall50_values)),
        recall_at_100=float(np.mean(recall100_values)),
        map_at_10=float(np.mean(map10_values)),
    )
