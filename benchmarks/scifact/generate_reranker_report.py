from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_summary(summary_path: Path) -> list[dict[str, str]]:
    with summary_path.open() as handle:
        return list(csv.DictReader(handle))


def row_label(row: dict[str, str]) -> str:
    if row["stage"] == "retriever_only":
        return row["retriever_model"]
    return f"{row['reranker_model']}@{row['candidate_k']}"


def plot_metric(
    rows: list[dict[str, str]],
    *,
    metric_key: str,
    title: str,
    output_path: Path,
    higher_is_better: bool = True,
) -> None:
    rows = sorted(rows, key=lambda row: float(row[metric_key]), reverse=higher_is_better)
    labels = [row_label(row) for row in rows]
    values = [float(row[metric_key]) for row in rows]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, values)
    plt.title(title)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel(metric_key)
    plt.tight_layout()
    for bar, value in zip(bars, values, strict=True):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_tradeoff(rows: list[dict[str, str]], *, output_path: Path) -> None:
    plt.figure(figsize=(9, 6))
    for row in rows:
        x = float(row["avg_total_query_latency_ms"])
        y = float(row["ndcg_at_10"])
        plt.scatter(x, y, s=80)
        plt.annotate(row_label(row), (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)

    plt.title("Reranker Quality vs Total Query Latency")
    plt.xlabel("avg_total_query_latency_ms")
    plt.ylabel("ndcg_at_10")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def build_report(run_dir: Path) -> Path:
    summary_csv = run_dir / "summary.csv"
    config_json = run_dir / "config.json"
    rows = load_summary(summary_csv)
    if not rows:
        raise ValueError("No rows found in summary.csv")

    plots_dir = run_dir / "plots"
    if len(rows) > 1:
        plot_metric(rows, metric_key="ndcg_at_10", title="SciFact nDCG@10", output_path=plots_dir / "ndcg_at_10.png")
        plot_metric(rows, metric_key="mrr_at_10", title="SciFact MRR@10", output_path=plots_dir / "mrr_at_10.png")
        plot_metric(rows, metric_key="map_at_10", title="SciFact MAP@10", output_path=plots_dir / "map_at_10.png")
        plot_metric(
            rows,
            metric_key="avg_total_query_latency_ms",
            title="Average Total Query Latency (ms)",
            output_path=plots_dir / "avg_total_query_latency_ms.png",
            higher_is_better=False,
        )
        plot_tradeoff(rows, output_path=plots_dir / "ndcg_vs_total_query_latency.png")

    with config_json.open() as handle:
        config = json.load(handle)

    best_ndcg = max(rows, key=lambda row: float(row["ndcg_at_10"]))
    best_map = max(rows, key=lambda row: float(row["map_at_10"]))
    lowest_latency = min(rows, key=lambda row: float(row["avg_total_query_latency_ms"]))

    report_path = run_dir / "REPORT.md"
    with report_path.open("w") as handle:
        handle.write("# SciFact Reranker Report\n\n")
        handle.write("## Run configuration\n\n")
        handle.write(f"- Retriever: `{config['retriever_model']}`\n")
        handle.write(f"- Split: `{config['split']}`\n")
        handle.write(f"- Document mode: `{config['document_mode']}`\n")
        handle.write(f"- Candidate Ks: `{config['candidate_ks']}`\n")
        handle.write(f"- Rerankers: `{', '.join(config['rerankers'])}`\n\n")

        handle.write("## Headline results\n\n")
        handle.write(f"- Best `nDCG@10`: `{row_label(best_ndcg)}` at `{float(best_ndcg['ndcg_at_10']):.4f}`\n")
        handle.write(f"- Best `MAP@10`: `{row_label(best_map)}` at `{float(best_map['map_at_10']):.4f}`\n")
        handle.write(
            f"- Lowest average total query latency: `{row_label(lowest_latency)}` at "
            f"`{float(lowest_latency['avg_total_query_latency_ms']):.2f} ms`\n\n"
        )

        handle.write("## Metrics table\n\n")
        handle.write(
            "| Label | Stage | Candidate K | Stage1 Recall@K | nDCG@10 | MRR@10 | MAP@10 | Recall@10 | Avg Rerank (ms) | Avg Total Query (ms) |\n"
        )
        handle.write(
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        )
        for row in sorted(rows, key=lambda row: float(row["ndcg_at_10"]), reverse=True):
            handle.write(
                f"| {row_label(row)} | {row['stage']} | {row['candidate_k']} | "
                f"{float(row['stage1_recall_at_k']):.4f} | {float(row['ndcg_at_10']):.4f} | "
                f"{float(row['mrr_at_10']):.4f} | {float(row['map_at_10']):.4f} | "
                f"{float(row['recall_at_10']):.4f} | {float(row['avg_reranker_latency_ms']):.2f} | "
                f"{float(row['avg_total_query_latency_ms']):.2f} |\n"
            )

        handle.write("\n## Plots\n\n")
        for plot_name in [
            "plots/ndcg_at_10.png",
            "plots/mrr_at_10.png",
            "plots/map_at_10.png",
            "plots/avg_total_query_latency_ms.png",
            "plots/ndcg_vs_total_query_latency.png",
        ]:
            handle.write(f"- `{plot_name}`\n")

    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plots and a markdown report for a SciFact reranker run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = build_report(args.run_dir)
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
