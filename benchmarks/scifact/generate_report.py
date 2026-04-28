from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_summary(summary_path: Path) -> list[dict[str, str]]:
    with summary_path.open() as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def format_optional_float(row: dict[str, str], key: str, decimals: int = 2) -> str:
    value = row.get(key)
    if value in (None, ""):
        return "n/a"
    return f"{float(value):.{decimals}f}"


def has_metric(rows: list[dict[str, str]], key: str) -> bool:
    return all(row.get(key) not in (None, "") for row in rows)


def plot_metric(
    rows: list[dict[str, str]],
    *,
    metric_key: str,
    title: str,
    output_path: Path,
    higher_is_better: bool = True,
) -> None:
    rows = sorted(rows, key=lambda row: float(row[metric_key]), reverse=higher_is_better)
    labels = [row["model"] for row in rows]
    values = [float(row[metric_key]) for row in rows]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values)
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel(metric_key)
    plt.tight_layout()

    for bar, value in zip(bars, values, strict=True):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_tradeoff(
    rows: list[dict[str, str]],
    *,
    x_key: str,
    y_key: str,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 6))
    for row in rows:
        x = float(row[x_key])
        y = float(row[y_key])
        plt.scatter(x, y, s=80)
        plt.annotate(row["model"], (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)

    plt.title(title)
    plt.xlabel(x_key)
    plt.ylabel(y_key)
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
    plot_paths: list[str] = []
    has_query_latency = has_metric(rows, "avg_query_end_to_end_latency_ms")
    if len(rows) > 1:
        plot_metric(
            rows,
            metric_key="ndcg_at_10",
            title="SciFact nDCG@10",
            output_path=plots_dir / "ndcg_at_10.png",
        )
        plot_metric(
            rows,
            metric_key="mrr_at_10",
            title="SciFact MRR@10",
            output_path=plots_dir / "mrr_at_10.png",
        )
        plot_metric(
            rows,
            metric_key="recall_at_100",
            title="SciFact Recall@100",
            output_path=plots_dir / "recall_at_100.png",
        )
        plot_metric(
            rows,
            metric_key="total_runtime_seconds",
            title="Total Runtime (seconds)",
            output_path=plots_dir / "runtime_seconds.png",
            higher_is_better=False,
        )
        if has_query_latency:
            plot_metric(
                rows,
                metric_key="avg_query_end_to_end_latency_ms",
                title="Average Query End-to-End Latency (ms)",
                output_path=plots_dir / "avg_query_end_to_end_latency_ms.png",
                higher_is_better=False,
            )
            plot_tradeoff(
                rows,
                x_key="avg_query_end_to_end_latency_ms",
                y_key="ndcg_at_10",
                title="Quality vs Query Latency",
                output_path=plots_dir / "ndcg_vs_avg_query_latency.png",
            )
        plot_paths = [
            "plots/ndcg_at_10.png",
            "plots/mrr_at_10.png",
            "plots/recall_at_100.png",
            "plots/runtime_seconds.png",
        ]
        if has_query_latency:
            plot_paths.extend(
                [
                    "plots/avg_query_end_to_end_latency_ms.png",
                    "plots/ndcg_vs_avg_query_latency.png",
                ]
            )

    with config_json.open() as handle:
        config = json.load(handle)

    best_by_ndcg = max(rows, key=lambda row: float(row["ndcg_at_10"]))
    best_by_recall = max(rows, key=lambda row: float(row["recall_at_100"]))
    fastest = min(rows, key=lambda row: float(row["total_runtime_seconds"]))
    lowest_query_latency = (
        min(rows, key=lambda row: float(row["avg_query_end_to_end_latency_ms"]))
        if has_query_latency
        else None
    )

    report_path = run_dir / "REPORT.md"
    with report_path.open("w") as handle:
        handle.write("# SciFact Benchmark Report\n\n")
        handle.write("## Run configuration\n\n")
        handle.write(f"- Split: `{config['split']}`\n")
        handle.write(f"- Document mode: `{config['document_mode']}`\n")
        handle.write(f"- Device: `{config['device']}`\n")
        handle.write(f"- Dtype: `{config['dtype']}`\n")
        handle.write(f"- Query count: `{config['query_count']}`\n")
        handle.write(f"- Corpus size: `{config['corpus_size']}`\n")
        handle.write(f"- Models: `{', '.join(config['models'])}`\n\n")

        handle.write("## Headline results\n\n")
        handle.write(
            f"- Best `nDCG@10`: `{best_by_ndcg['model']}` at `{float(best_by_ndcg['ndcg_at_10']):.4f}`\n"
        )
        handle.write(
            f"- Best `Recall@100`: `{best_by_recall['model']}` at `{float(best_by_recall['recall_at_100']):.4f}`\n"
        )
        handle.write(
            f"- Fastest total runtime: `{fastest['model']}` at `{float(fastest['total_runtime_seconds']):.2f}s`\n"
        )
        if lowest_query_latency is not None:
            handle.write(
                f"- Lowest average query end-to-end latency: `{lowest_query_latency['model']}` at "
                f"`{float(lowest_query_latency['avg_query_end_to_end_latency_ms']):.2f} ms`\n\n"
            )
        else:
            handle.write("- Lowest average query end-to-end latency: `n/a` in this older run format\n\n")

        handle.write("## Metrics table\n\n")
        handle.write(
            "| Model | nDCG@10 | MRR@10 | Recall@10 | Recall@50 | Recall@100 | MAP@10 | Avg Query E2E (ms) | Runtime (s) |\n"
        )
        handle.write(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        )
        for row in sorted(rows, key=lambda item: float(item["ndcg_at_10"]), reverse=True):
            handle.write(
                f"| {row['model']} | {float(row['ndcg_at_10']):.4f} | "
                f"{float(row['mrr_at_10']):.4f} | {float(row['recall_at_10']):.4f} | "
                f"{float(row['recall_at_50']):.4f} | {float(row['recall_at_100']):.4f} | "
                f"{float(row['map_at_10']):.4f} | {format_optional_float(row, 'avg_query_end_to_end_latency_ms')} | "
                f"{float(row['total_runtime_seconds']):.2f} |\n"
            )

        handle.write("\n## Latency table\n\n")
        handle.write(
            "| Model | Batch | Doc Encode (s) | Query Encode (s) | Retrieval Scoring (s) | Avg Query Encode (ms) | Avg Retrieval Scoring (ms) | Avg Query E2E (ms) | Docs/s | Queries/s |\n"
        )
        handle.write(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        )
        latency_sorted_rows = (
            sorted(rows, key=lambda item: float(item["avg_query_end_to_end_latency_ms"]))
            if has_query_latency
            else rows
        )
        for row in latency_sorted_rows:
            handle.write(
                f"| {row['model']} | {row.get('batch_size', '-')} | "
                f"{format_optional_float(row, 'document_encoding_seconds')} | "
                f"{format_optional_float(row, 'query_encoding_seconds')} | "
                f"{format_optional_float(row, 'retrieval_scoring_seconds')} | "
                f"{format_optional_float(row, 'avg_query_encoding_latency_ms')} | "
                f"{format_optional_float(row, 'avg_retrieval_scoring_latency_ms')} | "
                f"{format_optional_float(row, 'avg_query_end_to_end_latency_ms')} | "
                f"{format_optional_float(row, 'documents_per_second')} | "
                f"{format_optional_float(row, 'queries_per_second')} |\n"
            )

        handle.write("\n## Plots\n\n")
        if plot_paths:
            for plot_path in plot_paths:
                handle.write(f"- `{plot_path}`\n")
        else:
            handle.write("Skipped comparison plots because only one model result is present.\n")

    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plots and a markdown report for a SciFact benchmark run.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to a benchmark run directory containing summary.csv and config.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = build_report(args.run_dir)
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
