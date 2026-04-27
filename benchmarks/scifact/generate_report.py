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


def plot_metric(
    rows: list[dict[str, str]],
    *,
    metric_key: str,
    title: str,
    output_path: Path,
) -> None:
    rows = sorted(rows, key=lambda row: float(row[metric_key]), reverse=True)
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


def build_report(run_dir: Path) -> Path:
    summary_csv = run_dir / "summary.csv"
    config_json = run_dir / "config.json"
    rows = load_summary(summary_csv)
    if not rows:
        raise ValueError("No rows found in summary.csv")

    plots_dir = run_dir / "plots"
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
    )

    with config_json.open() as handle:
        config = json.load(handle)

    best_by_ndcg = max(rows, key=lambda row: float(row["ndcg_at_10"]))
    best_by_recall = max(rows, key=lambda row: float(row["recall_at_100"]))
    fastest = min(rows, key=lambda row: float(row["total_runtime_seconds"]))

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
            f"- Fastest total runtime: `{fastest['model']}` at `{float(fastest['total_runtime_seconds']):.2f}s`\n\n"
        )

        handle.write("## Metrics table\n\n")
        handle.write(
            "| Model | nDCG@10 | MRR@10 | Recall@10 | Recall@50 | Recall@100 | MAP@10 | Runtime (s) |\n"
        )
        handle.write(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        )
        for row in sorted(rows, key=lambda item: float(item["ndcg_at_10"]), reverse=True):
            handle.write(
                f"| {row['model']} | {float(row['ndcg_at_10']):.4f} | "
                f"{float(row['mrr_at_10']):.4f} | {float(row['recall_at_10']):.4f} | "
                f"{float(row['recall_at_50']):.4f} | {float(row['recall_at_100']):.4f} | "
                f"{float(row['map_at_10']):.4f} | {float(row['total_runtime_seconds']):.2f} |\n"
            )

        handle.write("\n## Plots\n\n")
        handle.write("- `plots/ndcg_at_10.png`\n")
        handle.write("- `plots/mrr_at_10.png`\n")
        handle.write("- `plots/recall_at_100.png`\n")
        handle.write("- `plots/runtime_seconds.png`\n")

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
