from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset, load_dataset_builder


DATASET_NAME = "open-index/open-arxiv"


def format_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unknown"

    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def print_dataset_info() -> None:
    builder = load_dataset_builder(DATASET_NAME)
    builder_info = builder.info
    description = (builder_info.description or "").strip()
    description_line = description.splitlines()[0] if description else "n/a"

    print("Dataset:", DATASET_NAME)
    print("Description:", description_line)
    print("Download size:", format_bytes(builder_info.download_size))
    print("Generated dataset size:", format_bytes(builder_info.dataset_size))
    print("Features:")
    for name, feature in builder_info.features.items():
        print(f"  - {name}: {feature}")

    train_split = builder_info.splits.get("train")
    if train_split is not None:
        print("Train rows:", f"{train_split.num_examples:,}")


def print_head(rows: int) -> None:
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)

    print(f"\nStreaming first {rows} rows:")
    for idx, record in enumerate(dataset.take(rows), start=1):
        authors_parsed = json.loads(record["authors_parsed"])
        versions = json.loads(record["versions"])
        first_authors = ", ".join(
            f"{first} {last}".strip()
            for last, first, _suffix in authors_parsed[:3]
        )

        print(f"\n[{idx}] id={record['id']}")
        print(f"title: {record['title']}")
        print(f"authors: {first_authors}")
        print(f"categories: {record['categories']}")
        print(f"updated: {record['update_date']}")
        print(f"versions: {len(versions)}")
        print(f"abstract: {record['abstract'][:400]}")


def maybe_download_local(download_dir: Path) -> None:
    print(f"\nDownloading dataset locally to: {download_dir}")
    dataset = load_dataset(DATASET_NAME, split="train", cache_dir=str(download_dir))
    print("Local rows available:", f"{len(dataset):,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the Hugging Face open-index/open-arxiv dataset."
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=3,
        help="How many sample rows to print from the streamed dataset.",
    )
    parser.add_argument(
        "--download-local",
        action="store_true",
        help="Download/cache the dataset locally after printing metadata and sample rows.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/hf_cache"),
        help="Directory to use for local Hugging Face dataset cache.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_dataset_info()
    print_head(args.rows)

    if args.download_local:
        maybe_download_local(args.cache_dir)


if __name__ == "__main__":
    main()
