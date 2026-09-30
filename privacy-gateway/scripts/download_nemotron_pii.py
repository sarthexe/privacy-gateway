"""Download and structurally inspect NVIDIA Nemotron-PII without copying it into Git."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import DatasetDict, load_dataset
from nemotron_pii import DATASET_ID, inspect_split, is_within_repository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and inspect nvidia/Nemotron-PII.")
    parser.add_argument("--cache-dir", type=Path, help="External Hugging Face cache directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cache_dir and is_within_repository(args.cache_dir):
        print("error: --cache-dir must be outside the repository", file=sys.stderr)
        return 2
    try:
        dataset = load_dataset(
            DATASET_ID, cache_dir=str(args.cache_dir) if args.cache_dir else None
        )
    except Exception as error:
        print(
            "error: could not download/load "
            f"{DATASET_ID}. Check network access, permissions, and cache path: {error}",
            file=sys.stderr,
        )
        return 1
    if not isinstance(dataset, DatasetDict):
        print(f"error: expected a DatasetDict, received {type(dataset).__name__}", file=sys.stderr)
        return 1

    report = {
        "dataset": DATASET_ID,
        "cache_dir": str(args.cache_dir) if args.cache_dir else "Hugging Face default cache",
        "splits": {
            name: inspect_split(split, list(split.column_names)) for name, split in dataset.items()
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
