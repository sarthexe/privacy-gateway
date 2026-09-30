"""Normalize Nemotron-PII records to Privacy Gateway's span representation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import DatasetDict, load_dataset
from nemotron_pii import (
    DATASET_ID,
    AnnotationError,
    is_within_repository,
    load_ontology,
    normalize_example,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize nvidia/Nemotron-PII outside this repository."
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="External directory for normalized JSONL."
    )
    parser.add_argument(
        "--cache-dir", type=Path, help="Optional external Hugging Face cache directory."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if is_within_repository(args.output_dir):
        print("error: --output-dir must be outside the repository", file=sys.stderr)
        return 2
    if args.cache_dir and is_within_repository(args.cache_dir):
        print("error: --cache-dir must be outside the repository", file=sys.stderr)
        return 2
    try:
        dataset = load_dataset(
            DATASET_ID, cache_dir=str(args.cache_dir) if args.cache_dir else None
        )
    except Exception as error:
        print(f"error: could not load {DATASET_ID}: {error}", file=sys.stderr)
        return 1
    if not isinstance(dataset, DatasetDict):
        print(f"error: expected a DatasetDict, received {type(dataset).__name__}", file=sys.stderr)
        return 1

    ontology = load_ontology()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split in dataset.items():
        output_path = args.output_dir / f"{split_name}.jsonl"
        with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
            for row_index, record in enumerate(split):
                try:
                    normalized = normalize_example(record, ontology)
                except AnnotationError as error:
                    print(
                        f"error: malformed annotation in {split_name} row {row_index}: {error}",
                        file=sys.stderr,
                    )
                    return 1
                output_file.write(json.dumps(normalized, ensure_ascii=False, sort_keys=True))
                output_file.write("\n")
        # Never log text or spans; only aggregate structural output is safe.
        print(f"prepared split={split_name} rows={len(split)} output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
