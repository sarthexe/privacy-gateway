"""Normalize a pinned public OOD dataset externally; never runs inference."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset

from scripts.evaluation_checkpoint import sha256_file
from scripts.gliner_evaluation import load_label_map
from scripts.nemotron_pii import is_within_repository, load_ontology
from scripts.ood_pii import DATASETS, OodNormalizationError, manifest, normalize


# Only fixed, source-independent adapter messages may enter the audit. Future
# errors containing dynamic source data must be redacted rather than persisted.
_SAFE_AUDIT_ERRORS = frozenset(
    {
        "source annotation offsets are not integers",
        "source annotation offsets are outside text",
        "source annotation value does not match its offsets",
        "source entity value is missing",
        "source entity cannot be uniquely reconstructed",
        "source annotation label is missing",
        "source text is missing",
        "language metadata is invalid",
        "AI4Privacy requires source_text and privacy_mask",
        "AI4Privacy privacy_mask item is invalid",
        "Gretel requires text and entities",
        "Gretel serialized entities are invalid",
        "Gretel entities must be a list",
        "Gretel entities must contain mappings",
        "Gretel entity item is invalid",
        "Argilla requires source-text and pii.suggestion",
        "Argilla pii.suggestion item is invalid",
        "unknown OOD dataset",
    }
)


def audit_error(error: OodNormalizationError) -> str:
    """Return a non-sensitive reason, never arbitrary exception details."""
    message = str(error)
    if message in _SAFE_AUDIT_ERRORS:
        return message
    return "normalization error (details redacted)"


def format_normalization_error(
    dataset: str, split: str, example_index: int, error: OodNormalizationError
) -> str:
    """Render actionable failure context without serializing source data."""
    return (
        f"error: normalization failed dataset={dataset} split={split} "
        f"example_index={example_index}: {error}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize a pinned OOD PII dataset (no inference).")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip entire invalid examples and write a non-sensitive audit JSONL.",
    )
    args = parser.parse_args(argv)
    if is_within_repository(args.output_dir):
        print("error: output directory must be outside the repository", file=sys.stderr)
        return 2
    spec = DATASETS[args.dataset]
    output = args.output_dir / f"{args.dataset}_{args.split}.jsonl"
    metadata = args.output_dir / f"{args.dataset}_{args.split}.metadata.json"
    audit = args.output_dir / f"{args.dataset}_{args.split}.audit.jsonl"
    if output.exists() or metadata.exists():
        print("error: normalized output or metadata already exists", file=sys.stderr)
        return 2
    if args.skip_invalid and audit.exists():
        print("error: normalization audit already exists", file=sys.stderr)
        return 2
    try:
        rows = load_dataset(spec.repository, revision=spec.revision, split=args.split)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        normalized_row_count = 0
        skipped_row_count = 0
        with ExitStack() as stack:
            handle = stack.enter_context(output.open("x", encoding="utf-8", newline="\n"))
            audit_handle = (
                stack.enter_context(audit.open("x", encoding="utf-8", newline="\n"))
                if args.skip_invalid
                else None
            )
            for index, row in enumerate(rows):
                try:
                    normalized = normalize(args.dataset, row, index, args.split, load_label_map())
                except OodNormalizationError as error:
                    if audit_handle is not None:
                        audit_handle.write(
                            json.dumps(
                                {"example_index": index, "error": audit_error(error)},
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        skipped_row_count += 1
                        continue
                    print(
                        format_normalization_error(args.dataset, args.split, index, error),
                        file=sys.stderr,
                    )
                    return 2
                # The entire example must normalize successfully before any of it
                # is written. Never emit the valid subset of an invalid example.
                handle.write(json.dumps(normalized, ensure_ascii=False, sort_keys=True) + "\n")
                normalized_row_count += 1
        metadata_payload = manifest(
            spec, args.split, normalized_row_count, sha256_file(output), load_ontology()
        )
        metadata_payload.update(
            {
                "source_row_count": len(rows),
                "normalized_row_count": normalized_row_count,
                "skipped_row_count": skipped_row_count,
                "audit_sha256": sha256_file(audit) if args.skip_invalid else None,
            }
        )
        metadata.write_text(
            json.dumps(metadata_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OodNormalizationError as error:
        print(f"error: normalization failed dataset={args.dataset} split={args.split}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
