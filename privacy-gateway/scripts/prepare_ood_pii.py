"""Normalize a pinned public OOD dataset externally; never runs inference."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Sequence
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import load_dataset
from scripts.evaluation_checkpoint import sha256_file
from scripts.gliner_evaluation import load_label_map
from scripts.nemotron_pii import is_within_repository, load_ontology
from scripts.ood_pii import DATASETS, manifest, normalize

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize a pinned OOD PII dataset (no inference).")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if is_within_repository(args.output_dir):
        print("error: output directory must be outside the repository", file=sys.stderr); return 2
    spec = DATASETS[args.dataset]; output = args.output_dir / f"{args.dataset}_{args.split}.jsonl"
    metadata = args.output_dir / f"{args.dataset}_{args.split}.metadata.json"
    if output.exists() or metadata.exists():
        print("error: normalized output or metadata already exists", file=sys.stderr); return 2
    try:
        rows = load_dataset(spec.repository, revision=spec.revision, split=args.split)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            for index, row in enumerate(rows):
                handle.write(json.dumps(normalize(args.dataset, row, index, args.split, load_label_map()), ensure_ascii=False, sort_keys=True) + "\n")
        metadata.write_text(json.dumps(manifest(spec, args.split, len(rows), sha256_file(output), load_ontology()), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as error:
        print(f"error: {type(error).__name__}", file=sys.stderr); return 2
    return 0
if __name__ == "__main__": raise SystemExit(main())
