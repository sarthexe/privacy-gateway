"""Select the GLiNER confidence threshold on the development subset only.

Inputs are dev predictions produced once at the lowest candidate threshold. GLiNER's
flat-NER decoder keeps spans with ``score > t`` greedily in descending score order,
so the predictions at any higher ``t`` are exactly the stored ones with
``score > t``; ``--verify`` checks that claim against a direct run at a higher
threshold. The pre-registered rule: pick the candidate with the highest dev
gateway-ontology micro-F1; ties (within 1e-4) go to the lower threshold, the
recall-favouring choice for a privacy filter. The test split is never read.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_gliner import ScoringError, coverage_section, run_scoring
from scripts.gliner_evaluation import DEFAULT_LABEL_MAP, apply_threshold, load_label_map
from scripts.gliner_predictions import iter_records, read_header

CANDIDATES = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
TIE_TOLERANCE = 1e-4
RULE = (
    "maximize development-subset gateway-ontology micro-F1 (strict exact span/type); "
    f"ties within {TIE_TOLERANCE} go to the lower threshold"
)


def select(results: dict[str, dict[str, Any]]) -> float:
    best = max(row["gateway"]["f1"] for row in results.values())
    tied = [float(t) for t, row in results.items() if row["gateway"]["f1"] >= best - TIE_TOLERANCE]
    return min(tied)


def verify_post_filter(base: Path, direct: Path) -> dict[str, Any]:
    """Compare stored-then-filtered predictions against a direct run at a higher threshold."""
    base_header, direct_header = read_header(base), read_header(direct)
    if (
        base_header["source"] != direct_header["source"]
        or base_header["model"] != (direct_header["model"])
    ):
        raise ScoringError("verification run uses a different source or model")
    threshold = float(direct_header["inference"]["threshold"])
    stored = {record.example_index: record for record in iter_records(base)}
    compared = identical = only_filtered = only_direct = 0
    for record in iter_records(direct):
        expected = {
            (p.label, p.start, p.end)
            for p in apply_threshold(stored[record.example_index].predictions, threshold)
        }
        actual = {(p.label, p.start, p.end) for p in record.predictions}
        compared += 1
        identical += expected == actual
        only_filtered += len(expected - actual)
        only_direct += len(actual - expected)
    return {
        "direct_threshold": threshold,
        "examples_compared": compared,
        "examples_identical": identical,
        "predictions_only_in_post_filtered": only_filtered,
        "predictions_only_in_direct_run": only_direct,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dev-only GLiNER threshold selection.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True, help="Dev predictions file.")
    parser.add_argument("--output", type=Path, required=True, help="Selection JSON.")
    parser.add_argument("--verify", type=Path, help="Dev predictions run directly at a higher t.")
    parser.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP)
    args = parser.parse_args(argv)
    header = read_header(args.predictions)
    if header["source"]["source"] != "dev":
        print("error: threshold selection accepts development predictions only", file=sys.stderr)
        return 2
    if float(header["inference"]["threshold"]) > min(CANDIDATES):
        print("error: dev predictions must be produced at the lowest candidate", file=sys.stderr)
        return 2
    label_map = load_label_map(args.label_map)
    results: dict[str, dict[str, Any]] = {}
    complete = False
    for threshold in CANDIDATES:
        score, _, complete = run_scoring(args.data_dir, args.predictions, threshold, label_map)
        totals = score.state.aggregate()
        tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        coverage = coverage_section(score, label_map)
        results[f"{threshold}"] = {
            "gateway": {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            },
            "native": coverage["native_label_overall"],
            "predictions_removed_by_ontology_filtering": coverage[
                "predictions_removed_by_ontology_filtering"
            ],
        }
        print(
            f"dev threshold={threshold}: gateway precision={precision:.4f} recall={recall:.4f} "
            f"f1={f1:.4f}; native f1={coverage['native_label_overall']['f1']:.4f}",
            flush=True,
        )
    if not complete:
        print("error: dev predictions are incomplete", file=sys.stderr)
        return 2
    selected = select(results)
    payload: dict[str, Any] = {
        "source": header["source"],
        "model": {
            "model_id": header["model"]["model_id"],
            "model_revision": header["model"]["model_revision"],
        },
        "inference": header["inference"],
        "candidates": list(CANDIDATES),
        "rule": RULE,
        "results": results,
        "selected_threshold": selected,
        "reference_threshold_model_card": 0.3,
        "caveat": (
            "The development subset is drawn from the Nemotron-PII train split, which the "
            "model card reports as GLiNER-PII's training data; dev scores are in-sample and "
            "likely optimistic. The test split was not read."
        ),
    }
    if args.verify:
        payload["post_filter_verification"] = verify_post_filter(args.predictions, args.verify)
        print(f"verification: {json.dumps(payload['post_filter_verification'])}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"selected threshold={selected} ({RULE})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
