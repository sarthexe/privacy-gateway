"""Offline EMAIL_ADDRESS hybrid experiment for stored GLiNER-24 predictions.

This module deliberately has no model dependency and never writes text or entity
values.  It reads a normalized Nemotron JSONL plus an existing offset-only GLiNER
prediction file, writes a *new* offset-only prediction file, and scores both
variants with the existing strict ``score_batch`` harness.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.batch_evaluation import Batch, Prediction, score_batch
from scripts.evaluation_checkpoint import EvaluationState, canonical_sha256, sha256_file
from scripts.gliner_evaluation import (
    GlinerLabelMap,
    RawPrediction,
    deduplicate,
    load_label_map,
    to_gateway,
)
from scripts.gliner_predictions import PredictionRecord, PredictionWriter, iter_records, read_header
from scripts.nemotron_pii import is_within_repository
from scripts.presidio_evaluation import ExactMatch, metric_rows, validate_normalized_example

EMAIL_LABEL = "email"
EMAIL_SCORE = 1.0
CONTAINED_GATEWAY_TYPES = frozenset({"PERSON", "USERNAME", "ORGANIZATION"})
SCORE_BATCH_SIZE = 1_000

# The lookarounds keep surrounding punctuation out while accepting conventional
# local parts and domain labels, including subdomains.  This is intentionally a
# recognizer module boundary: future IP/URL/cookie recognizers can implement the
# same ``recognize(text) -> list[RawPrediction]`` contract.
EMAIL_PATTERN = re.compile(
    r"(?<![A-Za-z0-9.!#$%&'*+/=?^_`{|}~-])"
    r"([A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*"
    r"@(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,63})"
    r"(?![A-Za-z0-9-])"
)


class HybridExperimentError(ValueError):
    """Inputs are invalid or cannot be safely compared; messages contain no PII."""


def recognize_email_addresses(text: str) -> list[RawPrediction]:
    """Return deterministic email spans using absolute half-open character offsets."""
    return [
        RawPrediction(EMAIL_LABEL, match.start(1), match.end(1), EMAIL_SCORE)
        for match in EMAIL_PATTERN.finditer(text)
    ]


def _strictly_contained(prediction: RawPrediction, container: RawPrediction) -> bool:
    return container.start <= prediction.start and prediction.end <= container.end


def hybrid_predictions(
    text: str, gliner_predictions: Iterable[RawPrediction], label_map: GlinerLabelMap
) -> tuple[list[RawPrediction], int]:
    """Add deterministic emails and suppress only configured GLiNER spans inside them."""
    emails = recognize_email_addresses(text)
    retained = [
        prediction
        for prediction in gliner_predictions
        if not (
            label_map.gateway_type(prediction.label) in CONTAINED_GATEWAY_TYPES
            and any(_strictly_contained(prediction, email) for email in emails)
        )
    ]
    result, duplicates = deduplicate([*retained, *emails])
    return result, duplicates


def _state_metrics(state: EvaluationState) -> dict[str, Any]:
    totals = state.aggregate()
    support = Counter({
        entity: state.true_positives[entity] + state.false_negatives[entity]
        for entity in state.entity_types()
    })
    predicted = Counter({
        entity: state.true_positives[entity] + state.false_positives[entity]
        for entity in state.entity_types()
    })
    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0.0
    recall = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "tp": totals["tp"],
        "fp": totals["fp"],
        "fn": totals["fn"],
        "predictions": totals["tp"] + totals["fp"],
        "per_entity": metric_rows(+support, +predicted, +state.true_positives),
    }


def _gateway_prediction(
    record: PredictionRecord, label_map: GlinerLabelMap
) -> tuple[ExactMatch, ...]:
    return tuple(to_gateway(record.predictions, label_map).kept)


def _flush_scores(
    original: EvaluationState,
    hybrid: EvaluationState,
    pending: list[
        tuple[int, tuple[ExactMatch, ...], tuple[ExactMatch, ...], tuple[ExactMatch, ...]]
    ],
    reached_eof: bool,
) -> None:
    if not pending:
        return
    batch = Batch([(index, b"") for index, _, _, _ in pending], 0, reached_eof)
    score_batch(original, batch, [(truth, old) for _, truth, old, _ in pending])
    score_batch(hybrid, batch, [(truth, new) for _, truth, _, new in pending])
    pending.clear()


def _record_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _per_entity(metrics: dict[str, Any], entity_type: str) -> dict[str, int | float]:
    """Return a stable zero row when a type is absent from both truth and predictions."""
    return dict(metrics["per_entity"].get(entity_type, {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "support": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "exact_matches": 0,
        "predictions": 0,
    }))


def _validate_max_width(identity: dict[str, Any]) -> None:
    try:
        max_width = identity["model"]["max_width"]
    except (KeyError, TypeError) as error:
        raise HybridExperimentError("prediction header has no model max_width") from error
    if max_width != 24:
        raise HybridExperimentError(
            "hybrid experiment requires GLiNER predictions with max_width=24"
        )


def run_experiment(
    normalized_path: Path, original_path: Path, hybrid_path: Path, label_map: GlinerLabelMap
) -> dict[str, Any]:
    """Create a new artifact and compare original/hybrid strict metrics in one stream."""
    if is_within_repository(hybrid_path):
        raise HybridExperimentError("hybrid predictions path must be outside the repository")
    original_identity = read_header(original_path)
    _validate_max_width(original_identity)
    expected = _record_count(normalized_path)
    output_identity = {
        "experiment": "deterministic_email_hybrid/v1",
        "input_predictions_sha256": sha256_file(original_path),
        "normalized_data_sha256": sha256_file(normalized_path),
        "source_identity": original_identity,
        "recognizers": ["email"],
        "conflict_resolution": "suppress_person_username_organization_strictly_contained_in_email",
    }
    writer = PredictionWriter(
        hybrid_path, output_identity, range(expected), resume=False, restart=False
    )
    original_state, hybrid_state = EvaluationState(), EvaluationState()
    pending: list[
        tuple[int, tuple[ExactMatch, ...], tuple[ExactMatch, ...], tuple[ExactMatch, ...]]
    ] = []
    total_before = total_after = 0
    records = iter_records(original_path)
    with normalized_path.open("r", encoding="utf-8") as data:
        for index, line in enumerate(data):
            record = next(records, None)
            if record is None or record.example_index != index:
                raise HybridExperimentError("prediction records do not align with normalized data")
            try:
                text, truth = validate_normalized_example(json.loads(line))
            except (ValueError, json.JSONDecodeError) as error:
                raise HybridExperimentError("normalized data has an invalid record") from error
            if any(p.end > len(text) for p in record.predictions):
                raise HybridExperimentError("prediction offsets are outside normalized text")
            transformed, removed_duplicates = hybrid_predictions(
                text, record.predictions, label_map
            )
            output = PredictionRecord(
                index,
                record.chunks,
                record.duplicates_removed + removed_duplicates,
                tuple(transformed),
            )
            writer.append([output])
            old_gateway = _gateway_prediction(record, label_map)
            new_gateway = _gateway_prediction(output, label_map)
            pending.append((index, tuple(truth), old_gateway, new_gateway))
            total_before += len(record.predictions)
            total_after += len(output.predictions)
            if len(pending) == SCORE_BATCH_SIZE:
                _flush_scores(original_state, hybrid_state, pending, False)
    if next(records, None) is not None:
        raise HybridExperimentError("prediction file has records beyond normalized data")
    _flush_scores(original_state, hybrid_state, pending, True)
    original_metrics, hybrid_metrics = _state_metrics(original_state), _state_metrics(hybrid_state)
    return {
        "experiment": "deterministic_email_hybrid/v1",
        "privacy": "aggregate metrics and offsets only; no source text or entity values",
        "input": {
            "normalized_data_sha256": output_identity["normalized_data_sha256"],
            "original_predictions_sha256": output_identity["input_predictions_sha256"],
        },
        "original_gliner_24": original_metrics,
        "hybrid": hybrid_metrics,
        "deltas": {key: hybrid_metrics[key] - original_metrics[key] for key in ("tp", "fp", "fn")},
        "email_address_before": _per_entity(original_metrics, "EMAIL_ADDRESS"),
        "email_address_after": _per_entity(hybrid_metrics, "EMAIL_ADDRESS"),
        "person_before": _per_entity(original_metrics, "PERSON"),
        "person_after": _per_entity(hybrid_metrics, "PERSON"),
        "total_predictions_before": total_before,
        "total_predictions_after": total_after,
        "output_identity_sha256": canonical_sha256(output_identity),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline deterministic-email hybrid for GLiNER max_width=24."
    )
    parser.add_argument(
        "--normalized-test",
        type=Path,
        required=True,
        help="Existing normalized Nemotron test JSONL.",
    )
    parser.add_argument(
        "--gliner-predictions",
        type=Path,
        required=True,
        help="Existing GLiNER-24 offset-only JSONL.",
    )
    parser.add_argument(
        "--hybrid-predictions", type=Path, required=True, help="New external offset-only JSONL."
    )
    parser.add_argument(
        "--report", type=Path, required=True, help="New aggregate-only JSON report."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.report.exists():
        print("error: report already exists", file=sys.stderr)
        return 2
    try:
        report = run_experiment(
            args.normalized_test,
            args.gliner_predictions,
            args.hybrid_predictions,
            load_label_map(),
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (HybridExperimentError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
