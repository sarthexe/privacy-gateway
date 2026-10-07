"""Offline PERSON-inside-email suppression experiment.

Reads normalized OOD records and persisted GLiNER predictions, builds an in-memory
scoring view, and writes aggregate metrics only. It does not run GLiNER, does not
rewrite prediction or dataset files, and does not change production detection.

Eligibility uses source text and the model prediction only. Ground truth is read
later, and only to score a view that has already been chosen.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.batch_evaluation import Batch, score_batch
from scripts.evaluation_checkpoint import EvaluationState, sha256_file
from scripts.gliner_email_hybrid import recognize_email_addresses
from scripts.gliner_evaluation import (
    UNMAPPED,
    GlinerLabelMap,
    RawPrediction,
    apply_threshold,
    load_label_map,
    to_gateway,
)
from scripts.gliner_predictions import iter_records, read_header
from scripts.nemotron_pii import is_within_repository
from scripts.presidio_evaluation import (
    SUPPORTED_TYPES,
    ExactMatch,
    adapter_category,
    evaluation_type,
    validate_normalized_example,
)

# Frozen recognizer. Validity is whatever this function emits: no DNS, no network,
# and no second parser. Spans it does not return are not email candidates.
EMAIL_RECOGNIZER = recognize_email_addresses

EXPERIMENT_ID = "person_in_email_strict_suppression/v1"
ALLOWED_THRESHOLDS = frozenset({0.3, 0.7})
SCORE_BATCH_SIZE = 1_000
EXPECTED_NORMALIZED_SHA256 = {
    "ai4privacy": "575c7105a3ad5d27cbea7a467bebc360b08edfcdebadd71a45d5962a12472fca",
    "gretel": "2b628ffc09effc56475300038c648252a5faf9eb4f8877c9221ef31e5ca727f9",
    "argilla": "551e98472d8c0df3e5ec1590f0b62960d3be667266eadf65518715fd534ee6cf",
}


class PersonEmailExperimentError(ValueError):
    """The experiment cannot be scored safely. Messages contain no source text."""


@dataclass(frozen=True)
class Classification:
    """Thresholded predictions split by containment. Order of ``kept`` is original order."""

    kept: tuple[RawPrediction, ...]
    strict: tuple[RawPrediction, ...]
    loose: tuple[RawPrediction, ...]

    @property
    def loose_only(self) -> tuple[RawPrediction, ...]:
        strict_ids = {id(item) for item in self.strict}
        return tuple(item for item in self.loose if id(item) not in strict_ids)


def strictly_inside(inner_start: int, inner_end: int, outer_start: int, outer_end: int) -> bool:
    """True only when the inner span is strictly inside the outer half-open span."""
    return outer_start < inner_start < inner_end < outer_end


def spans_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    """Half-open overlap. Touching endpoints do not overlap."""
    return start_a < end_b and start_b < end_a


def _gateway_type(prediction: RawPrediction, label_map: GlinerLabelMap) -> str:
    try:
        return label_map.gateway_type(prediction.label)
    except KeyError as error:
        raise PersonEmailExperimentError(
            "prediction label is outside the pinned label map"
        ) from error


def _require_text(text: str, predictions: Sequence[RawPrediction]) -> None:
    if not isinstance(text, str):
        raise PersonEmailExperimentError("source text is missing")
    for prediction in predictions:
        if prediction.start < 0 or prediction.end <= prediction.start or prediction.end > len(text):
            raise PersonEmailExperimentError("prediction offsets are outside source text")


def classify_predictions(
    text: str,
    predictions: Sequence[RawPrediction],
    label_map: GlinerLabelMap,
) -> Classification:
    """Choose suppressions from source text and predictions. Ground truth is not an input."""
    _require_text(text, predictions)
    emails = tuple(EMAIL_RECOGNIZER(text))
    kept: list[RawPrediction] = []
    strict: list[RawPrediction] = []
    loose: list[RawPrediction] = []
    for prediction in predictions:
        is_person = _gateway_type(prediction, label_map) == "PERSON"
        contained = is_person and any(
            strictly_inside(prediction.start, prediction.end, email.start, email.end)
            for email in emails
        )
        overlapped = is_person and any(
            spans_overlap(prediction.start, prediction.end, email.start, email.end)
            for email in emails
        )
        if contained:
            strict.append(prediction)
            loose.append(prediction)
        elif overlapped:
            loose.append(prediction)
            kept.append(prediction)
        else:
            kept.append(prediction)
    return Classification(tuple(kept), tuple(strict), tuple(loose))


def scoring_views(
    text: str,
    predictions: Sequence[RawPrediction],
    label_map: GlinerLabelMap,
    threshold: float,
) -> tuple[tuple[RawPrediction, ...], Classification]:
    """Apply the persisted-score threshold, then the strict suppression view."""
    if threshold not in ALLOWED_THRESHOLDS:
        raise PersonEmailExperimentError("threshold must stay at 0.3 or 0.7")
    baseline = tuple(apply_threshold(predictions, threshold))
    return baseline, classify_predictions(text, baseline, label_map)


def _prediction_id(prediction: RawPrediction) -> tuple[str, int, int, float]:
    return (prediction.label, prediction.start, prediction.end, prediction.score)


def removal_is_subsequence(
    baseline: Sequence[RawPrediction], suppressed_view: Sequence[RawPrediction]
) -> int | None:
    """Return how many baseline rows were dropped, or None if anything else changed."""
    index = 0
    removed = 0
    for prediction in suppressed_view:
        while index < len(baseline) and _prediction_id(baseline[index]) != _prediction_id(
            prediction
        ):
            removed += 1
            index += 1
        if index == len(baseline):
            return None
        index += 1
    return removed + (len(baseline) - index)


def _rates(tp: int, fp: int, fn: int) -> dict[str, int | float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _entity_rates(state: EvaluationState, entity_type: str) -> dict[str, int | float]:
    return _rates(
        state.true_positives[entity_type],
        state.false_positives[entity_type],
        state.false_negatives[entity_type],
    )


def _delta(
    baseline: Mapping[str, int | float], candidate: Mapping[str, int | float]
) -> dict[str, int | float]:
    return {
        key: candidate[key] - baseline[key]
        for key in ("tp", "fp", "fn", "precision", "recall", "f1")
    }


def _supported(entities: Sequence[ExactMatch]) -> tuple[ExactMatch, ...]:
    return tuple(entity for entity in entities if entity.entity_type in SUPPORTED_TYPES)


def _gateway_entities(
    predictions: Sequence[RawPrediction], label_map: GlinerLabelMap
) -> tuple[ExactMatch, ...]:
    return tuple(to_gateway(predictions, label_map).kept)


def _empty_account() -> dict[str, int]:
    return {
        "original_person_predictions": 0,
        "eligible_strict_suppressions": 0,
        "loose_overlap_candidates": 0,
        "loose_only_candidates": 0,
        "exact_person_gt_matches": 0,
        "person_gt_mismatches": 0,
        "overlapping_email_gt": 0,
        "overlapping_unmapped_gt": 0,
        "overlapping_unsupported_gt": 0,
        "overlapping_other_supported_gt": 0,
        "no_gt_overlap": 0,
    }


def account_suppressed(
    suppressed: Sequence[RawPrediction],
    truth: Sequence[ExactMatch],
    label_map: GlinerLabelMap,
) -> dict[str, int]:
    """Classify an already-chosen suppression set. This is evaluation, not eligibility."""
    counts = _empty_account()
    counts["eligible_strict_suppressions"] = len(suppressed)
    for prediction in suppressed:
        if _gateway_type(prediction, label_map) != "PERSON":
            raise PersonEmailExperimentError("suppression accounting saw a non-PERSON prediction")
        span = ExactMatch(evaluation_type("PERSON"), prediction.start, prediction.end)
        exact = any(
            item.entity_type == "PERSON" and item.start == span.start and item.end == span.end
            for item in truth
        )
        overlaps_person = overlaps_email = overlaps_unmapped = False
        overlaps_unsupported = overlaps_other = overlaps_any = False
        for item in truth:
            if not spans_overlap(span.start, span.end, item.start, item.end):
                continue
            overlaps_any = True
            if item.entity_type == "PERSON":
                overlaps_person = True
            elif item.entity_type == "EMAIL_ADDRESS":
                overlaps_email = True
            elif item.entity_type == UNMAPPED:
                overlaps_unmapped = True
            elif adapter_category(item.entity_type) == "unsupported_entity":
                overlaps_unsupported = True
            else:
                overlaps_other = True
        if exact:
            counts["exact_person_gt_matches"] += 1
        elif overlaps_person:
            counts["person_gt_mismatches"] += 1
        if overlaps_email:
            counts["overlapping_email_gt"] += 1
        if overlaps_unmapped:
            counts["overlapping_unmapped_gt"] += 1
        if overlaps_unsupported:
            counts["overlapping_unsupported_gt"] += 1
        if overlaps_other:
            counts["overlapping_other_supported_gt"] += 1
        if not overlaps_any:
            counts["no_gt_overlap"] += 1
    return counts


def _add_counts(total: dict[str, int], extra: Mapping[str, int]) -> None:
    for key, value in extra.items():
        total[key] += value


def _as_tuples(entities: Sequence[ExactMatch]) -> tuple[tuple[str, int, int], ...]:
    """Adapt matches to the tuple rows ``score_batch`` unpacks."""
    return tuple((entity.entity_type, entity.start, entity.end) for entity in entities)


def _flush(
    states: dict[str, EvaluationState],
    pending: list[
        tuple[int, tuple[ExactMatch, ...], tuple[ExactMatch, ...], tuple[ExactMatch, ...]]
    ],
    reached_eof: bool,
) -> None:
    if not pending:
        return
    if not any(truth or baseline or candidate for _, truth, baseline, candidate in pending):
        pending.clear()
        return
    batch = Batch([(index, b"") for index, _, _, _ in pending], 0, reached_eof)

    def pair(
        left: Sequence[ExactMatch], right: Sequence[ExactMatch]
    ) -> tuple[tuple[tuple[str, int, int], ...], tuple[tuple[str, int, int], ...]]:
        return (_as_tuples(left), _as_tuples(right))

    pairs = {
        "baseline_full": [pair(truth, baseline) for _, truth, baseline, _ in pending],
        "candidate_full": [pair(truth, candidate) for _, truth, _, candidate in pending],
        "baseline_supported": [
            pair(_supported(truth), _supported(baseline)) for _, truth, baseline, _ in pending
        ],
        "candidate_supported": [
            pair(_supported(truth), _supported(candidate)) for _, truth, _, candidate in pending
        ],
    }
    for name, predictions in pairs.items():
        if any(truth or predicted for truth, predicted in predictions):
            score_batch(states[name], batch, predictions)
    pending.clear()


def _require_max_width(identity: Mapping[str, Any]) -> None:
    found: list[object] = []
    for key in ("model", "protocol"):
        section = identity.get(key)
        if isinstance(section, Mapping) and "max_width" in section:
            found.append(section["max_width"])
    if not found or any(value != 24 for value in found):
        raise PersonEmailExperimentError(
            "experiment requires persisted predictions with max_width=24"
        )


def _validate_paths(normalized: Path, predictions: Path, report: Path) -> None:
    if not normalized.is_file():
        raise PersonEmailExperimentError("normalized dataset is missing")
    if not predictions.is_file():
        raise PersonEmailExperimentError("persisted predictions are missing")
    if is_within_repository(report):
        raise PersonEmailExperimentError("report path must be outside the repository")
    if report.exists():
        raise PersonEmailExperimentError("report already exists")
    if report.resolve() in {normalized.resolve(), predictions.resolve()}:
        raise PersonEmailExperimentError("report path must not replace an input")


def _table(
    account: Mapping[str, int], person_delta: Mapping[str, int | float]
) -> dict[str, int | float]:
    suppressed = account["eligible_strict_suppressions"]
    exact = account["exact_person_gt_matches"]
    clean_fp = account["no_gt_overlap"]
    other = suppressed - exact - clean_fp
    if other < 0:
        raise PersonEmailExperimentError("suppression accounting partition is inconsistent")
    return {
        "suppressed_person": suppressed,
        "suppressed_tp": exact,
        "suppressed_fp": clean_fp,
        "suppressed_other_or_overlap": other,
        "strict_scorer_fp": suppressed - exact,
        "net_tp": person_delta["tp"],
        "net_fp": person_delta["fp"],
        "net_fn": person_delta["fn"],
        "net_f1": person_delta["f1"],
    }


def run_condition(
    dataset: str,
    normalized_path: Path,
    predictions_path: Path,
    threshold: float,
    label_map: GlinerLabelMap | None = None,
) -> dict[str, Any]:
    """Score one dataset/threshold pair. Inputs are hashed and only opened for reading."""
    if dataset not in EXPECTED_NORMALIZED_SHA256:
        raise PersonEmailExperimentError("dataset is outside the pinned OOD trio")
    if threshold not in ALLOWED_THRESHOLDS:
        raise PersonEmailExperimentError("threshold must stay at 0.3 or 0.7")
    label_map = load_label_map() if label_map is None else label_map
    normalized_hash = sha256_file(normalized_path)
    predictions_hash = sha256_file(predictions_path)
    if normalized_hash != EXPECTED_NORMALIZED_SHA256[dataset]:
        raise PersonEmailExperimentError("normalized dataset hash does not match the pinned digest")
    _require_max_width(read_header(predictions_path))

    states = {
        name: EvaluationState()
        for name in ("baseline_full", "candidate_full", "baseline_supported", "candidate_supported")
    }
    account = _empty_account()
    pending: list[
        tuple[int, tuple[ExactMatch, ...], tuple[ExactMatch, ...], tuple[ExactMatch, ...]]
    ] = []
    file_predictions = scored_predictions = removed_predictions = 0
    started = time.perf_counter()
    records = iter_records(predictions_path)
    with normalized_path.open("r", encoding="utf-8", newline="\n") as handle:
        for index, line in enumerate(handle):
            record = next(records, None)
            if record is None or record.example_index != index:
                raise PersonEmailExperimentError(
                    "prediction records do not align with normalized line order"
                )
            try:
                text, truth = validate_normalized_example(json.loads(line))
            except (ValueError, json.JSONDecodeError) as error:
                raise PersonEmailExperimentError("normalized record is invalid") from error
            file_predictions += len(record.predictions)
            try:
                baseline, classified = scoring_views(text, record.predictions, label_map, threshold)
            except PersonEmailExperimentError:
                raise
            except (KeyError, TypeError, ValueError) as error:
                raise PersonEmailExperimentError("prediction record cannot be scored") from error
            dropped = removal_is_subsequence(baseline, classified.kept)
            if dropped is None or dropped != len(classified.strict):
                raise PersonEmailExperimentError(
                    "scoring view changed a prediction it did not remove"
                )
            scored_predictions += len(baseline)
            removed_predictions += dropped
            truth_tuple = tuple(truth)
            account["original_person_predictions"] += sum(
                _gateway_type(item, label_map) == "PERSON" for item in baseline
            )
            account["loose_overlap_candidates"] += len(classified.loose)
            account["loose_only_candidates"] += len(classified.loose_only)
            _add_counts(account, account_suppressed(classified.strict, truth_tuple, label_map))
            pending.append(
                (
                    index,
                    truth_tuple,
                    _gateway_entities(baseline, label_map),
                    _gateway_entities(classified.kept, label_map),
                )
            )
            if len(pending) == SCORE_BATCH_SIZE:
                _flush(states, pending, False)
            if index and index % 5000 == 0:
                print(
                    f"progress dataset={dataset} threshold={threshold} records={index} "
                    f"elapsed_s={time.perf_counter() - started:.1f}",
                    file=sys.stderr,
                    flush=True,
                )
    if next(records, None) is not None:
        raise PersonEmailExperimentError("prediction file has records beyond normalized data")
    _flush(states, pending, True)
    if (
        sha256_file(normalized_path) != normalized_hash
        or sha256_file(predictions_path) != predictions_hash
    ):
        raise PersonEmailExperimentError("input hash changed while the experiment was reading")

    person_base = _entity_rates(states["baseline_full"], "PERSON")
    person_new = _entity_rates(states["candidate_full"], "PERSON")
    person_delta = _delta(person_base, person_new)
    if _entity_rates(states["baseline_supported"], "PERSON") != person_base:
        raise PersonEmailExperimentError("PERSON baseline disagreed across scoring views")
    if _entity_rates(states["candidate_supported"], "PERSON") != person_new:
        raise PersonEmailExperimentError("PERSON suppression disagreed across scoring views")
    full_base, full_new = (
        _rates(**states["baseline_full"].aggregate()),
        _rates(**states["candidate_full"].aggregate()),
    )
    supported_base = _rates(**states["baseline_supported"].aggregate())
    supported_new = _rates(**states["candidate_supported"].aggregate())
    return {
        "experiment": EXPERIMENT_ID,
        "privacy": "aggregate metrics only; no source text, offsets, or ids",
        "dataset": dataset,
        "threshold": threshold,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "input": {
            "normalized_sha256": normalized_hash,
            "predictions_sha256": predictions_hash,
            "normalized_sha256_after": normalized_hash,
            "predictions_sha256_after": predictions_hash,
        },
        "invariants": {
            "original_total_predictions": file_predictions,
            "scored_predictions": scored_predictions,
            "suppressed_predictions": removed_predictions,
            "added_predictions": 0,
            "modified_predictions": 0,
            "deleted_predictions": removed_predictions,
        },
        "containment": {
            "original_person_predictions": account["original_person_predictions"],
            "strict_suppressions": account["eligible_strict_suppressions"],
            "loose_overlap_candidates": account["loose_overlap_candidates"],
            "loose_only_not_strict": account["loose_only_candidates"],
        },
        "accounting": {
            key: account[key]
            for key in (
                "exact_person_gt_matches",
                "person_gt_mismatches",
                "overlapping_email_gt",
                "overlapping_unmapped_gt",
                "overlapping_unsupported_gt",
                "overlapping_other_supported_gt",
                "no_gt_overlap",
            )
        },
        "person": {
            "baseline": person_base,
            "strict_suppression": person_new,
            "delta": person_delta,
        },
        "full_strict": {
            "baseline": full_base,
            "strict_suppression": full_new,
            "delta": _delta(full_base, full_new),
        },
        "gateway_supported_only": {
            "baseline": supported_base,
            "strict_suppression": supported_new,
            "delta": _delta(supported_base, supported_new),
        },
        "tp_loss_table": _table(account, person_delta),
        "definitions": {
            "full_strict": "all entities after the DATE/TIME adapter, including UNMAPPED",
            "gateway_supported_only": "Phase 1 entity types only",
            "suppressed_tp": "removed PERSON predictions that exactly match PERSON ground truth",
            "suppressed_fp": "removed PERSON predictions with no ground-truth overlap",
            "suppressed_other_or_overlap": "removed non-exact predictions overlapping ground truth",
            "strict_scorer_fp": "removed PERSON predictions that are not exact PERSON matches",
            "strict_containment": "email_start < person_start < person_end < email_end",
            "loose_diagnostic": "half-open email overlap; not used for suppression",
            "email_validity": "frozen recognize_email_addresses only; no DNS or network",
            "eligibility": "source text and model prediction only",
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score strict PERSON-inside-email suppression on persisted GLiNER predictions."
    )
    parser.add_argument("--dataset", choices=sorted(EXPECTED_NORMALIZED_SHA256), required=True)
    parser.add_argument("--normalized", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--threshold", choices=("0.3", "0.7"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _validate_paths(args.normalized, args.predictions, args.report)
        report = run_condition(
            args.dataset, args.normalized, args.predictions, float(args.threshold)
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, PersonEmailExperimentError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
