"""Offline, aggregate-only diagnostics for existing GLiNER OOD predictions.

Run once per normalized dataset/prediction threshold pair. No model is loaded.
Error groups count unmatched entities with multiplicity. Boundary and confusion
diagnostics count overlapping GT/prediction pairs; boundary flags are not exclusive.
Full strict metrics retain every annotation; a separate gateway-supported-only
view filters evaluation types on both sides without changing the scorer.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator
from itertools import zip_longest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.batch_evaluation import Batch, score_batch
from scripts.evaluation_checkpoint import EvaluationState, sha256_file
from scripts.gliner_evaluation import GlinerLabelMap, load_label_map, to_gateway
from scripts.gliner_predictions import iter_records, read_header
from scripts.nemotron_pii import is_within_repository
from scripts.presidio_evaluation import (
    SUPPORTED_TYPES,
    ExactMatch,
    adapter_category,
    metric_rows,
    validate_normalized_example,
)

BOUNDARY_PATTERNS = (
    "prediction_starts_before_gt",
    "prediction_starts_after_gt",
    "prediction_ends_before_gt",
    "prediction_ends_after_gt",
    "prediction_contains_gt",
    "gt_contains_prediction",
)


def _normalized_records(path: Path) -> Iterator[dict[str, Any]]:
    # Only LF is a record separator, not U+2028/U+2029/U+0085 within JSON text.
    with path.open("r", encoding="utf-8", newline="\n") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                raise ValueError("normalized record is not valid JSON") from None
            if not isinstance(row, dict):
                raise ValueError("normalized record must be an object")
            yield row


def _overlaps(left: ExactMatch, right: ExactMatch) -> bool:
    return left.start < right.end and right.start < left.end


def _boundaries(gt: ExactMatch, prediction: ExactMatch) -> list[str]:
    conditions = (
        prediction.start < gt.start,
        prediction.start > gt.start,
        prediction.end < gt.end,
        prediction.end > gt.end,
        prediction.start <= gt.start and prediction.end >= gt.end,
        gt.start <= prediction.start and gt.end >= prediction.end,
    )
    return [
        name for name, applies in zip(BOUNDARY_PATTERNS, conditions, strict=True) if applies
    ]


def _validate_native(row: dict[str, Any], text: str, label_map: GlinerLabelMap) -> None:
    native = row.get("native_entities")
    if not isinstance(native, list):
        raise ValueError("normalized record must contain native_entities")
    # Validate native offsets without changing or using them to repair gateway GT.
    entities = []
    for item in native:
        if not isinstance(item, dict) or item.get("native_label") not in (
            *label_map.labels, "UNMAPPED"
        ):
            raise ValueError("native ground-truth label is invalid")
        entities.append(
            {"type": item["native_label"], "start": item.get("start"), "end": item.get("end")}
        )
    validate_normalized_example({"text": text, "entities": entities})
    if Counter((item["start"], item["end"]) for item in entities) != Counter(
        (item["start"], item["end"]) for item in row["entities"]
    ):
        raise ValueError("native and gateway ground-truth spans differ")


def _strict_summary(state: EvaluationState) -> dict[str, int | float]:
    """Summarize an existing strict score using the shared metrics helper."""
    totals = state.aggregate()
    gt_count = totals["tp"] + totals["fn"]
    prediction_count = totals["tp"] + totals["fp"]
    overall = metric_rows(
        Counter({"overall": gt_count}),
        Counter({"overall": prediction_count}),
        Counter({"overall": totals["tp"]}),
    )["overall"]
    return {
        **totals,
        "precision": overall["precision"],
        "recall": overall["recall"],
        "f1": overall["f1"],
        "gt_entity_count": gt_count,
        "prediction_count": prediction_count,
    }


def analyze(normalized: Path, predictions: Path) -> dict[str, Any]:
    """Validate and score an existing pair; return only allowlisted aggregates."""
    label_map = load_label_map()
    identity = read_header(predictions)
    normalized_hash = sha256_file(normalized)
    if identity.get("normalized_sha256") != normalized_hash:
        raise ValueError("prediction identity does not match normalized dataset")
    model = identity.get("model")
    if not isinstance(model, dict) or any(
        model.get(key) != value
        for key, value in (
            ("model_id", label_map.model_id), ("model_revision", label_map.model_revision)
        )
    ):
        raise ValueError("prediction model differs from pinned label map")
    threshold = identity.get("threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, int | float)
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("prediction threshold is invalid")

    # Unknown source types are scored unchanged but never echoed into reports.
    allowed_types = SUPPORTED_TYPES | set(label_map.mapping.values())

    def safe_type(entity_type: str) -> str:
        return entity_type if entity_type in allowed_types else "UNSUPPORTED_ENTITY"

    state = EvaluationState()
    supported_state = EvaluationState()
    gt_entity_count = supported_gt_count = unsupported_gt_count = 0
    gt_adapter_categories: Counter[str] = Counter()
    fn_groups: Counter[str] = Counter()
    fp_groups: Counter[str] = Counter()
    fn_types: dict[str, Counter[str]] = defaultdict(Counter)
    fp_types: dict[str, Counter[str]] = defaultdict(Counter)
    boundaries: Counter[str] = Counter(dict.fromkeys(BOUNDARY_PATTERNS, 0))
    boundary_types: dict[str, Counter[str]] = defaultdict(Counter)
    confusions: dict[str, Counter[str]] = defaultdict(Counter)
    boundary_pairs = confusion_pairs = unmapped = count = 0
    removed_predictions = 0

    for index, (row, record) in enumerate(
        zip_longest(_normalized_records(normalized), iter_records(predictions))
    ):
        if row is None or record is None:
            raise ValueError("prediction record count differs from normalized dataset")
        # OOD prediction indices are dense file positions, NOT original source IDs
        # (normalization can explicitly exclude whole invalid source examples).
        if record.example_index != index:
            raise ValueError("prediction record indices are not in dataset order")
        text, truth = validate_normalized_example(row)
        _validate_native(row, text, label_map)
        for item in record.predictions:
            if (
                item.label not in label_map.mapping
                or item.end > len(text)
                or not math.isfinite(item.score)
            ):
                raise ValueError("prediction label, offsets, or score are invalid")
        mapped = to_gateway(record.predictions, label_map)
        predicted = mapped.kept
        removed_predictions += sum(mapped.removed_by_label.values())
        score_batch(
            state, Batch([(index, b"")], 0, False),
            [(tuple((e.entity_type, e.start, e.end) for e in truth),
              tuple((e.entity_type, e.start, e.end) for e in predicted))],
        )
        # Both lists already use evaluation_type() through the existing validator
        # and to_gateway(). Do not duplicate DATE/TIME mapping or alter annotations.
        supported_truth = [e for e in truth if e.entity_type in SUPPORTED_TYPES]
        supported_predictions = [e for e in predicted if e.entity_type in SUPPORTED_TYPES]
        score_batch(
            supported_state, Batch([(index, b"")], 0, False),
            [(tuple((e.entity_type, e.start, e.end) for e in supported_truth),
              tuple((e.entity_type, e.start, e.end) for e in supported_predictions))],
        )
        gt_entity_count += len(truth)
        supported_gt_count += len(supported_truth)
        unsupported_gt_count += sum(
            e.entity_type not in SUPPORTED_TYPES and e.entity_type != "UNMAPPED"
            for e in truth
        )
        # Preserve the existing adapter's distinction between raw DATE/TIME and
        # directly supported DATE_TIME. Only fixed category names are reported.
        gt_adapter_categories.update(adapter_category(e["type"]) for e in row["entities"])
        truth_counts, prediction_counts = Counter(truth), Counter(predicted)
        matches = truth_counts & prediction_counts
        missed, spurious = truth_counts - matches, prediction_counts - matches
        unmapped += sum(n for e, n in truth_counts.items() if e.entity_type == "UNMAPPED")
        for entity, multiplicity in missed.items():
            category = adapter_category(entity.entity_type)
            same_overlap = any(
                other != entity and other.entity_type == entity.entity_type
                and _overlaps(entity, other) for other in prediction_counts
            )
            group = category if category in {
                "unmapped_source_label", "unsupported_entity"
            } else ("span_mismatch" if same_overlap else "detector_miss")
            fn_groups[group] += multiplicity
            fn_types[safe_type(entity.entity_type)][group] += multiplicity
        for entity, multiplicity in spurious.items():
            same_overlap = any(
                other.entity_type == entity.entity_type and _overlaps(entity, other)
                for other in truth_counts
            )
            group = "span_mismatch" if same_overlap else "detector_recognizer"
            fp_groups[group] += multiplicity
            fp_types[safe_type(entity.entity_type)][group] += multiplicity
        for gt, gt_count in truth_counts.items():
            for prediction, prediction_count in prediction_counts.items():
                if not _overlaps(gt, prediction):
                    continue
                pairs = gt_count * prediction_count
                if gt.entity_type != prediction.entity_type:
                    # UNMAPPED/unsupported GT is a mapping issue, not a known-type
                    # confusion. Confusions are supplementary, not extra FP/FN.
                    if adapter_category(gt.entity_type) not in {
                        "unmapped_source_label", "unsupported_entity"
                    }:
                        confusion_pairs += pairs
                        confusions[safe_type(gt.entity_type)][
                            safe_type(prediction.entity_type)
                        ] += pairs
                elif gt != prediction and (gt in missed or prediction in spurious):
                    boundary_pairs += pairs
                    flags = _boundaries(gt, prediction)
                    for flag in flags:
                        boundaries[flag] += pairs
                        boundary_types[safe_type(gt.entity_type)][flag] += pairs
        count += 1

    # Use the existing metrics helper; explicit TP/FP/FN names are report aliases.
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    for source, target in (
        (state.true_positives, tp), (state.false_positives, fp),
        (state.false_negatives, fn),
    ):
        for entity_type, value in source.items():
            target[safe_type(entity_type)] += value
    support = Counter({kind: tp[kind] + fn[kind] for kind in tp.keys() | fn.keys()})
    predicted_support = Counter({kind: tp[kind] + fp[kind] for kind in tp.keys() | fp.keys()})
    metrics = metric_rows(support, predicted_support, tp)
    for kind, values in metrics.items():
        values.update(tp=tp[kind], fp=fp[kind], fn=fn[kind])

    def counts_by_type(values: dict[str, Counter[str]]) -> dict[str, dict[str, int]]:
        return {kind: dict(sorted(groups.items())) for kind, groups in sorted(values.items())}

    return {
        "privacy": "aggregate-only; no source text, entity values, snippets, or offsets",
        "normalized_sha256": normalized_hash,
        "predictions_sha256": sha256_file(predictions),
        "label_map": label_map.as_identity(),
        "threshold": threshold,
        "record_count": count,
        "overall": state.aggregate(),
        "full_strict": _strict_summary(state),
        "gateway_supported_only_strict": _strict_summary(supported_state),
        "ontology_coverage": {
            "gt_entity_count": gt_entity_count,
            "gateway_supported_gt_count": supported_gt_count,
            "gateway_supported_gt_pct": (
                100 * supported_gt_count / gt_entity_count if gt_entity_count else 0.0
            ),
            "unmapped_gt_count": unmapped,
            "unsupported_but_mapped_gt_count": unsupported_gt_count,
            "gt_by_adapter_category": dict(sorted(gt_adapter_categories.items())),
        },
        "per_entity_type": metrics,
        "unmapped_ground_truth": unmapped,
        "unmapped_predictions_removed": removed_predictions,
        "false_negatives": {
            "by_failure_category": dict(sorted(fn_groups.items())),
            "by_entity_type": counts_by_type(fn_types),
        },
        "false_positives": {
            "by_failure_category": dict(sorted(fp_groups.items())),
            "by_entity_type": counts_by_type(fp_types),
        },
        "span_mismatches": {
            "counting": "non-exact same-type overlapping pairs involving an unmatched entity",
            "overlap_pairs": boundary_pairs,
            "boundary_patterns": dict(boundaries),
            "by_entity_type": counts_by_type(boundary_types),
        },
        "likely_type_confusions": {
            "counting": "different-type overlapping pairs; excludes UNMAPPED/unsupported GT",
            "overlap_pairs": confusion_pairs,
            "by_ground_truth_and_predicted_type": counts_by_type(confusions),
        },
    }


def run(normalized: Path, predictions: Path, output: Path) -> None:
    if is_within_repository(output):
        raise ValueError("report output must be outside repository")
    if output.resolve() in {normalized.resolve(), predictions.resolve()} or output.exists():
        raise ValueError("report output already exists or aliases an input")
    report = analyze(normalized, predictions)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        run(args.normalized, args.predictions, args.output)
    except (OSError, ValueError, KeyError, TypeError):
        # Third-party parser/scorer exceptions may contain source-controlled data.
        # Never echo exception strings or paths from untrusted benchmark inputs.
        print(
            "error: OOD analysis failed; check input integrity and output location",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())