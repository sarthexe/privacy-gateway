"""Aggregate-only offline phone experiment; never loads a model or writes predictions."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any

import phonenumbers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_gliner_ood_errors import (
    _normalized_records,
    _strict_summary,
    _validate_native,
)
from scripts.batch_evaluation import Batch, score_batch
from scripts.evaluation_checkpoint import EvaluationState, sha256_file
from scripts.gliner_evaluation import GlinerLabelMap, RawPrediction, load_label_map, to_gateway
from scripts.gliner_predictions import iter_records, read_header
from scripts.nemotron_pii import is_within_repository
from scripts.presidio_evaluation import SUPPORTED_TYPES, validate_normalized_example

VARIANT = "gliner_plus_validated_phone_candidates"
PHONE = "PHONE_NUMBER"
BOUNDARIES = (
    "exact_match",
    "candidate_contained_by_gt",
    "gt_contained_by_candidate",
    "partial_overlap",
    "no_phone_gt_overlap",
)
# Broad extraction intentionally includes invalid numeric strings. No GT is consulted.
# Horizontal separators only: a candidate cannot consume the next line.
PATTERN = re.compile(
    r"(?<![\w+])(?:\+[\t ]*)?(?:\([\t ]*)?\d"
    r"(?:[\d\t ()./-]*\d)?\)?"
    r"(?:[\t ]*(?:extension|ext\.?|x|#)[\t ]*\d+)?(?!\w)",
    re.IGNORECASE,
)


class PhoneExperimentError(ValueError):
    """Fixed, source-independent failure messages."""


@dataclass(frozen=True)
class RegionPolicy:
    region: str | None = None

    def __post_init__(self) -> None:
        if self.region is not None and (
            not isinstance(self.region, str) or self.region not in phonenumbers.SUPPORTED_REGIONS
        ):
            raise PhoneExperimentError("region must be an explicit uppercase supported region")


@dataclass(frozen=True, order=True)
class Span:
    start: int
    end: int


@dataclass(frozen=True)
class CandidateResult:
    raw: tuple[Span, ...]
    validated: tuple[Span, ...]
    rejections: Counter[str]


@dataclass(frozen=True)
class MergeResult:
    predictions: tuple[RawPrediction, ...]
    added: tuple[Span, ...]
    duplicates_skipped: int


def deduplicate_candidates(candidates: Iterable[Span], text_length: int) -> tuple[Span, ...]:
    spans = set()
    for span in candidates:
        if (
            not isinstance(span, Span)
            or isinstance(span.start, bool)
            or isinstance(span.end, bool)
            or not isinstance(span.start, int)
            or not isinstance(span.end, int)
            or not 0 <= span.start < span.end <= text_length
        ):
            raise PhoneExperimentError("candidate offsets are invalid")
        spans.add(span)
    return tuple(sorted(spans))


def phone_candidates(text: str, policy: RegionPolicy) -> CandidateResult:
    """Retain raw and validated original spans in memory only."""
    if not isinstance(text, str) or not isinstance(policy, RegionPolicy):
        raise PhoneExperimentError("candidate input or policy is invalid")
    raw = deduplicate_candidates(
        (
            Span(match.start(), match.end())
            for match in PATTERN.finditer(text)
            if sum(char.isdecimal() for char in match.group()) >= 7
        ),
        len(text),
    )
    validated = []
    rejections: Counter[str] = Counter()
    for span in raw:
        value = text[span.start:span.end]
        if len(value) > 128:
            rejections["candidate_too_long"] += 1
            continue
        international = value.startswith("+")
        if not international and policy.region is None:
            rejections["explicit_region_required"] += 1
            continue
        try:
            # Only the parser normalizes. Source text and candidate offsets never change.
            parsed = phonenumbers.parse(value, None if international else policy.region)
        except phonenumbers.NumberParseException:
            rejections["parse_error"] += 1
            continue
        if not phonenumbers.is_valid_number(parsed):
            rejections["invalid_number"] += 1
            continue
        validated.append(span)
    return CandidateResult(raw, tuple(validated), rejections)


def merge_phone_candidates(
    text: str,
    predictions: Sequence[RawPrediction],
    candidates: Iterable[Span],
    label_map: GlinerLabelMap,
) -> MergeResult:
    """Append only; never inspect GT, suppress conflicts, or deduplicate the baseline."""
    if not isinstance(text, str):
        raise PhoneExperimentError("merge text must be a string")
    original = tuple(predictions)
    for item in original:
        if (
            not isinstance(item, RawPrediction)
            or item.label not in label_map.mapping
            or isinstance(item.start, bool)
            or isinstance(item.end, bool)
            or not isinstance(item.start, int)
            or not isinstance(item.end, int)
            or not 0 <= item.start < item.end <= len(text)
            or isinstance(item.score, bool)
            or not isinstance(item.score, int | float)
            or not math.isfinite(item.score)
        ):
            raise PhoneExperimentError("baseline prediction is invalid")
    if label_map.mapping.get("phone_number") != PHONE:
        raise PhoneExperimentError("label map has no canonical phone label")
    existing = {
        Span(p.start, p.end) for p in original if label_map.mapping[p.label] == PHONE
    }
    unique = deduplicate_candidates(candidates, len(text))
    added = tuple(span for span in unique if span not in existing)
    # 1.0 is an acceptance marker, NOT model confidence; no artifact is written.
    hybrid = original + tuple(
        RawPrediction("phone_number", span.start, span.end, 1.0) for span in added
    )
    old_nonphone = tuple(p for p in original if label_map.mapping[p.label] != PHONE)
    new_nonphone = tuple(p for p in hybrid if label_map.mapping[p.label] != PHONE)
    if hybrid[:len(original)] != original or old_nonphone != new_nonphone:
        raise PhoneExperimentError("append-only non-phone invariant failed")
    return MergeResult(hybrid, added, len(unique) - len(added))


def overlaps(left: Span, right: Span) -> bool:
    return left.start < right.end and right.start < left.end


def classify_candidate(candidate: Span, truth: Iterable[Span]) -> str:
    """Exclusive candidate classification, with documented priority for multiple GT."""
    relations = set()
    for gt in truth:
        if candidate == gt:
            relations.add("exact_match")
        elif gt.start <= candidate.start and candidate.end <= gt.end:
            relations.add("candidate_contained_by_gt")
        elif candidate.start <= gt.start and gt.end <= candidate.end:
            relations.add("gt_contained_by_candidate")
        elif overlaps(candidate, gt):
            relations.add("partial_overlap")
    return next((key for key in BOUNDARIES[:-1] if key in relations), BOUNDARIES[-1])


def _phone_metrics(state: EvaluationState) -> dict[str, Any]:
    phone = EvaluationState()
    phone.true_positives[PHONE] = state.true_positives[PHONE]
    phone.false_positives[PHONE] = state.false_positives[PHONE]
    phone.false_negatives[PHONE] = state.false_negatives[PHONE]
    return _strict_summary(phone)


def _delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    return {
        key: after[key] - before[key]
        for key in ("tp", "fp", "fn", "precision", "recall", "f1")
    }


def _flush(condition: dict[str, Any], reached_eof: bool = False) -> None:
    pending = condition["pending"]
    if not pending:
        return
    batch = Batch([(row[0], b"") for row in pending], 0, reached_eof)
    for supported in (False, True):
        for hybrid in (False, True):
            pairs = []
            for _, truth, old, new in pending:
                predicted = new if hybrid else old
                pairs.append(tuple(
                    tuple(
                        (e.entity_type, e.start, e.end) for e in entities
                        if not supported or e.entity_type in SUPPORTED_TYPES
                    )
                    for entities in (truth, predicted)
                ))
            score_batch(condition["states"][supported, hybrid], batch, pairs)
    pending.clear()


def run_experiment(
    normalized: Path,
    prediction_paths: Sequence[Path],
    policy: RegionPolicy,
    progress_every: int = 5000,
) -> dict[str, Any]:
    """Evaluate both thresholds in one dataset stream; export only allowlisted aggregates."""
    started = time.perf_counter()
    label_map = load_label_map()
    normalized_hash = sha256_file(normalized)
    paths = tuple(prediction_paths)
    if len(paths) != 2 or len({path.resolve() for path in paths}) != 2:
        raise PhoneExperimentError("supply exactly two distinct existing prediction files")
    conditions = []
    for path in paths:
        identity = read_header(path)
        model = identity.get("model")
        threshold = identity.get("threshold")
        if (
            identity.get("normalized_sha256") != normalized_hash
            or not isinstance(model, dict)
            or model.get("model_id") != label_map.model_id
            or model.get("model_revision") != label_map.model_revision
            or model.get("max_width") != 24
            or isinstance(threshold, bool)
            or threshold not in (0.3, 0.7)
        ):
            raise PhoneExperimentError("prediction identity differs from required pinned inputs")
        conditions.append({
            "path": path, "hash": sha256_file(path), "threshold": threshold,
            "identity": identity,
            "states": {
                (supported, hybrid): EvaluationState()
                for supported in (False, True) for hybrid in (False, True)
            },
            "pending": [], "regression": Counter(), "added_boundaries": Counter(),
            "recovered_fn_evidence": Counter(), "added_fp_coverage": Counter(),
        })
    if {c["threshold"] for c in conditions} != {0.3, 0.7}:
        raise PhoneExperimentError("both existing thresholds 0.3 and 0.7 are required")
    if any(
        {k: v for k, v in c["identity"].items() if k != "threshold"}
        != {k: v for k, v in conditions[0]["identity"].items() if k != "threshold"}
        for c in conditions[1:]
    ):
        raise PhoneExperimentError("threshold input identities otherwise differ")
    raw_counts = Counter(dict.fromkeys(BOUNDARIES, 0))
    validated_counts = Counter(dict.fromkeys(BOUNDARIES, 0))
    rejections: Counter[str] = Counter()
    examples = raw_total = validated_total = 0
    streams = [_normalized_records(normalized), *(iter_records(c["path"]) for c in conditions)]
    for index, bundle in enumerate(zip_longest(*streams)):
        row, *records = bundle
        if row is None or any(record is None for record in records):
            raise PhoneExperimentError("dataset and prediction record counts differ")
        if any(record.example_index != index for record in records):
            raise PhoneExperimentError("prediction record indices are not aligned")
        text, truth = validate_normalized_example(row)
        _validate_native(row, text, label_map)
        candidates = phone_candidates(text, policy)
        phone_truth = [Span(e.start, e.end) for e in truth if e.entity_type == PHONE]
        raw_total += len(candidates.raw)
        validated_total += len(candidates.validated)
        rejections.update(candidates.rejections)
        raw_counts.update(classify_candidate(span, phone_truth) for span in candidates.raw)
        validated_counts.update(
            classify_candidate(span, phone_truth) for span in candidates.validated
        )
        for condition, record in zip(conditions, records, strict=True):
            merged = merge_phone_candidates(
                text, record.predictions, candidates.validated, label_map
            )
            old = tuple(to_gateway(record.predictions, label_map).kept)
            new = tuple(to_gateway(merged.predictions, label_map).kept)
            condition["pending"].append((index, tuple(truth), old, new))
            regression = condition["regression"]
            regression["baseline_predictions"] += len(record.predictions)
            regression["hybrid_predictions"] += len(merged.predictions)
            regression["added_predictions"] += len(merged.added)
            regression["exact_duplicate_candidates_skipped"] += merged.duplicates_skipped
            regression["removed_predictions"] += 0
            regression["changed_non_phone_predictions"] += 0
            regression["records_verified"] += 1
            for span in merged.added:
                boundary = classify_candidate(span, phone_truth)
                condition["added_boundaries"][boundary] += 1
                if boundary == "exact_match":
                    if any(
                        e.entity_type == PHONE and overlaps(span, Span(e.start, e.end)) for e in old
                    ):
                        evidence = "existing_nonexact_phone_overlap"
                    elif any(
                        overlaps(span, Span(p.start, p.end)) for p in record.predictions
                    ):
                        evidence = "other_native_prediction_overlap"
                    else:
                        evidence = "no_baseline_native_prediction_overlap"
                    condition["recovered_fn_evidence"][evidence] += 1
                else:
                    overlapping = [e for e in truth if overlaps(span, Span(e.start, e.end))]
                    if boundary != "no_phone_gt_overlap":
                        coverage = "nonexact_phone_gt_overlap"
                    elif any(e.entity_type in SUPPORTED_TYPES for e in overlapping):
                        coverage = "other_supported_gt_overlap"
                    elif any(e.entity_type == "UNMAPPED" for e in overlapping):
                        coverage = "unmapped_gt_overlap"
                    elif overlapping:
                        coverage = "other_unsupported_gt_overlap"
                    else:
                        coverage = "no_annotated_gt_overlap"
                    condition["added_fp_coverage"][coverage] += 1
            if len(condition["pending"]) >= 64:
                _flush(condition)
        examples += 1
        if progress_every > 0 and examples % progress_every == 0:
            print(json.dumps({
                "event": "progress", "examples": examples,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }), flush=True)
    if not examples:
        raise PhoneExperimentError("normalized dataset is empty")
    if sha256_file(normalized) != normalized_hash:
        raise PhoneExperimentError("normalized input changed during evaluation")
    outputs = []
    for condition in sorted(conditions, key=lambda c: c["threshold"]):
        _flush(condition, True)
        if sha256_file(condition["path"]) != condition["hash"]:
            raise PhoneExperimentError("prediction input changed during evaluation")
        regression = dict(condition["regression"])
        if (
            regression["removed_predictions"] != 0
            or regression["changed_non_phone_predictions"] != 0
            or regression["hybrid_predictions"] - regression["baseline_predictions"]
            != regression["added_predictions"]
        ):
            raise PhoneExperimentError("regression totals failed")
        output = {
            "threshold": condition["threshold"],
            "input_predictions_sha256": condition["hash"],
            "regression": regression,
            "added_candidate_boundaries": dict(condition["added_boundaries"]),
            "recovered_fn_evidence": dict(condition["recovered_fn_evidence"]),
            "added_fp_annotation_coverage": dict(condition["added_fp_coverage"]),
        }
        for supported, name in ((False, "full_strict"), (True, "gateway_supported_only_strict")):
            before = _strict_summary(condition["states"][supported, False])
            after = _strict_summary(condition["states"][supported, True])
            output[name] = {"baseline": before, "hybrid": after, "delta": _delta(before, after)}
        old_phone = _phone_metrics(condition["states"][False, False])
        new_phone = _phone_metrics(condition["states"][False, True])
        delta = _delta(old_phone, new_phone)
        output["phone_number"] = {
            "baseline": old_phone, "hybrid": new_phone, "delta": delta,
        }
        if (
            delta["tp"] != condition["added_boundaries"]["exact_match"]
            or delta["fn"] != -delta["tp"]
            or delta["tp"] + delta["fp"] != regression["added_predictions"]
            or sum(condition["added_fp_coverage"].values()) != delta["fp"]
        ):
            raise PhoneExperimentError("candidate and strict-score accounting differ")
        for name in ("full_strict", "gateway_supported_only_strict"):
            if any(output[name]["delta"][key] != delta[key] for key in ("tp", "fp", "fn")):
                raise PhoneExperimentError("non-phone strict metrics changed")
        outputs.append(output)
    if sum(raw_counts.values()) != raw_total or (
        validated_total + sum(rejections.values()) != raw_total
    ):
        raise PhoneExperimentError("candidate accounting failed")
    return {
        "experiment": VARIANT,
        "parser": {"library": "phonenumbers", "version": phonenumbers.__version__},
        "policy": {
            "explicit_region": policy.region,
            "international_requires_leading_plus": True,
            "validation": "is_valid_number",
            "raw_minimum_decimal_digits": 7,
            "maximum_validation_span_characters": 128,
            "boundaries": "original spans; no trimming, splitting, or expansion",
            "candidate_boundary_priority": list(BOUNDARIES),
            "merge": "append only; exact PHONE_NUMBER duplicates skipped, including fax aliases",
            "synthetic_score": (
                "1.0 acceptance marker; not model confidence; never threshold-filtered"
            ),
        },
        "privacy": "aggregate counts only; no source text, values, example IDs, or raw offsets",
        "normalized_sha256": normalized_hash,
        "examples": examples,
        "candidates": {
            "raw": raw_total, "validated": validated_total,
            "rejection_reasons": dict(rejections),
            "raw_boundaries": dict(raw_counts),
            "validated_boundaries": dict(validated_counts),
        },
        "conditions": outputs,
        "elapsed_seconds": time.perf_counter() - started,
        "inputs_unchanged": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, action="append", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--region", default=None, help="Explicit uppercase region; default: none.")
    parser.add_argument("--progress-every", type=int, default=5000)
    args = parser.parse_args(argv)
    if args.report.exists() or is_within_repository(args.report):
        print("error: report must be a new path outside the repository", file=sys.stderr)
        return 2
    try:
        report = run_experiment(
            args.normalized, args.predictions, RegionPolicy(args.region), args.progress_every
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except (ValueError, OSError, TypeError, AttributeError):
        print("error: experiment failed input, policy, or invariant validation", file=sys.stderr)
        return 2
    print(json.dumps({
        "event": "complete", "examples": report["examples"],
        "elapsed_seconds": round(report["elapsed_seconds"], 3),
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())