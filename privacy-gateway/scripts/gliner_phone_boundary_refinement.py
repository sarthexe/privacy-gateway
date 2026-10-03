"""Frozen-rule offline Variant B: parser-identity-preserving phone boundaries only."""

from __future__ import annotations

import argparse
import json
import string
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any

import phonenumbers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.batch_evaluation import Batch, score_batch
from scripts.evaluation_checkpoint import EvaluationState, sha256_file
from scripts.gliner_evaluation import load_label_map, to_gateway
from scripts.gliner_phone_hybrid_experiment import (
    BOUNDARIES,
    PHONE,
    PhoneExperimentError,
    RegionPolicy,
    Span,
    _delta,
    _normalized_records,
    _phone_metrics,
    _strict_summary,
    _validate_native,
    classify_candidate,
    deduplicate_candidates,
    merge_phone_candidates,
    phone_candidates,
)
from scripts.gliner_predictions import iter_records, read_header
from scripts.nemotron_pii import is_within_repository
from scripts.presidio_evaluation import SUPPORTED_TYPES, validate_normalized_example

VARIANT = "gliner_phone_parser_identity_boundary_refinement"
VARIANTS = ("baseline", "variant_a", "variant_b")
# Plus is a calling-code indicator, not removable formatting.
EDGE_FORMATTING = frozenset(string.punctuation.replace("+", "") + " \t")


@dataclass(frozen=True)
class Refinement:
    span: Span | None
    reason: str


def _identity(number: phonenumbers.PhoneNumber) -> tuple[Any, ...]:
    return (
        number.country_code, number.national_number, number.extension,
        bool(number.italian_leading_zero), number.number_of_leading_zeros,
    )


def refine_candidate(text: str, candidate: Span, policy: RegionPolicy) -> Refinement:
    """No GT parameter: only a subset of an A-validated source span may be selected."""
    if not isinstance(text, str) or not isinstance(policy, RegionPolicy):
        raise PhoneExperimentError("refinement input or policy is invalid")
    deduplicate_candidates([candidate], len(text))
    value = text[candidate.start:candidate.end]
    if not value.startswith("+") and policy.region is None:
        return Refinement(None, "explicit_region_required")
    region = None if value.startswith("+") else policy.region
    try:
        original = phonenumbers.parse(value, region)
        if not phonenumbers.is_valid_number(original):
            return Refinement(None, "invalid_input_number")
        matches = list(phonenumbers.PhoneNumberMatcher(
            value, region, leniency=phonenumbers.Leniency.VALID, max_tries=100
        ))
    except phonenumbers.NumberParseException:
        return Refinement(None, "parser_failure")
    eligible = set()
    for match in matches:
        selected = value[match.start:match.end]
        # Reapply the identical explicit-region policy to the proposed subspan.
        if not selected.startswith("+") and policy.region is None:
            continue
        outside = value[:match.start] + value[match.end:]
        if any(char not in EDGE_FORMATTING for char in outside):
            continue
        if _identity(match.number) != _identity(original):
            continue
        eligible.add(Span(candidate.start + match.start, candidate.start + match.end))
    if not eligible:
        return Refinement(None, "no_identity_preserving_boundary")
    shortest = min(span.end - span.start for span in eligible)
    tightest = sorted(span for span in eligible if span.end - span.start == shortest)
    if len(tightest) != 1:
        return Refinement(None, "ambiguous_tightest_boundary")
    selected = tightest[0]
    return Refinement(selected, "unchanged" if selected == candidate else "refined")


def _phone_counts(entities: Sequence[Any]) -> Counter[Span]:
    return Counter(Span(e.start, e.end) for e in entities if e.entity_type == PHONE)


def recovery_accounting(
    truth: Counter[Span], baseline: Counter[Span], a: Counter[Span], b: Counter[Span],
) -> dict[str, int]:
    """Multiset identity accounting, not subtraction of aggregate TP/FP totals."""
    recovered_a = (truth & a) - (truth & baseline)
    recovered_b = (truth & b) - (truth & baseline)
    fp_baseline = baseline - truth
    added_fp_a = (a - truth) - fp_baseline
    added_fp_b = (b - truth) - fp_baseline
    return {
        "variant_a_recovered_fn": sum(recovered_a.values()),
        "variant_b_retained_recoveries": sum((recovered_a & recovered_b).values()),
        "variant_b_new_recoveries": sum((recovered_b - recovered_a).values()),
        "variant_a_added_fp": sum(added_fp_a.values()),
        "variant_a_added_fp_remaining": sum((added_fp_a & added_fp_b).values()),
        "variant_a_added_fp_disappeared": sum((added_fp_a - added_fp_b).values()),
        "variant_b_new_fp": sum((added_fp_b - added_fp_a).values()),
        "variant_b_total_added_fp": sum(added_fp_b.values()),
    }


def _flush(condition: dict[str, Any], eof: bool = False) -> None:
    pending = condition["pending"]
    if not pending:
        return
    batch = Batch([(row[0], b"") for row in pending], 0, eof)
    for supported in (False, True):
        for variant_index, variant in enumerate(VARIANTS):
            pairs = [
                tuple(
                    tuple(
                        (e.entity_type, e.start, e.end) for e in entities
                        if not supported or e.entity_type in SUPPORTED_TYPES
                    )
                    for entities in (row[1], row[variant_index + 2])
                )
                for row in pending
            ]
            score_batch(condition["states"][supported, variant], batch, pairs)
    pending.clear()


def _same_metrics(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    if any(
        abs(actual[key] - expected[key]) > 1e-12
        for key in ("tp", "fp", "fn", "precision", "recall", "f1")
    ):
        raise PhoneExperimentError("baseline or frozen Variant A did not reproduce")


def run_experiment(
    normalized: Path, prediction_paths: Sequence[Path], saved_a: Path,
    policy: RegionPolicy, progress_every: int = 5000,
) -> dict[str, Any]:
    started = time.perf_counter()
    mapping = load_label_map()
    normalized_hash = sha256_file(normalized)
    saved_hash = sha256_file(saved_a)
    reference = json.loads(saved_a.read_text())
    expected = next(
        (d for d in reference["datasets"] if d["normalized_sha256"] == normalized_hash), None
    )
    if expected is None or expected["policy"]["explicit_region"] != policy.region:
        raise PhoneExperimentError("normalized input or region differs from frozen Variant A")
    if expected["parser"]["version"] != phonenumbers.__version__:
        raise PhoneExperimentError("parser version differs from frozen Variant A")
    if len(prediction_paths) != 2 or len({p.resolve() for p in prediction_paths}) != 2:
        raise PhoneExperimentError("two distinct existing threshold files are required")
    conditions = []
    for path in prediction_paths:
        header = read_header(path)
        threshold = header.get("threshold")
        old = next((c for c in expected["conditions"] if c["threshold"] == threshold), None)
        model = header.get("model", {})
        fingerprint = sha256_file(path)
        if (
            isinstance(threshold, bool) or old is None
            or fingerprint != old["input_predictions_sha256"]
            or header.get("normalized_sha256") != normalized_hash
            or model.get("model_id") != mapping.model_id
            or model.get("model_revision") != mapping.model_revision
            or model.get("max_width") != 24
        ):
            raise PhoneExperimentError("persisted prediction identity differs from frozen inputs")
        conditions.append({
            "path": path, "hash": fingerprint, "threshold": threshold, "expected": old,
            "states": {
                (supported, variant): EvaluationState()
                for supported in (False, True) for variant in VARIANTS
            },
            "pending": [], "retention": Counter(), "regression": Counter(),
            "added_boundaries_a": Counter(), "added_boundaries_b": Counter(),
        })
    if {c["threshold"] for c in conditions} != {0.3, 0.7}:
        raise PhoneExperimentError("both fixed thresholds are required")
    counts = Counter()
    actions: Counter[str] = Counter()
    boundaries_a = Counter(dict.fromkeys(BOUNDARIES, 0))
    boundaries_b = Counter(dict.fromkeys(BOUNDARIES, 0))
    transitions: Counter[str] = Counter()
    streams = [_normalized_records(normalized), *(iter_records(c["path"]) for c in conditions)]
    for index, bundle in enumerate(zip_longest(*streams)):
        row, *records = bundle
        if row is None or any(r is None or r.example_index != index for r in records):
            raise PhoneExperimentError("dataset and prediction records are not aligned")
        text, truth = validate_normalized_example(row)
        _validate_native(row, text, mapping)
        candidates = phone_candidates(text, policy)  # Frozen A; no replacement extractor.
        phone_truth = [Span(e.start, e.end) for e in truth if e.entity_type == PHONE]
        refined = []
        for span in candidates.validated:
            old_relation = classify_candidate(span, phone_truth)
            result = refine_candidate(text, span, policy)
            actions[result.reason] += 1
            boundaries_a[old_relation] += 1
            if result.span is None:
                new_relation = "rejected"
            else:
                refined.append(result.span)
                new_relation = classify_candidate(result.span, phone_truth)
            transitions[old_relation + " -> " + new_relation] += 1
        selected = deduplicate_candidates(refined, len(text))
        boundaries_b.update(classify_candidate(span, phone_truth) for span in selected)
        counts["examples"] += 1
        counts["raw_candidates"] += len(candidates.raw)
        counts["validated_candidates"] += len(candidates.validated)
        counts["selected_unique_candidates_b"] += len(selected)
        counts["refinement_deduplications"] += len(refined) - len(selected)
        for condition, record in zip(conditions, records, strict=True):
            a = merge_phone_candidates(text, record.predictions, candidates.validated, mapping)
            b = merge_phone_candidates(text, record.predictions, selected, mapping)
            baseline = tuple(to_gateway(record.predictions, mapping).kept)
            mapped_a = tuple(to_gateway(a.predictions, mapping).kept)
            mapped_b = tuple(to_gateway(b.predictions, mapping).kept)
            condition["pending"].append((index, truth, baseline, mapped_a, mapped_b))
            condition["retention"].update(recovery_accounting(
                Counter(phone_truth), _phone_counts(baseline),
                _phone_counts(mapped_a), _phone_counts(mapped_b),
            ))
            condition["added_boundaries_a"].update(
                classify_candidate(span, phone_truth) for span in a.added
            )
            condition["added_boundaries_b"].update(
                classify_candidate(span, phone_truth) for span in b.added
            )
            if b.predictions[:len(record.predictions)] != record.predictions:
                raise PhoneExperimentError("existing model prediction invariant failed")
            regression = condition["regression"]
            regression.update({
                "baseline_native_predictions": len(record.predictions),
                "variant_a_native_predictions": len(a.predictions),
                "variant_b_native_predictions": len(b.predictions),
                "variant_a_added": len(a.added), "variant_b_added": len(b.added),
                "existing_predictions_modified": 0, "existing_predictions_deleted": 0,
                "non_phone_predictions_changed": 0, "records_verified": 1,
            })
            if len(condition["pending"]) >= 64:
                _flush(condition)
        if progress_every > 0 and counts["examples"] % progress_every == 0:
            print(json.dumps({
                "event": "progress", "examples": counts["examples"],
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }), flush=True)
    if not counts["examples"]:
        raise PhoneExperimentError("empty normalized dataset")
    if (
        sha256_file(normalized) != normalized_hash or sha256_file(saved_a) != saved_hash
        or any(sha256_file(c["path"]) != c["hash"] for c in conditions)
    ):
        raise PhoneExperimentError("input hash changed during evaluation")
    if counts["raw_candidates"] != expected["candidates"]["raw"] or (
        counts["validated_candidates"] != expected["candidates"]["validated"]
        or dict(boundaries_a) != expected["candidates"]["validated_boundaries"]
    ):
        raise PhoneExperimentError("frozen Variant A candidate set did not reproduce")
    outputs = []
    for condition in sorted(conditions, key=lambda c: c["threshold"]):
        _flush(condition, True)
        output = {
            "threshold": condition["threshold"], "input_predictions_sha256": condition["hash"],
            "regression": dict(condition["regression"]),
            "recovery_retention": dict(condition["retention"]),
            "added_candidate_boundaries_a": dict(condition["added_boundaries_a"]),
            "added_candidate_boundaries_b": dict(condition["added_boundaries_b"]),
        }
        for supported, view in (
            (False, "full_strict"), (True, "gateway_supported_only_strict"),
        ):
            metrics = {
                variant: _strict_summary(condition["states"][supported, variant])
                for variant in VARIANTS
            }
            _same_metrics(metrics["baseline"], condition["expected"][view]["baseline"])
            _same_metrics(metrics["variant_a"], condition["expected"][view]["hybrid"])
            metrics["delta_b_vs_a"] = _delta(metrics["variant_a"], metrics["variant_b"])
            output[view] = metrics
        phone = {
            variant: _phone_metrics(condition["states"][False, variant]) for variant in VARIANTS
        }
        _same_metrics(phone["variant_a"], condition["expected"]["phone_number"]["hybrid"])
        phone["delta_b_vs_a"] = _delta(phone["variant_a"], phone["variant_b"])
        output["phone_number"] = phone
        retention = output["recovery_retention"]
        recovered = retention["variant_a_recovered_fn"]
        retention["retention_pct"] = (
            100 * retention["variant_b_retained_recoveries"] / recovered if recovered else None
        )
        if (
            recovered != phone["variant_a"]["tp"] - phone["baseline"]["tp"]
            or retention["variant_b_total_added_fp"]
            != phone["variant_b"]["fp"] - phone["baseline"]["fp"]
            or retention["variant_a_added_fp"]
            != phone["variant_a"]["fp"] - phone["baseline"]["fp"]
            or sum(output["added_candidate_boundaries_b"].values())
            != output["regression"]["variant_b_added"]
        ):
            raise PhoneExperimentError("recovery or FP identity accounting failed")
        for view in ("full_strict", "gateway_supported_only_strict"):
            if any(
                output[view]["delta_b_vs_a"][k] != phone["delta_b_vs_a"][k]
                for k in ("tp", "fp", "fn")
            ):
                raise PhoneExperimentError("non-phone strict count invariant failed")
        outputs.append(output)
    return {
        "experiment": VARIANT, "parser_version": phonenumbers.__version__,
        "explicit_region": policy.region, "normalized_sha256": normalized_hash,
        "frozen_variant_a_report_sha256": saved_hash, "candidate_counts": dict(counts),
        "refinement_actions": dict(actions),
        "validated_boundaries_a": dict(boundaries_a),
        "selected_unique_boundaries_b": dict(boundaries_b),
        "candidate_boundary_transitions": dict(transitions),
        "conditions": outputs, "inputs_unchanged": True,
        "elapsed_seconds": time.perf_counter() - started,
        "privacy": "aggregate only; no values, snippets, offsets, or example IDs",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, action="append", required=True)
    parser.add_argument("--variant-a-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--region", default=None)
    parser.add_argument("--progress-every", type=int, default=5000)
    args = parser.parse_args(argv)
    if args.report.exists() or is_within_repository(args.report):
        print("error: report must be a new external path", file=sys.stderr)
        return 2
    try:
        report = run_experiment(
            args.normalized, args.predictions, args.variant_a_report,
            RegionPolicy(args.region), args.progress_every,
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except (ValueError, OSError, TypeError, AttributeError, KeyError):
        print(
            "error: boundary experiment failed input, parser, or invariant checks",
            file=sys.stderr,
        )
        return 2
    print(json.dumps({
        "event": "complete", "examples": report["candidate_counts"]["examples"],
        "elapsed_seconds": round(report["elapsed_seconds"], 3),
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())