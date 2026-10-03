"""Source-only PAN candidates and narrowly scoped offline type arbitration."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_gliner_ood_errors import (
    _normalized_records,
    _strict_summary,
    _validate_native,
)
from scripts.evaluation_checkpoint import EvaluationState, sha256_file
from scripts.gliner_evaluation import GlinerLabelMap, RawPrediction, load_label_map, to_gateway
from scripts.gliner_phone_boundary_refinement import _flush, _same_metrics
from scripts.gliner_phone_hybrid_experiment import (
    Span,
    _delta,
    classify_candidate,
    deduplicate_candidates,
    overlaps,
)
from scripts.gliner_predictions import iter_records, read_header
from scripts.nemotron_pii import is_within_repository
from scripts.presidio_evaluation import validate_normalized_example

CARD = "CREDIT_CARD"
VERSIONS = ("baseline", "variant_a", "variant_b")
CLASSES = ("checksum_valid", "checksum_invalid", "unable_to_validate")
RAW = re.compile(r"(?<![\w])(?:[0-9][0-9 \t-]*[0-9]|[0-9])(?![\w])")
CONTEXT = re.compile(
    r"(?<!\w)(?:(?:credit|debit|payment)[ \t]+card"
    r"(?:[ \t]+(?:number|no\.?))?|card[ \t]+(?:number|no\.?)"
    r"|cc[ \t]+(?:number|no\.?))[ \t]*[:=#-]?[ \t]*$", re.IGNORECASE,
)
COMPETITORS = frozenset({"account_number", "national_id", "customer_id", "employee_id"})


class CardExperimentError(ValueError):
    """Fixed messages without source content."""


@dataclass(frozen=True)
class Candidate:
    span: Span
    checksum: str
    explicit_context: bool


@dataclass(frozen=True)
class Candidates:
    accepted: tuple[Candidate, ...]
    stats: Counter[str]


@dataclass(frozen=True)
class Merge:
    predictions: tuple[RawPrediction, ...]
    added: tuple[Candidate, ...]
    suppressed: tuple[RawPrediction, ...]
    stats: Counter[str]


def luhn(digits: str) -> bool:
    if not isinstance(digits, str) or not re.fullmatch(r"[0-9]+", digits):
        raise CardExperimentError("checksum input must be ASCII digits")
    total = 0
    for index, digit in enumerate(reversed(digits)):
        value = int(digit)
        if index % 2:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def validation(value: str) -> tuple[str, str]:
    """Validation only; no rewritten source string leaves this function."""
    if not isinstance(value, str):
        raise CardExperimentError("validation input must be a string")
    if len(value) > 128 or not re.fullmatch(r"[0-9]+(?:[ \t-]+[0-9]+)*", value):
        return "unable_to_validate", "malformed_structure"
    digits = re.sub(r"[ \t-]", "", value)
    if not 13 <= len(digits) <= 19:
        return "unable_to_validate", "invalid_length"
    if "-" in value:
        if " " in value or "\t" in value or "--" in value:
            return "unable_to_validate", "malformed_separators"
        groups = value.split("-")
    else:
        groups = re.split(r"[ \t]+", value)
    if len(groups) > 1 and (
        len(groups) > 8 or any(not 1 <= len(group) <= 6 for group in groups)
    ):
        return "unable_to_validate", "malformed_grouping"
    return ("checksum_valid" if luhn(digits) else "checksum_invalid"), "validated_structure"


def has_context(text: str, span: Span) -> bool:
    start = max(0, span.start - 64)
    # Context cannot cross a line and must end immediately before the candidate.
    start = max(
        start, text.rfind("\n", start, span.start) + 1,
        text.rfind("\r", start, span.start) + 1,
    )
    # Search the original string so lookbehind sees the character before the window.
    return CONTEXT.search(text, start, span.start) is not None


def generate_candidates(text: str) -> Candidates:
    if not isinstance(text, str):
        raise CardExperimentError("candidate text must be a string")
    extracted = [Span(m.start(), m.end()) for m in RAW.finditer(text)]
    raw = deduplicate_candidates(extracted, len(text))
    stats: Counter[str] = Counter({
        "raw_candidates": len(extracted), "unique_raw_candidates": len(raw),
        "duplicate_candidates_removed": len(extracted) - len(raw),
        "length_valid": 0, "checksum_valid": 0, "checksum_invalid": 0,
        "parser_validation_rejected": 0, "accepted": 0,
        "checksum_invalid_without_context_rejected": 0,
    })
    accepted = []
    for span in raw:
        value = text[span.start:span.end]
        if 13 <= sum(c in "0123456789" for c in value) <= 19:
            stats["length_valid"] += 1
        checksum, reason = validation(value)
        connected = (
            span.start > 0 and text[span.start - 1] in "+-*"
            or span.start > 1 and text[span.start - 1] in "./"
            and text[span.start - 2].isdecimal()
            or span.end < len(text) and text[span.end] == "*"
            or span.end + 1 < len(text) and text[span.end] in "./"
            and text[span.end + 1].isdecimal()
        )
        if connected:
            stats["parser_validation_rejected"] += 1
            stats["rejection_connected_sign_mask_or_numeric_fragment"] += 1
            continue
        if checksum == "unable_to_validate":
            stats["parser_validation_rejected"] += 1
            stats["rejection_" + reason] += 1
            continue
        stats[checksum] += 1
        context = has_context(text, span)
        if checksum == "checksum_invalid" and not context:
            stats["checksum_invalid_without_context_rejected"] += 1
            continue
        accepted.append(Candidate(span, checksum, context))
        stats["accepted"] += 1
        stats["accepted_" + checksum] += 1
    if len(raw) != (
        stats["parser_validation_rejected"] + stats["checksum_valid"] + stats["checksum_invalid"]
    ):
        raise CardExperimentError("candidate partition invariant failed")
    stats["candidates_rejected"] = (
        stats["parser_validation_rejected"] + stats["checksum_invalid_without_context_rejected"]
    )
    return Candidates(tuple(accepted), stats)


def merge_candidates(
    text: str, predictions: Sequence[RawPrediction], candidates: Sequence[Candidate],
    mapping: GlinerLabelMap, arbitrate: bool = False,
) -> Merge:
    if not isinstance(text, str) or mapping.mapping.get("credit_debit_card") != CARD:
        raise CardExperimentError("merge input or canonical label is invalid")
    original = tuple(predictions)
    for p in original:
        if (
            not isinstance(p, RawPrediction) or p.label not in mapping.mapping
            or isinstance(p.start, bool) or isinstance(p.end, bool)
            or not isinstance(p.start, int) or not isinstance(p.end, int)
            or not 0 <= p.start < p.end <= len(text)
            or isinstance(p.score, bool) or not isinstance(p.score, int | float)
            or not math.isfinite(p.score)
        ):
            raise CardExperimentError("model prediction is invalid")
    by_span = {}
    for c in candidates:
        if not isinstance(c, Candidate):
            raise CardExperimentError("candidate object is invalid")
        deduplicate_candidates([c.span], len(text))
        checksum, _ = validation(text[c.span.start:c.span.end])
        if (
            checksum != c.checksum or not isinstance(c.explicit_context, bool)
            or c.explicit_context != has_context(text, c.span)
            or checksum not in CLASSES[:2]
            or checksum == "checksum_invalid" and not c.explicit_context
        ):
            raise CardExperimentError("candidate acceptance evidence is invalid")
        if c.span in by_span and by_span[c.span] != c:
            raise CardExperimentError("duplicate candidate evidence differs")
        by_span[c.span] = c
    unique = tuple(by_span[s] for s in sorted(by_span))
    stats = Counter({
        "candidate_duplicates_removed": len(candidates) - len(unique),
        "exact_duplicates_skipped": 0, "cross_type_conflicts_encountered": 0,
        "candidates_with_cross_type_conflicts": 0, "conflicts_resolved": 0,
        "candidates_resolved": 0, "rejected_as_ambiguous": 0,
        "predictions_suppressed": 0, "predictions_changed": 0,
        "model_predictions_preserved": 0, "candidate_additions": 0,
    })
    remove = set()
    for c in unique:
        conflicts = [
            (i, p) for i, p in enumerate(original)
            if mapping.mapping[p.label] != CARD and overlaps(c.span, Span(p.start, p.end))
        ]
        stats["cross_type_conflicts_encountered"] += len(conflicts)
        stats["candidates_with_cross_type_conflicts"] += bool(conflicts)
        if not arbitrate or not conflicts:
            continue
        strong = c.checksum == "checksum_valid" and c.explicit_context
        eligible = all(
            p.label in COMPETITORS
            and mapping.mapping[p.label] in {"BANK_ACCOUNT", "ID_CARD"}
            and Span(p.start, p.end) == c.span
            for _, p in conflicts
        )
        if not strong or not eligible:
            stats["rejected_as_ambiguous"] += 1
            stats["ambiguous_insufficient_priority" if not strong
                  else "ambiguous_nonexact_or_protected_overlap"] += 1
            continue
        stats["candidates_resolved"] += 1
        stats["conflicts_resolved"] += len(conflicts)
        for i, p in conflicts:
            remove.add(i)
            stats["suppressed_native_" + p.label] += 1
            stats["suppressed_type_" + mapping.mapping[p.label]] += 1
    kept = tuple(p for i, p in enumerate(original) if i not in remove)
    suppressed = tuple(p for i, p in enumerate(original) if i in remove)
    existing = {Span(p.start, p.end) for p in kept if mapping.mapping[p.label] == CARD}
    added = tuple(c for c in unique if c.span not in existing)
    stats["exact_duplicates_skipped"] = len(unique) - len(added)
    stats["predictions_suppressed"] = len(suppressed)
    stats["model_predictions_preserved"] = len(kept)
    stats["candidate_additions"] = len(added)
    merged = kept + tuple(
        RawPrediction("credit_debit_card", c.span.start, c.span.end, 1.0) for c in added
    )
    if merged[:len(kept)] != kept or (not arbitrate and kept != original):
        raise CardExperimentError("preservation invariant failed")
    if any(p.label not in COMPETITORS for p in suppressed):
        raise CardExperimentError("unrelated prediction suppression failed")
    return Merge(merged, added, suppressed, stats)


def type_metrics(state: EvaluationState, entity_type: str = CARD) -> dict[str, Any]:
    part = EvaluationState()
    for field in ("true_positives", "false_positives", "false_negatives"):
        getattr(part, field)[entity_type] = getattr(state, field)[entity_type]
    return _strict_summary(part)


def run_experiment(
    normalized: Path, predictions: Sequence[Path], reference: Path,
    input_reference: Path, dataset: str,
    progress_every: int = 5000,
) -> dict[str, Any]:
    started = time.perf_counter()
    mapping = load_label_map()
    norm_hash, ref_hash = sha256_file(normalized), sha256_file(reference)
    input_ref_hash = sha256_file(input_reference)
    fingerprints = next(
        (d for d in json.loads(input_reference.read_text())["datasets"]
         if d["dataset"] == dataset), None
    )
    frozen = next(
        (d for d in json.loads(reference.read_text())["datasets"] if d["dataset"] == dataset), None
    )
    if (
        frozen is None or fingerprints is None
        or fingerprints["normalized_sha256"] != norm_hash
        or len(predictions) != 2 or len({p.resolve() for p in predictions}) != 2
    ):
        raise CardExperimentError("two fixed inputs and a frozen baseline are required")
    conditions = []
    for path in predictions:
        header = read_header(path)
        model = header.get("model", {})
        threshold = header.get("threshold")
        expected = next((c for c in frozen["conditions"] if c["threshold"] == threshold), None)
        pinned = next(
            (c for c in fingerprints["conditions"] if c["threshold"] == threshold), None
        )
        if (
            isinstance(threshold, bool) or threshold not in (0.3, 0.7) or expected is None
            or header.get("normalized_sha256") != norm_hash
            or model.get("model_id") != mapping.model_id
            or model.get("model_revision") != mapping.model_revision
            or model.get("max_width") != 24
            or pinned is None or sha256_file(path) != pinned["input_predictions_sha256"]
        ):
            raise CardExperimentError("prediction identity differs from pinned baseline")
        conditions.append({
            "path": path, "hash": sha256_file(path), "header": header,
            "threshold": threshold, "expected": expected,
            "states": {
                (supported, version): EvaluationState()
                for supported in (False, True) for version in VERSIONS
            }, "pending": [], "a": Counter(), "b": Counter(), "gt_matches": Counter(),
            "tier_outcomes": Counter(), "suppression_outcomes": Counter(),
            "boundaries": Counter(),
        })
    if {c["threshold"] for c in conditions} != {0.3, 0.7} or any(
        {k: v for k, v in c["header"].items() if k != "threshold"}
        != {k: v for k, v in conditions[0]["header"].items() if k != "threshold"}
        for c in conditions
    ):
        raise CardExperimentError("threshold identities otherwise differ")
    gt_classes = Counter(dict.fromkeys(CLASSES, 0))
    gt_native: dict[str, Counter[str]] = {}
    stats: Counter[str] = Counter()
    examples = 0
    streams = [_normalized_records(normalized), *(iter_records(c["path"]) for c in conditions)]
    for index, bundle in enumerate(zip_longest(*streams)):
        row, *records = bundle
        if row is None or any(r is None or r.example_index != index for r in records):
            raise CardExperimentError("dataset and prediction indices are not aligned")
        text, truth = validate_normalized_example(row)
        _validate_native(row, text, mapping)
        gt_spans = Counter(Span(e.start, e.end) for e in truth if e.entity_type == CARD)
        partitions: dict[str, Counter[Span]] = {k: Counter() for k in CLASSES}
        for entity in truth:
            if entity.entity_type == CARD:
                span = Span(entity.start, entity.end)
                category, _ = validation(text[span.start:span.end])
                partitions[category][span] += 1
                gt_classes[category] += 1
        for entity in row["native_entities"]:
            if mapping.mapping.get(entity["native_label"]) == CARD:
                category, _ = validation(text[entity["start"]:entity["end"]])
                gt_native.setdefault(entity["native_label"], Counter())[category] += 1
        result = generate_candidates(text)
        stats.update(result.stats)
        for condition, record in zip(conditions, records, strict=True):
            a = merge_candidates(text, record.predictions, result.accepted, mapping)
            b = merge_candidates(text, record.predictions, result.accepted, mapping, True)
            mapped = [
                tuple(to_gateway(p, mapping).kept)
                for p in (record.predictions, a.predictions, b.predictions)
            ]
            condition["pending"].append((index, truth, *mapped))
            condition["a"].update(a.stats)
            condition["b"].update(b.stats)
            for name, entities in zip(VERSIONS, mapped, strict=True):
                card_counts = Counter(
                    Span(e.start, e.end) for e in entities if e.entity_type == CARD
                )
                for category, partition in partitions.items():
                    condition["gt_matches"][name + "_" + category] += sum(
                        (partition & card_counts).values()
                    )
            for c in a.added:
                condition["tier_outcomes"][c.checksum + "_tp"] += bool(gt_spans[c.span])
                condition["tier_outcomes"][c.checksum + "_fp"] += not bool(gt_spans[c.span])
                relation = classify_candidate(c.span, gt_spans)
                if relation == "no_phone_gt_overlap":
                    relation = "no_credit_card_gt_overlap"
                condition["boundaries"][relation] += 1
            # GT is used here only to measure the consequence of the already-fixed rule.
            available = Counter((e.entity_type, e.start, e.end) for e in truth)
            for removed in to_gateway(b.suppressed, mapping).kept:
                key = (removed.entity_type, removed.start, removed.end)
                was_tp = bool(available[key])
                available[key] -= was_tp
                condition["suppression_outcomes"][
                    removed.entity_type + ("_exact_gt_matches_removed" if was_tp
                                           else "_nonmatching_predictions_removed")
                ] += 1
            if len(condition["pending"]) >= 64:
                _flush(condition)
        examples += 1
        if progress_every > 0 and examples % progress_every == 0:
            print(json.dumps({
                "event": "progress", "examples": examples,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }), flush=True)
    if not examples:
        raise CardExperimentError("empty normalized dataset")
    if (
        sha256_file(normalized) != norm_hash or sha256_file(reference) != ref_hash
        or sha256_file(input_reference) != input_ref_hash
    ) or any(
        sha256_file(c["path"]) != c["hash"] for c in conditions
    ):
        raise CardExperimentError("input hash changed during evaluation")
    outputs = []
    for condition in sorted(conditions, key=lambda c: c["threshold"]):
        _flush(condition, True)
        output = {
            "threshold": condition["threshold"], "prediction_sha256": condition["hash"],
            "candidate_hybrid": dict(condition["a"]),
            "arbitration": dict(condition["b"]),
            "gt_partition_matches": dict(condition["gt_matches"]),
            "candidate_acceptance_tier_tp_fp": dict(condition["tier_outcomes"]),
            "suppression_gt_consequences": dict(condition["suppression_outcomes"]),
            "added_candidate_boundaries": dict(condition["boundaries"]),
        }
        for supported, view in (
            (False, "full_strict"), (True, "gateway_supported_only_strict"),
        ):
            output[view] = {
                name: _strict_summary(condition["states"][supported, name]) for name in VERSIONS
            }
            _same_metrics(output[view]["baseline"], condition["expected"][view])
            output[view]["delta_a_vs_baseline"] = _delta(
                output[view]["baseline"], output[view]["variant_a"]
            )
            output[view]["delta_b_vs_a"] = _delta(
                output[view]["variant_a"], output[view]["variant_b"]
            )
        output["credit_card"] = {
            name: type_metrics(condition["states"][False, name]) for name in VERSIONS
        }
        expected_card = next(
            v for v in condition["expected"]["inventory"] if v["type"] == CARD
        )
        _same_metrics(output["credit_card"]["baseline"], expected_card)
        _same_metrics(output["credit_card"]["variant_a"], output["credit_card"]["variant_b"])
        # BANK_ACCOUNT and ID_CARD are outside the fixed seven-type view.
        _same_metrics(
            output["gateway_supported_only_strict"]["variant_a"],
            output["gateway_supported_only_strict"]["variant_b"],
        )
        output["credit_card"]["delta_a_vs_baseline"] = _delta(
            output["credit_card"]["baseline"], output["credit_card"]["variant_a"]
        )
        output["credit_card"]["delta_b_vs_a"] = _delta(
            output["credit_card"]["variant_a"], output["credit_card"]["variant_b"]
        )
        output["credit_card_gt_partition_outcomes"] = {
            name: {
                category: {
                    "gt": gt_classes[category],
                    "tp": condition["gt_matches"][name + "_" + category],
                    "fn": gt_classes[category]
                    - condition["gt_matches"][name + "_" + category],
                }
                for category in CLASSES
            }
            for name in VERSIONS
        }
        for name in VERSIONS:
            metrics = output["credit_card"][name]
            if metrics["gt_entity_count"] != sum(gt_classes.values()):
                raise CardExperimentError("immutable GT count invariant failed")
            if sum(
                condition["gt_matches"][name + "_" + category] for category in CLASSES
            ) != metrics["tp"]:
                raise CardExperimentError("GT checksum partition invariant failed")
        # Removing contextual checksum-invalid additions gives a strict Luhn-only ablation.
        tier = condition["tier_outcomes"]
        ablation = EvaluationState()
        a_state = condition["states"][False, "variant_a"]
        ablation.true_positives[CARD] = (
            a_state.true_positives[CARD] - tier["checksum_invalid_tp"]
        )
        ablation.false_positives[CARD] = (
            a_state.false_positives[CARD] - tier["checksum_invalid_fp"]
        )
        ablation.false_negatives[CARD] = (
            a_state.false_negatives[CARD] + tier["checksum_invalid_tp"]
        )
        output["credit_card_luhn_only_additions_ablation"] = _strict_summary(ablation)
        for supported in (False, True):
            old = condition["states"][supported, "baseline"]
            new = condition["states"][supported, "variant_a"]
            for label in set(old.entity_types()) | set(new.entity_types()):
                if label != CARD and any(
                    getattr(old, field)[label] != getattr(new, field)[label]
                    for field in ("true_positives", "false_positives", "false_negatives")
                ):
                    raise CardExperimentError("hybrid non-card invariant failed")
        if (
            condition["a"]["model_predictions_preserved"]
            != condition["b"]["model_predictions_preserved"]
            + condition["b"]["predictions_suppressed"]
            or condition["a"]["candidate_additions"] != condition["b"]["candidate_additions"]
        ):
            raise CardExperimentError("arbitration preservation accounting failed")
        output["invariants"] = {
            "variant_a_non_card_predictions_unchanged": True,
            "variant_b_unrelated_predictions_unchanged": True,
            "surviving_model_objects_scores_offsets_order_unchanged": True,
            "inputs_unchanged": True,
            "arbitration_credit_card_metrics_equal_hybrid": True,
            "arbitration_supported_only_metrics_equal_hybrid": True,
        }
        outputs.append(output)
    return {
        "dataset": dataset, "examples": examples,
        "variants": {"variant_a": "gliner_plus_validated_credit_card_candidates",
                     "variant_b": "gliner_credit_card_conservative_arbitration"},
        "normalized_sha256": norm_hash, "baseline_report_sha256": ref_hash,
        "frozen_input_reference_sha256": input_ref_hash,
        "candidate_statistics": dict(stats), "gt_checksum_partition": dict(gt_classes),
        "gt_checksum_partition_by_native_label": {
            k: dict(v) for k, v in gt_native.items()
        },
        "credit_card_gt_count": sum(gt_classes.values()),
        "zero_credit_card_gt_support": not sum(gt_classes.values()),
        "conditions": outputs, "elapsed_seconds": time.perf_counter() - started,
        "coverage_warning": "Zero-GT strict FP are not confirmed semantic failures.",
        "no_inference": True, "privacy": "aggregates only; no values, snippets, offsets or IDs",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, action="append", required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--input-reference", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=5000)
    args = parser.parse_args(argv)
    if args.report.exists() or is_within_repository(args.report):
        print("error: report must be a new external path", file=sys.stderr)
        return 2
    try:
        result = run_experiment(
            args.normalized, args.predictions, args.baseline_report, args.input_reference,
            args.dataset, args.progress_every,
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except (ValueError, OSError, TypeError, AttributeError, KeyError, StopIteration):
        print("error: credit-card experiment failed input or invariant checks", file=sys.stderr)
        return 2
    print(json.dumps({
        "event": "complete", "examples": result["examples"],
        "elapsed_seconds": round(result["elapsed_seconds"], 3),
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())