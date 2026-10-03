"""Offline one-to-one DATE_TIME replacements anchored to immutable model spans."""

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

from scripts import gliner_datetime_boundary_refinement as frozen
from scripts.analyze_gliner_ood_errors import _normalized_records, _strict_summary, _validate_native
from scripts.evaluation_checkpoint import EvaluationState, sha256_file
from scripts.gliner_evaluation import GlinerLabelMap, RawPrediction, load_label_map, to_gateway
from scripts.gliner_phone_boundary_refinement import _same_metrics
from scripts.gliner_phone_hybrid_experiment import Span, _delta, classify_candidate, overlaps
from scripts.gliner_predictions import iter_records, read_header
from scripts.nemotron_pii import is_within_repository
from scripts.presidio_evaluation import evaluation_type, validate_normalized_example

TARGET = "DATE_TIME"
RADIUS = 32
MAX_ANCHOR = 256
PERIPHERAL = " \t\r\n()[]{}\"'“”‘’.,;:!?"
RELATIONS = (
    "exact_match", "candidate_contained_by_gt", "gt_contained_by_candidate",
    "partial_overlap", "no_datetime_gt_overlap",
)


class AnchoredError(ValueError):
    """Errors never contain source-controlled data."""


@dataclass(frozen=True)
class Decision:
    original: Span
    refined: Span
    status: str
    reason: str


def contains(outer: Span, inner: Span) -> bool:
    return outer.start <= inner.start and inner.end <= outer.end


def decide(text: str, anchor: Span) -> Decision:
    """Inspect only a bounded neighborhood; never receives GT or dataset identity."""
    if (
        not isinstance(text, str) or not isinstance(anchor, Span)
        or any(isinstance(v, bool) or not isinstance(v, int)
               for v in (anchor.start, anchor.end))
        or not 0 <= anchor.start < anchor.end <= len(text)
    ):
        raise AnchoredError("anchor input is invalid")

    def rejected(reason: str) -> Decision:
        return Decision(anchor, anchor, "rejected", reason)

    if anchor.end - anchor.start > MAX_ANCHOR:
        return rejected("anchor_size_limit")
    left, right = max(0, anchor.start - RADIUS), min(len(text), anchor.end + RADIUS)
    local = text[left:right]
    # The frozen generator is restricted to this neighborhood, never the document.
    candidates, _, _ = frozen.generate(local)
    overlapping = [
        c for c in candidates
        if overlaps(anchor, Span(left + c.core.start, left + c.core.end))
    ]
    if len(overlapping) > 1:
        return rejected("ambiguous_multiple_cores")
    if not overlapping:
        for match in frozen.PATTERN.finditer(local):
            if overlaps(anchor, Span(left + match.start(), left + match.end())):
                reason, kind = frozen.validate_core(match[0])
                if reason and reason.startswith("ambiguous"):
                    return rejected(reason)
                if kind != "calendar_date" and re.match(
                    r"[ \t]+[A-Z]{2,5}(?!\w)", local[match.end():]
                ):
                    return rejected("ambiguous_timezone_attachment")
        return rejected("no_valid_overlapping_core")
    c = overlapping[0]
    core = Span(left + c.core.start, left + c.core.end)
    if core.start == left and left > 0 or core.end == right and right < len(text):
        return rejected("neighborhood_edge")
    # A narrower core may remove only peripheral characters, never digits or words.
    removed = text[anchor.start:core.start] + text[core.end:anchor.end]
    if any(ch not in PERIPHERAL for ch in removed):
        return rejected("semantic_content_would_be_removed")
    value = local[c.core.start:c.core.end]
    date = frozen.DATE_RE.match(value)
    clock_start = 0
    if date:
        date_span = Span(core.start, core.start + date.end())
        if not contains(anchor, date_span):
            return rejected("calendar_component_not_in_anchor")
        if date.end() == len(value):
            clock_start = len(value)
        else:
            gap = re.match(r"(?:[Tt]|[ \t]+)", value[date.end():])
            if gap is None:
                return rejected("invalid_attachment")
            clock_start = date.end() + gap.end()
    clock = frozen.TIME_RE.match(value, clock_start) if clock_start < len(value) else None
    if clock:
        minimum = re.match(r"[0-9]{1,2}:[0-9]{2}", clock[0])
        # Hour-only clocks must already include their AM/PM marker in the anchor.
        minimum_end = minimum.end() if minimum else len(clock[0])
        support = Span(core.start + clock_start, core.start + clock_start + minimum_end)
        if not contains(anchor, support):
            return rejected("clock_structure_not_in_anchor")
    # No leftward semantic expansion; suffix completion must be part of the parsed clock.
    if core.start < anchor.start or core.end > anchor.end and clock is None:
        return rejected("unsupported_expansion")
    if core == anchor:
        return Decision(anchor, anchor, "unchanged", "already_structurally_complete")
    return Decision(anchor, core, "replaced", "unique_structural_boundary")


def refine(
    text: str, predictions: Sequence[RawPrediction], mapping: GlinerLabelMap | None = None,
) -> tuple[tuple[RawPrediction, ...], tuple[RawPrediction, ...], dict[int, Decision]]:
    """Return original tuple, one-to-one scoring view, and in-memory trace only."""
    if not isinstance(text, str):
        raise AnchoredError("source is invalid")
    mapping = mapping or load_label_map()
    original = tuple(predictions)
    revised = []
    decisions = {}
    cache: dict[Span, Decision] = {}
    for index, p in enumerate(original):
        if (
            not isinstance(p, RawPrediction) or p.label not in mapping.mapping
            or any(isinstance(v, bool) or not isinstance(v, int) for v in (p.start, p.end))
            or not 0 <= p.start < p.end <= len(text)
            or isinstance(p.score, bool) or not isinstance(p.score, int | float)
            or not math.isfinite(p.score)
        ):
            raise AnchoredError("prediction is invalid")
        if evaluation_type(mapping.mapping[p.label]) != TARGET:
            revised.append(p)
            continue
        span = Span(p.start, p.end)
        if span not in cache:
            cache[span] = decide(text, span)
        d = cache[span]
        decisions[index] = d
        revised.append(
            RawPrediction(p.label, d.refined.start, d.refined.end, p.score)
            if d.status == "replaced" else p
        )
    result = tuple(revised)
    if len(result) != len(original) or any(
        before.label != after.label or before.score != after.score
        or index not in decisions and before is not after
        or index in decisions and decisions[index].status != "replaced" and before is not after
        for index, (before, after) in enumerate(zip(original, result, strict=True))
    ):
        raise AnchoredError("one-to-one preservation failed")
    return original, result, decisions


def relation(span: Span, truth: Counter[Span]) -> str:
    value = classify_candidate(span, truth)
    return "no_datetime_gt_overlap" if value == "no_phone_gt_overlap" else value


def diagnose(
    original: Sequence[RawPrediction], revised: Sequence[RawPrediction],
    decisions: dict[int, Decision], truth: Counter[Span], condition: dict[str, Any],
) -> None:
    counts = condition["accounting"]
    counts.update({
        "original_model_predictions": len(original), "refined_view_predictions": len(revised),
        "datetime_anchors": len(decisions), "deleted_predictions": 0,
        "added_predictions": 0, "changed_non_datetime_predictions": 0,
    })
    before = Counter(d.original for d in decisions.values())
    after = Counter(d.refined for d in decisions.values())
    before_exact, after_exact = before & truth, after & truth
    condition["effects"]["recovered_fn_gross"] += sum((after_exact - before_exact).values())
    condition["effects"]["lost_tp_gross"] += sum((before_exact - after_exact).values())
    stable = Counter(d.original for d in decisions.values() if d.status != "replaced")
    before_capacity, after_capacity = truth - stable, truth - stable
    recovery_capacity = after_exact - before_exact
    for index, d in decisions.items():
        old_relation, new_relation = relation(d.original, truth), relation(d.refined, truth)
        condition["baseline_relations"][old_relation] += 1
        condition["anchored_relations"][new_relation] += 1
        if d.status != "replaced":
            counts["unchanged_predictions"] += 1
            counts["unchanged_datetime_anchors"] += 1
            if d.status == "rejected":
                counts["rejected_refinements"] += 1
                counts["ambiguous_refinements"] += d.reason.startswith("ambiguous")
                condition["rejections"][d.reason] += 1
            continue
        if original[index].label != revised[index].label:
            raise AnchoredError("replacement label failed")
        counts["replacements"] += 1
        counts["changed_predictions"] += 1
        condition["changed_original_relations"][old_relation] += 1
        condition["changed_relations"][new_relation] += 1
        effects = condition["effects"]
        matched_before = bool(before_capacity[d.original])
        matched_after = bool(after_capacity[d.refined])
        if matched_before:
            before_capacity[d.original] -= 1
        if matched_after:
            after_capacity[d.refined] -= 1
        effects["changed_predictions_tp"] += matched_after
        effects["changed_predictions_fp"] += not matched_after
        effects["changed_became_fp_from_tp"] += matched_before and not matched_after
        if recovery_capacity[d.refined]:
            effects["changed_recovered_fn_instances"] += 1
            recovery_capacity[d.refined] -= 1
        exact_before = old_relation == "exact_match"
        exact_after = new_relation == "exact_match"
        effects["changed_boundary_exact"] += exact_after
        effects["changed_boundary_nonexact"] += not exact_after
        effects["nonexact_to_exact"] += not exact_before and exact_after
        effects["nonexact_to_nonexact"] += not exact_before and not exact_after
        effects["exact_to_nonexact"] += exact_before and not exact_after
        inside_before = old_relation == "candidate_contained_by_gt"
        inside_after = new_relation == "candidate_contained_by_gt"
        effects["introduced_inside_gt"] += not inside_before and inside_after
        effects["resolved_inside_gt"] += inside_before and not inside_after
    counts["unchanged_predictions"] += len(original) - len(decisions)
    counts["unchanged_non_datetime_predictions"] += len(original) - len(decisions)


def run(
    normalized: Path, predictions: Sequence[Path], previous_report: Path,
    fingerprints: Path, dataset: str, progress: int = 5000,
) -> dict[str, Any]:
    started = time.perf_counter()
    mapping = load_label_map()
    sources = [normalized, previous_report, fingerprints, *predictions]
    hashes = {str(p): sha256_file(p) for p in sources}
    previous = next(d for d in json.loads(previous_report.read_text())["datasets"]
                    if d["dataset"] == dataset)
    pinned = next(d for d in json.loads(fingerprints.read_text())["datasets"]
                  if d["dataset"] == dataset)
    if (
        len(predictions) != 2 or previous["normalized_sha256"] != hashes[str(normalized)]
        or pinned["normalized_sha256"] != hashes[str(normalized)]
    ):
        raise AnchoredError("dataset identity failed")
    conditions, headers = [], []
    for path in predictions:
        header = read_header(path)
        threshold = header.get("threshold")
        expected = next(c for c in pinned["conditions"] if c["threshold"] == threshold)
        prior = next(c for c in previous["conditions"] if c["threshold"] == threshold)
        if (
            isinstance(threshold, bool) or threshold not in (0.3, 0.7)
            or hashes[str(path)] != expected["input_predictions_sha256"]
            or hashes[str(path)] != prior["prediction_sha256"]
            or header.get("normalized_sha256") != hashes[str(normalized)]
            or header["model"]["model_id"] != mapping.model_id
            or header["model"]["model_revision"] != mapping.model_revision
            or header["model"]["max_width"] != 24
        ):
            raise AnchoredError("prediction identity failed")
        headers.append({k: v for k, v in header.items() if k != "threshold"})
        conditions.append({
            "path": path, "threshold": threshold, "previous": prior, "pending": [],
            "states": {(s, v): EvaluationState() for s in (False, True)
                       for v in ("baseline", "variant")},
            **{key: Counter() for key in (
                "accounting", "effects", "rejections", "baseline_relations",
                "anchored_relations", "changed_original_relations", "changed_relations",
            )},
        })
    if len({c["threshold"] for c in conditions}) != 2 or headers[0] != headers[1]:
        raise AnchoredError("threshold identity failed")
    count = 0
    streams = [_normalized_records(normalized), *(iter_records(c["path"]) for c in conditions)]
    for index, (row, *records) in enumerate(zip_longest(*streams)):
        if row is None or any(p is None or p.example_index != index for p in records):
            raise AnchoredError("record alignment failed")
        text, truth = validate_normalized_example(row)
        _validate_native(row, text, mapping)
        # All decisions finish before any GT-dependent diagnostic is called.
        views = [refine(text, record.predictions, mapping) for record in records]
        date_truth = Counter(Span(e.start, e.end) for e in truth if e.entity_type == TARGET)
        for c, record, (original, revised, decisions) in zip(
            conditions, records, views, strict=True
        ):
            if original != tuple(record.predictions):
                raise AnchoredError("original preservation failed")
            before = tuple(to_gateway(original, mapping).kept)
            after = tuple(to_gateway(revised, mapping).kept)
            c["pending"].append((index, truth, before, after))
            diagnose(original, revised, decisions, date_truth, c)
            if len(c["pending"]) >= 64:
                frozen.flush(c)
        count += 1
        if progress > 0 and count % progress == 0:
            print(json.dumps({"event": "progress", "examples": count,
                              "elapsed_seconds": round(time.perf_counter() - started, 3)}),
                  flush=True)
    if not count or any(sha256_file(p) != hashes[str(p)] for p in sources):
        raise AnchoredError("input integrity failed")
    outputs = []
    for c in sorted(conditions, key=lambda item: item["threshold"]):
        frozen.flush(c)
        output = {key: dict(c[key]) for key in ("accounting", "effects", "rejections")}
        for key in ("baseline_relations", "anchored_relations", "changed_original_relations",
                    "changed_relations"):
            output[key] = {r: c[key][r] for r in RELATIONS}
        output["threshold"] = c["threshold"]
        output["prediction_sha256"] = hashes[str(c["path"])]
        for supported, view in ((False, "full_strict"), (True, "gateway_supported_only_strict")):
            before, after = (c["states"][supported, v] for v in ("baseline", "variant"))
            output[view] = {
                "baseline": _strict_summary(before), "anchored": _strict_summary(after),
                "previous": c["previous"][view]["variant"],
            }
            _same_metrics(output[view]["baseline"], c["previous"][view]["baseline"])
            output[view]["delta_vs_baseline"] = _delta(
                output[view]["baseline"], output[view]["anchored"]
            )
            for label in set(before.entity_types()) | set(after.entity_types()):
                if label != TARGET and any(
                    getattr(before, f)[label] != getattr(after, f)[label]
                    for f in ("true_positives", "false_positives", "false_negatives")
                ):
                    raise AnchoredError("non-target metric preservation failed")
        output["datetime"] = {
            "baseline": frozen.metrics(c["states"][False, "baseline"]),
            "previous": c["previous"]["datetime"]["variant"],
            "anchored": frozen.metrics(c["states"][False, "variant"]),
        }
        _same_metrics(output["datetime"]["baseline"], c["previous"]["datetime"]["baseline"])
        for comparator in ("baseline", "previous"):
            output["datetime"]["delta_vs_" + comparator] = _delta(
                output["datetime"][comparator], output["datetime"]["anchored"]
            )
        delta = output["datetime"]["delta_vs_baseline"]
        accounting = c["accounting"]
        if (
            delta["tp"] != c["effects"]["recovered_fn_gross"] - c["effects"]["lost_tp_gross"]
            or delta["fp"] != -delta["tp"] or delta["fn"] != -delta["tp"]
            or accounting["original_model_predictions"] != accounting["refined_view_predictions"]
            or accounting["original_model_predictions"]
            != accounting["replacements"] + accounting["unchanged_predictions"]
        ):
            raise AnchoredError("replacement or recovery accounting failed")
        output["subspan_comparison"] = {
            "previous_appended_inside_gt":
                c["previous"]["core_relations"].get("candidate_contained_by_gt", 0),
            "anchored_changed_inside_gt":
                c["changed_relations"]["candidate_contained_by_gt"],
            "introduced_inside_gt": c["effects"]["introduced_inside_gt"],
            "resolved_inside_gt": c["effects"]["resolved_inside_gt"],
            "all_baseline_inside_gt": c["baseline_relations"]["candidate_contained_by_gt"],
            "all_anchored_inside_gt": c["anchored_relations"]["candidate_contained_by_gt"],
            "denominators": "previous additions versus anchored replacements; no append population",
        }
        for key in ("replacements", "changed_predictions", "unchanged_predictions",
                    "rejected_refinements", "deleted_predictions", "added_predictions",
                    "unchanged_datetime_anchors", "unchanged_non_datetime_predictions",
                    "ambiguous_refinements"):
            output["accounting"].setdefault(key, 0)
        for key in ("recovered_fn_gross", "lost_tp_gross", "changed_predictions_tp",
                    "changed_predictions_fp", "changed_became_fp_from_tp",
                    "changed_recovered_fn_instances", "changed_boundary_exact",
                    "changed_boundary_nonexact", "nonexact_to_exact", "nonexact_to_nonexact",
                    "exact_to_nonexact", "introduced_inside_gt", "resolved_inside_gt"):
            output["effects"].setdefault(key, 0)
        output["invariants"] = {
            "original_records_preserved": True, "one_to_one_replacement_trace": True,
            "non_datetime_predictions_unchanged": True, "no_unanchored_candidates": True,
        }
        outputs.append(output)
    return {
        "dataset": dataset, "examples": count, "conditions": outputs,
        "normalized_sha256": hashes[str(normalized)],
        "previous_report_sha256": hashes[str(previous_report)],
        "elapsed_seconds": time.perf_counter() - started, "inputs_unchanged": True,
        "no_inference": True, "radius": RADIUS, "maximum_anchor_characters": MAX_ANCHOR,
        "privacy": "aggregates only; in-memory traces are never exported",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, action="append", required=True)
    parser.add_argument("--previous-report", type=Path, required=True)
    parser.add_argument("--input-reference", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=5000)
    args = parser.parse_args(argv)
    if args.report.exists() or is_within_repository(args.report):
        print("error: report must be a new external path", file=sys.stderr)
        return 2
    try:
        result = run(args.normalized, args.predictions, args.previous_report,
                     args.input_reference, args.dataset, args.progress_every)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except (ValueError, TypeError, AttributeError, KeyError, StopIteration, OSError):
        print("error: anchored experiment failed input or invariant checks", file=sys.stderr)
        return 2
    print(json.dumps({"event": "complete", "examples": result["examples"],
                      "elapsed_seconds": round(result["elapsed_seconds"], 3)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())