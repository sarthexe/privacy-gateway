"""Isolated source-only LOCATION boundary replacements; persisted predictions only."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import unicodedata
from collections import Counter
from collections.abc import Sequence
from itertools import zip_longest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_gliner_ood_errors import _normalized_records, _strict_summary, _validate_native
from scripts.evaluation_checkpoint import EvaluationState, sha256_file
from scripts.gliner_datetime_anchored_refinement import Decision
from scripts.gliner_datetime_boundary_refinement import flush
from scripts.gliner_evaluation import GlinerLabelMap, RawPrediction, load_label_map, to_gateway
from scripts.gliner_phone_boundary_refinement import _same_metrics
from scripts.gliner_phone_hybrid_experiment import Span, _delta, classify_candidate, overlaps
from scripts.gliner_predictions import iter_records, read_header
from scripts.nemotron_pii import is_within_repository
from scripts.presidio_evaluation import evaluation_type, validate_normalized_example

TARGET = "LOCATION"
RADIUS = 32
MAX_ANCHOR = 256
JOINERS = "-‐‑'’ʼ"
OPENERS = "([{"
CLOSERS = ")]}"
QUOTES = "\"'“”‘’"
SEPARATORS = ",;:!?"
RELATIONS = ("exact_match", "prediction_inside_gt", "gt_inside_prediction",
             "partial_overlap", "no_overlap")


class LocationError(ValueError):
    """Source-independent failure messages."""


def letter(ch: str) -> bool:
    return ch.isalpha() or unicodedata.category(ch).startswith("M")


def atom(text: str, index: int) -> bool:
    if not 0 <= index < len(text):
        return False
    ch = text[index]
    return letter(ch) or (
        ch in JOINERS and index > 0 and index + 1 < len(text)
        and letter(text[index - 1]) and letter(text[index + 1])
    )


def decide(text: str, anchor: Span) -> Decision:
    """No GT, dataset, language, capitalisation or other model labels enter here."""
    if (
        not isinstance(text, str) or not isinstance(anchor, Span)
        or any(isinstance(x, bool) or not isinstance(x, int)
               for x in (anchor.start, anchor.end))
        or not 0 <= anchor.start < anchor.end <= len(text)
    ):
        raise LocationError("anchor input is invalid")

    def reject(reason: str) -> Decision:
        return Decision(anchor, anchor, "rejected", reason)

    if anchor.end - anchor.start > MAX_ANCHOR:
        return reject("anchor_size_limit")
    window_start = max(0, anchor.start - RADIUS)
    window_end = min(len(text), anchor.end + RADIUS)
    local = text[window_start:window_end]
    start, end = anchor.start - window_start, anchor.end - window_start
    raw = local[start:end].strip()
    if not any(letter(ch) for ch in raw):
        return reject("unsupported_non_name_anchor")
    # Collections are not enclosing punctuation around one unambiguous name.
    if raw[0] in "[{" and any(ch in raw for ch in ",:;\"'"):
        return reject("ambiguous_structured_container")
    while start < end and local[start].isspace():
        start += 1
    while start < end and local[end - 1].isspace():
        end -= 1
    # Peel wrappers. An apostrophe belonging to a connected lexical token is kept.
    while start < end:
        if local[start] in OPENERS:
            close = CLOSERS[OPENERS.index(local[start])]
            start += 1
            if start < end and local[end - 1] == close:
                end -= 1
        elif local[start] in QUOTES and not atom(local, start):
            opening = local[start]
            closing = {"“": "”", "‘": "’"}.get(opening, opening)
            start += 1
            if start < end and local[end - 1] == closing:
                end -= 1
        elif local[start].isspace():
            start += 1
        else:
            break
    while start < end:
        ch = local[end - 1]
        if ch.isspace() or ch in SEPARATORS:
            end -= 1
        elif ch in CLOSERS:
            opener = OPENERS[CLOSERS.index(ch)]
            body = local[start:end]
            if body.count(opener) >= body.count(ch):
                break  # preserve a closer owned by internal name structure
            end -= 1
        elif ch in QUOTES and not atom(local, end - 1):
            # Do not remove a quote owned by a name-internal quoted component.
            opener = {"”": "“", "’": "‘"}.get(ch, ch)
            if opener in local[start:end - 1]:
                break
            end -= 1
        else:
            break
    if start >= end or not any(letter(ch) for ch in local[start:end]):
        return reject("no_name_core")
    if local[end - 1] == ".":
        return reject("ambiguous_terminal_period")
    # Complete only the same contiguous letter/mark/hyphen/apostrophe token.
    before_start, before_end = start, end
    if atom(local, start):
        while start > 0 and atom(local, start - 1):
            start -= 1
    if atom(local, end - 1):
        while end < len(local) and atom(local, end):
            end += 1
    expanded = start != before_start or end != before_end
    if expanded:
        if sum(letter(ch) for ch in local[before_start:before_end]) < 2:
            return reject("ambiguous_short_token_support")
        if start == 0 and window_start > 0 or end == len(local) and window_end < len(text):
            return reject("neighborhood_edge")
        if re.search(r"['’ʼ]s$", local[start:end], re.IGNORECASE):
            return reject("ambiguous_possessive_completion")
    if (start > 0 and (local[start - 1].isdigit() or local[start - 1] in "_/–—")
            or end < len(local) and (local[end].isdigit() or local[end] in "_/–—")):
        return reject("ambiguous_non_name_attachment")
    # Only close an already model-supported internal qualifier, without new words.
    body = local[start:end]
    balance = 0
    for ch in body:
        if ch == "(":
            balance += 1
        elif ch == ")":
            balance -= 1
        if balance < 0:
            return reject("ambiguous_parenthesis_structure")
    if balance:
        if "(" not in body[1:] or not any(letter(ch) for ch in body.split("(", 1)[1]):
            return reject("ambiguous_parenthesis_structure")
        while balance and end < len(local) and local[end] == ")":
            balance -= 1
            end += 1
        if balance:
            return reject("ambiguous_incomplete_qualifier")
    # Do not convert an external qualifier/possessive into model-supported words.
    tail = local[end:].lstrip(" \t")
    if tail.startswith("(") or re.match(r",\s*\w", tail):
        return reject("ambiguous_external_qualifier")
    if re.match(r"['’ʼ]s(?:\W|$)", tail, re.IGNORECASE):
        return reject("ambiguous_possessive_completion")
    refined = Span(window_start + start, window_start + end)
    if not overlaps(refined, anchor):
        return reject("anchor_overlap_failed")
    if refined == anchor:
        return Decision(anchor, anchor, "unchanged", "already_complete_or_no_supported_extension")
    return Decision(anchor, refined, "replaced", "structural_boundary_only")


def refine(text: str, predictions: Sequence[RawPrediction], mapping: GlinerLabelMap | None = None):
    if not isinstance(text, str):
        raise LocationError("source is invalid")
    mapping = mapping or load_label_map()
    original = tuple(predictions)
    revised, decisions, cache = [], {}, {}
    for index, p in enumerate(original):
        if (
            not isinstance(p, RawPrediction) or p.label not in mapping.mapping
            or any(isinstance(v, bool) or not isinstance(v, int) for v in (p.start, p.end))
            or not 0 <= p.start < p.end <= len(text)
            or isinstance(p.score, bool) or not isinstance(p.score, int | float)
            or not math.isfinite(p.score)
        ):
            raise LocationError("prediction is invalid")
        if evaluation_type(mapping.mapping[p.label]) != TARGET:
            revised.append(p)
            continue
        span = Span(p.start, p.end)
        if span not in cache:
            cache[span] = decide(text, span)
        d = cache[span]
        decisions[index] = d
        revised.append(RawPrediction(p.label, d.refined.start, d.refined.end, p.score)
                       if d.status == "replaced" else p)
    revised = tuple(revised)
    if len(original) != len(revised) or any(
        p.label != q.label or p.score != q.score
        or (i not in decisions or decisions[i].status != "replaced") and p is not q
        for i, (p, q) in enumerate(zip(original, revised, strict=True))
    ):
        raise LocationError("one-to-one preservation failed")
    return original, revised, decisions


def relation(span: Span, truth: Counter[Span]) -> str:
    value = classify_candidate(span, truth)
    return {
        "candidate_contained_by_gt": "prediction_inside_gt",
        "gt_contained_by_candidate": "gt_inside_prediction",
        "no_phone_gt_overlap": "no_overlap",
    }.get(value, value)


def pair_counts(predicted: Counter[Span], truth: Counter[Span]) -> Counter[str]:
    """Reproduce root-cause nonexclusive pairs, including multiplicities."""
    missing, spurious = truth - predicted, predicted - truth
    counts = Counter(exact_match_tp_entities=sum((truth & predicted).values()))
    for gt, multiplicity in truth.items():
        for p, pred_count in predicted.items():
            if gt == p or not overlaps(gt, p) or not (missing[gt] or spurious[p]):
                continue
            count = multiplicity * pred_count
            counts["nonexact_overlap_pairs_total"] += count
            if gt.start <= p.start and p.end <= gt.end:
                counts["prediction_contained_by_gt"] += count
            elif p.start <= gt.start and gt.end <= p.end:
                counts["gt_contained_by_prediction"] += count
            else:
                counts["partial_overlap"] += count
    return counts


def diagnose(original, revised, decisions, truth: Counter[Span], c: dict[str, Any]) -> None:
    a, effects = c["accounting"], c["effects"]
    a.update(original_predictions=len(original), scoring_view_predictions=len(revised),
             original_location_predictions=len(decisions), deleted_predictions=0,
             added_predictions=0, changed_non_location_predictions=0)
    before = Counter(d.original for d in decisions.values())
    after = Counter(d.refined for d in decisions.values())
    old_exact, new_exact = before & truth, after & truth
    recovered = new_exact - old_exact
    effects["recovered_fn_gross"] += sum(recovered.values())
    effects["lost_tp_gross"] += sum((old_exact - new_exact).values())
    c["baseline_pairs"].update(pair_counts(before, truth))
    c["refined_pairs"].update(pair_counts(after, truth))
    stable = Counter(d.original for d in decisions.values() if d.status != "replaced")
    old_capacity, new_capacity = truth - stable, truth - stable
    missing, spurious = truth - before, before - truth
    for d in decisions.values():
        old_relation, new_relation = relation(d.original, truth), relation(d.refined, truth)
        c["baseline_relations"][old_relation] += 1
        c["refined_relations"][new_relation] += 1
        targets = [g for g in truth if g != d.original
                   and g.start <= d.original.start and d.original.end <= g.end
                   and (missing[g] or spurious[d.original])]
        if targets:
            c["truncation"]["baseline_cohort_anchors"] += 1
            exact = d.refined in targets
            c["truncation"]["converted_to_original_longer_gt_exact"] += exact
            c["truncation"]["remaining_nonexact_to_original_longer_gt"] += not exact
            c["truncation"]["changed_cohort_anchors"] += d.status == "replaced"
        if d.status != "replaced":
            a["unchanged_predictions"] += 1
            a["unchanged_location_predictions"] += 1
            if d.status == "rejected":
                a["rejected_refinements"] += 1
                a["ambiguous_refinements"] += d.reason.startswith("ambiguous")
                c["rejections"][d.reason] += 1
            continue
        a["replacements"] += 1
        a["changed_predictions"] += 1
        c["changed_relations"][new_relation] += 1
        matched_before = bool(old_capacity[d.original])
        matched_after = bool(new_capacity[d.refined])
        if matched_before:
            old_capacity[d.original] -= 1
        if matched_after:
            new_capacity[d.refined] -= 1
        effects["changed_tp_instances"] += matched_after
        effects["changed_fp_instances"] += not matched_after
        effects["new_fp_from_previous_tp"] += matched_before and not matched_after
        old_is_exact, new_is_exact = old_relation == "exact_match", new_relation == "exact_match"
        effects["nonexact_to_exact"] += not old_is_exact and new_is_exact
        effects["exact_to_nonexact"] += old_is_exact and not new_is_exact
        effects["nonexact_to_nonexact"] += not old_is_exact and not new_is_exact
        if targets:
            c["truncation"]["changed_cohort_fp_instances"] += not matched_after
            c["truncation"]["cohort_new_fp_from_tp"] += matched_before and not matched_after
        if recovered[d.refined]:
            effects["changed_recovered_fn"] += 1
            if targets:
                c["truncation"]["cohort_fn_recovered"] += 1
            recovered[d.refined] -= 1
    a["unchanged_predictions"] += len(original) - len(decisions)


def metrics(state: EvaluationState) -> dict[str, Any]:
    part = EvaluationState()
    for field in ("true_positives", "false_positives", "false_negatives"):
        getattr(part, field)[TARGET] = getattr(state, field)[TARGET]
    return _strict_summary(part)


def run(normalized: Path, predictions: Sequence[Path], baseline_report: Path,
        fingerprints: Path, dataset: str, progress: int = 5000) -> dict[str, Any]:
    started = time.perf_counter()
    mapping = load_label_map()
    sources = [normalized, baseline_report, fingerprints, *predictions]
    hashes = {str(p): sha256_file(p) for p in sources}
    baseline = next(d for d in json.loads(baseline_report.read_text())["datasets"]
                    if d["dataset"] == dataset)
    pinned = next(d for d in json.loads(fingerprints.read_text())["datasets"]
                  if d["dataset"] == dataset)
    if len(predictions) != 2 or pinned["normalized_sha256"] != hashes[str(normalized)]:
        raise LocationError("dataset identity failed")
    conditions, headers = [], []
    for path in predictions:
        header = read_header(path)
        threshold = header.get("threshold")
        expected = next(c for c in pinned["conditions"] if c["threshold"] == threshold)
        base = next(c for c in baseline["conditions"] if c["threshold"] == threshold)
        if (
            isinstance(threshold, bool) or threshold not in (0.3, 0.7)
            or hashes[str(path)] != expected["input_predictions_sha256"]
            or header.get("normalized_sha256") != hashes[str(normalized)]
            or header["model"]["model_id"] != mapping.model_id
            or header["model"]["model_revision"] != mapping.model_revision
            or header["model"]["max_width"] != 24
        ):
            raise LocationError("prediction identity failed")
        headers.append({k: v for k, v in header.items() if k != "threshold"})
        conditions.append({
            "path": path, "threshold": threshold, "expected": base, "pending": [],
            "states": {(s, v): EvaluationState() for s in (False, True)
                       for v in ("baseline", "variant")},
            **{key: Counter() for key in (
                "accounting", "effects", "truncation", "rejections", "baseline_pairs",
                "refined_pairs", "baseline_relations", "refined_relations", "changed_relations",
            )},
        })
    if len({c["threshold"] for c in conditions}) != 2 or headers[0] != headers[1]:
        raise LocationError("threshold identity failed")
    count = 0
    streams = [_normalized_records(normalized), *(iter_records(c["path"]) for c in conditions)]
    for index, (row, *records) in enumerate(zip_longest(*streams)):
        if row is None or any(p is None or p.example_index != index for p in records):
            raise LocationError("record alignment failed")
        text, truth = validate_normalized_example(row)
        _validate_native(row, text, mapping)
        # Complete source/model decisions before diagnostics touch GT.
        views = [refine(text, r.predictions, mapping) for r in records]
        location_truth = Counter(Span(e.start, e.end) for e in truth if e.entity_type == TARGET)
        for c, record, (original, revised, decisions) in zip(
            conditions, records, views, strict=True
        ):
            if original != tuple(record.predictions):
                raise LocationError("original preservation failed")
            c["pending"].append((index, truth, tuple(to_gateway(original, mapping).kept),
                                 tuple(to_gateway(revised, mapping).kept)))
            diagnose(original, revised, decisions, location_truth, c)
            if len(c["pending"]) >= 64:
                flush(c)
        count += 1
        if progress > 0 and count % progress == 0:
            print(json.dumps({"event": "progress", "examples": count,
                              "elapsed_seconds": round(time.perf_counter() - started, 3)}),
                  flush=True)
    if not count or any(sha256_file(p) != hashes[str(p)] for p in sources):
        raise LocationError("input integrity failed")
    outputs = []
    for c in sorted(conditions, key=lambda item: item["threshold"]):
        flush(c)
        output = {k: dict(c[k]) for k in ("accounting", "effects", "truncation", "rejections")}
        for key in ("baseline_relations", "refined_relations", "changed_relations"):
            output[key] = {r: c[key][r] for r in RELATIONS}
        for key in ("baseline_pairs", "refined_pairs"):
            output[key] = {r: c[key][r] for r in (
                "exact_match_tp_entities", "nonexact_overlap_pairs_total",
                "prediction_contained_by_gt", "gt_contained_by_prediction", "partial_overlap",
            )}
        for key, value in output["baseline_pairs"].items():
            if value != c["expected"]["span_behavior"][TARGET].get(key, 0):
                raise LocationError("baseline boundary-pair reproduction failed")
        output.update(threshold=c["threshold"], prediction_sha256=hashes[str(c["path"])])
        for supported, view in ((False, "full_strict"), (True, "gateway_supported_only_strict")):
            before, after = (c["states"][supported, v] for v in ("baseline", "variant"))
            output[view] = {
                "baseline": _strict_summary(before), "variant_b": _strict_summary(after),
            }
            _same_metrics(output[view]["baseline"], c["expected"][view])
            output[view]["delta"] = _delta(output[view]["baseline"], output[view]["variant_b"])
            for label in set(before.entity_types()) | set(after.entity_types()):
                if label != TARGET and any(getattr(before, f)[label] != getattr(after, f)[label]
                                          for f in ("true_positives", "false_positives",
                                                    "false_negatives")):
                    raise LocationError("non-target metric preservation failed")
        output["location"] = {v: metrics(c["states"][False, key])
                              for v, key in (("baseline", "baseline"), ("variant_b", "variant"))}
        _same_metrics(output["location"]["baseline"],
                      next(m for m in c["expected"]["inventory"] if m["type"] == TARGET))
        output["location"]["delta"] = _delta(output["location"]["baseline"],
                                             output["location"]["variant_b"])
        a, e, delta = c["accounting"], c["effects"], output["location"]["delta"]
        if (
            a["original_predictions"] != a["scoring_view_predictions"]
            or a["original_predictions"] != a["replacements"] + a["unchanged_predictions"]
            or delta["tp"] != e["recovered_fn_gross"] - e["lost_tp_gross"]
            or delta["fp"] != -delta["tp"] or delta["fn"] != -delta["tp"]
        ):
            raise LocationError("replacement/recovery accounting failed")
        for section, keys in (
            ("accounting", ("replacements", "changed_predictions", "unchanged_predictions",
                            "rejected_refinements", "ambiguous_refinements",
                            "unchanged_location_predictions")),
            ("effects", ("recovered_fn_gross", "lost_tp_gross", "changed_recovered_fn",
                         "changed_tp_instances", "changed_fp_instances", "new_fp_from_previous_tp",
                         "nonexact_to_exact", "exact_to_nonexact", "nonexact_to_nonexact")),
            ("truncation", ("baseline_cohort_anchors", "changed_cohort_anchors",
                            "converted_to_original_longer_gt_exact",
                            "remaining_nonexact_to_original_longer_gt",
                            "changed_cohort_fp_instances", "cohort_new_fp_from_tp",
                            "cohort_fn_recovered")),
        ):
            for key in keys:
                output[section].setdefault(key, 0)
        output["invariants"] = {
            "all_original_records_preserved_separately": True,
            "one_to_one_trace_preserving_label_score_order_multiplicity": True,
            "non_location_predictions_unchanged": True, "no_unanchored_additions": True,
        }
        outputs.append(output)
    return {"dataset": dataset, "examples": count, "conditions": outputs,
            "normalized_sha256": hashes[str(normalized)], "inputs_unchanged": True,
            "no_inference": True, "radius": RADIUS, "maximum_anchor_characters": MAX_ANCHOR,
            "elapsed_seconds": time.perf_counter() - started, "privacy": "aggregates only"}


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
        result = run(args.normalized, args.predictions, args.baseline_report,
                     args.input_reference, args.dataset, args.progress_every)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
    except (ValueError, TypeError, AttributeError, KeyError, StopIteration, OSError):
        print("error: location experiment failed input or invariant checks", file=sys.stderr)
        return 2
    print(json.dumps({"event": "complete", "examples": result["examples"],
                      "elapsed_seconds": round(result["elapsed_seconds"], 3)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())