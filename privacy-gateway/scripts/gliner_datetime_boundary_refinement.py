"""Append-only source-structural DATE_TIME boundary experiment; no inference."""

from __future__ import annotations

import argparse
import calendar
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

from scripts.analyze_gliner_ood_errors import _normalized_records, _strict_summary, _validate_native
from scripts.batch_evaluation import Batch, score_batch
from scripts.evaluation_checkpoint import EvaluationState, sha256_file
from scripts.gliner_evaluation import GlinerLabelMap, RawPrediction, load_label_map, to_gateway
from scripts.gliner_phone_boundary_refinement import _same_metrics
from scripts.gliner_phone_hybrid_experiment import Span, _delta, classify_candidate, overlaps
from scripts.gliner_predictions import iter_records, read_header
from scripts.nemotron_pii import is_within_repository
from scripts.presidio_evaluation import SUPPORTED_TYPES, validate_normalized_example

TARGET = "DATE_TIME"
MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
MONTHS = {
    token.lower(): month
    for month, name in enumerate(MONTH_NAMES, 1)
    for token in (name, name[:3])
}
MONTH = "(?:" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\.?"
DAY = r"[0-9]{1,2}(?:st|nd|rd|th)?"
YEAR = r"[0-9]{4}"
DATE = (
    rf"(?:{YEAR}[-/.][0-9]{{1,2}}[-/.][0-9]{{1,2}}"
    rf"|[0-9]{{1,2}}[-/.][0-9]{{1,2}}[-/.]{YEAR}"
    rf"|{DAY}[ \t]+{MONTH}[ \t]+{YEAR}"
    rf"|{MONTH}[ \t]+{DAY},?[ \t]+{YEAR})"
)
AMPM = r"(?:[ap]m|[ap]\.m\.)"
TIME = (
    rf"(?:[0-9]{{1,2}}:[0-9]{{2}}(?::[0-9]{{2}}(?:\.[0-9]{{1,9}})?)?"
    rf"(?:[ \t]*{AMPM})?|[0-9]{{1,2}}[ \t]*{AMPM})"
)
ZONES = "UTC|GMT|PST|PDT|EST|EDT|CST|CDT|MST|MDT|CET|CEST|EET|EEST|IST|JST|AEST|AEDT|ACST|ACDT"
OFFSET = r"[+-][0-9]{2}:?[0-9]{2}"
ZONE = rf"(?:(?-i:Z|{ZONES})(?:{OFFSET})?|{OFFSET})"
CORE = rf"(?:{DATE}(?:[Tt]|[ \t]+){TIME}(?:[ \t]*{ZONE})?|{TIME}(?:[ \t]*{ZONE})?|{DATE})"
PATTERN = re.compile(rf"(?<!\w){CORE}(?![\w:])", re.IGNORECASE)
DATE_RE = re.compile(DATE, re.IGNORECASE)
TIME_RE = re.compile(TIME, re.IGNORECASE)
ZONE_RE = re.compile(ZONE)


class DateExperimentError(ValueError):
    """Fixed source-independent errors."""


@dataclass(frozen=True)
class Candidate:
    raw: Span
    core: Span
    kind: str


def validate_date(value: str) -> str | None:
    """Return a rejection reason; never resolve numeric order using locale or GT."""
    if re.fullmatch(r"[0-9./-]+", value):
        pieces = re.split(r"[-/.]", value)
        separators = re.findall(r"[-/.]", value)
        if len(pieces) != 3 or len(set(separators)) != 1:
            return "malformed_date"
        if len(pieces[0]) == 4:
            year, month, day = map(int, pieces)
        elif len(pieces[2]) == 4:
            first, second, year = map(int, pieces)
            if 1 <= first <= 12 and 1 <= second <= 12 and first != second:
                return "ambiguous_numeric_order"
            if first > 12:
                day, month = first, second
            else:
                month, day = first, second
        else:
            return "malformed_date"
    else:
        tokens = re.findall(r"[A-Za-z]+|[0-9]+", value)
        names = [x for x in tokens if x.lower() in MONTHS]
        numbers = [int(x) for x in tokens if x.isdecimal()]
        if len(names) != 1 or len(numbers) != 2:
            return "malformed_date"
        day, year = numbers
        month = MONTHS[names[0].lower()]
        ordinal = re.search(r"[0-9]+(st|nd|rd|th)", value, re.IGNORECASE)
        suffix = "th" if 10 <= day % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
        if ordinal and ordinal[1].lower() != suffix:
            return "malformed_ordinal"
    if not 1 <= year <= 9999 or not 1 <= month <= 12:
        return "invalid_calendar_date"
    if not 1 <= day <= calendar.monthrange(year, month)[1]:
        return "invalid_calendar_date"
    return None


def validate_time(value: str) -> str | None:
    marker = re.search(rf"{AMPM}$", value, re.IGNORECASE)
    clock = value[:marker.start()].strip() if marker else value
    parts = clock.split(":")
    hour = int(parts[0])
    if not (1 <= hour <= 12 if marker else 0 <= hour <= 23):
        return "invalid_clock"
    if len(parts) > 1 and not 0 <= int(parts[1]) <= 59:
        return "invalid_clock"
    if len(parts) > 2 and not 0 <= int(parts[2].split(".")[0]) <= 59:
        return "invalid_clock"
    return None


def validate_core(value: str) -> tuple[str | None, str]:
    date_match = DATE_RE.match(value)
    clock_start = 0
    if date_match:
        reason = validate_date(date_match[0])
        if reason:
            return reason, "calendar_date"
        if date_match.end() == len(value):
            return None, "calendar_date"
        gap = re.match(r"(?:[Tt]|[ \t]+)", value[date_match.end():])
        if not gap:
            return "malformed_attachment", "date_time"
        clock_start = date_match.end() + gap.end()
    clock = TIME_RE.match(value, clock_start)
    if clock is None:
        return "malformed_clock", "time"
    reason = validate_time(clock[0])
    if reason:
        return reason, "date_time" if date_match else "time"
    tail = value[clock.end():].strip(" \t")
    if tail:
        if not ZONE_RE.fullmatch(tail):
            return "malformed_timezone", "timezone_qualified"
        offset = re.search(r"([+-])([0-9]{2}):?([0-9]{2})$", tail)
        if offset and (int(offset[2]) > 23 or int(offset[3]) > 59):
            return "malformed_timezone", "timezone_qualified"
        return None, "timezone_qualified"
    if date_match:
        return None, "iso_timestamp" if value[date_match.end()] in "Tt" else "date_time"
    return None, "time"


def generate(text: str) -> tuple[tuple[Candidate, ...], Counter[str], dict[str, Counter[str]]]:
    if not isinstance(text, str):
        raise DateExperimentError("source text must be a string")
    stats: Counter[str] = Counter(dict.fromkeys(
        ("raw_candidates", "structurally_valid", "rejected", "refined",
         "unchanged", "ambiguous", "duplicate_candidates_removed", "shadowed_components"), 0
    ))
    classes: dict[str, Counter[str]] = {}
    candidates = []
    for match in PATTERN.finditer(text):
        core = Span(match.start(), match.end())
        reason, kind = validate_core(match[0])
        stats["raw_candidates"] += 1
        part = classes.setdefault(kind, Counter())
        part["raw_candidates"] += 1
        # Never accept a fragment of a longer numeric/clock chain or timezone sequence.
        connected = (
            core.start > 1 and text[core.start - 1] in "/.:-"
            and text[core.start - 2].isdecimal()
            or core.start > 1 and text[core.start - 2:core.start] == "::"
            or core.end + 1 < len(text) and text[core.end] in "/.:-"
            and text[core.end + 1].isdecimal()
        )
        repeated_zone = re.match(rf"[ \t]+{ZONE}(?!\w)", text[core.end:])
        if connected or repeated_zone:
            reason = "ambiguous_attachment"
        if kind == "calendar_date" and re.match(
            r"(?:[Tt]|[ \t]+)[0-9]{1,2}:", text[core.end:]
        ):
            reason = "malformed_attachment"
        if kind != "calendar_date" and re.match(
            r"[ \t]+[A-Z]{2,5}(?!\w)", text[core.end:]
        ):
            reason = "ambiguous_timezone_attachment"
        if reason:
            stats["rejected"] += 1
            part["rejected"] += 1
            stats["rejection_" + reason] += 1
            if reason.startswith("ambiguous"):
                stats["ambiguous"] += 1
                part["ambiguous"] += 1
            continue
        start, end = core.start, core.end
        for _ in range(3):
            if start and text[start - 1] in "([{\"'“‘":
                start -= 1
            if end < len(text) and text[end] in ")]},;.!?\\\"'”’":
                end += 1
        raw = Span(start, end)
        candidate = Candidate(raw, core, kind)
        candidates.append(candidate)
        stats["structurally_valid"] += 1
        part["structurally_valid"] += 1
        status = "refined" if raw != core else "unchanged"
        stats[status] += 1
        part[status] += 1
        if kind in {"date_time", "iso_timestamp", "timezone_qualified"}:
            stats["shadowed_components"] += bool(DATE_RE.match(match[0]))
    unique = tuple(dict.fromkeys(candidates))
    stats["duplicate_candidates_removed"] = len(candidates) - len(unique)
    if stats["raw_candidates"] != stats["structurally_valid"] + stats["rejected"]:
        raise DateExperimentError("candidate accounting failed")
    return unique, stats, classes


def merge(
    text: str, old: Sequence[RawPrediction], candidates: Sequence[Candidate],
    mapping: GlinerLabelMap | None = None,
):
    mapping = mapping or load_label_map()
    original = tuple(old)
    for p in original:
        if (
            not isinstance(p, RawPrediction) or p.label not in mapping.mapping
            or isinstance(p.start, bool) or isinstance(p.end, bool)
            or not isinstance(p.start, int) or not isinstance(p.end, int)
            or not 0 <= p.start < p.end <= len(text)
            or isinstance(p.score, bool) or not isinstance(p.score, int | float)
            or not math.isfinite(p.score)
        ):
            raise DateExperimentError("model prediction is invalid")
    existing = {Span(e.start, e.end) for e in to_gateway(original, mapping).kept
                if e.entity_type == TARGET}
    unique = tuple(dict.fromkeys(candidates))
    for c in unique:
        if (
            not isinstance(c, Candidate)
            or not 0 <= c.raw.start <= c.core.start < c.core.end <= c.raw.end <= len(text)
            or validate_core(text[c.core.start:c.core.end]) != (None, c.kind)
        ):
            raise DateExperimentError("candidate is invalid")
    added = tuple(c for c in unique if c.core not in existing)
    result = original + tuple(
        RawPrediction("date_time", c.core.start, c.core.end, 1.0) for c in added
    )
    if result[:len(original)] != original:
        raise DateExperimentError("model preservation failed")
    return result, added, len(unique) - len(added)


def flush(condition: dict[str, Any]) -> None:
    rows = condition["pending"]
    if not rows:
        return
    batch = Batch([(row[0], b"") for row in rows], 0, False)
    for supported in (False, True):
        for i, version in enumerate(("baseline", "variant")):
            pairs = [
                tuple(tuple((e.entity_type, e.start, e.end) for e in entities
                            if not supported or e.entity_type in SUPPORTED_TYPES)
                      for entities in (row[1], row[i + 2]))
                for row in rows
            ]
            score_batch(condition["states"][supported, version], batch, pairs)
    rows.clear()


def metrics(state: EvaluationState) -> dict[str, Any]:
    part = EvaluationState()
    for field in ("true_positives", "false_positives", "false_negatives"):
        getattr(part, field)[TARGET] = getattr(state, field)[TARGET]
    return _strict_summary(part)


def run(normalized: Path, predictions: Sequence[Path], reference: Path,
        fingerprints: Path, dataset: str, progress: int = 5000) -> dict[str, Any]:
    started = time.perf_counter()
    mapping = load_label_map()
    sources = [normalized, reference, fingerprints, *predictions]
    hashes = {str(p): sha256_file(p) for p in sources}
    frozen = next(d for d in json.loads(reference.read_text())["datasets"]
                  if d["dataset"] == dataset)
    pinned = next(d for d in json.loads(fingerprints.read_text())["datasets"]
                  if d["dataset"] == dataset)
    if len(predictions) != 2 or pinned["normalized_sha256"] != hashes[str(normalized)]:
        raise DateExperimentError("fixed dataset identity failed")
    conditions = []
    headers = []
    for path in predictions:
        header = read_header(path)
        threshold = header.get("threshold")
        old = next(c for c in frozen["conditions"] if c["threshold"] == threshold)
        expected = next(c for c in pinned["conditions"] if c["threshold"] == threshold)
        if (
            isinstance(threshold, bool) or threshold not in (0.3, 0.7)
            or hashes[str(path)] != expected["input_predictions_sha256"]
            or header.get("normalized_sha256") != hashes[str(normalized)]
            or header["model"]["model_id"] != mapping.model_id
            or header["model"]["model_revision"] != mapping.model_revision
            or header["model"]["max_width"] != 24
        ):
            raise DateExperimentError("fixed prediction identity failed")
        headers.append({k: v for k, v in header.items() if k != "threshold"})
        conditions.append({
            "path": path, "threshold": threshold, "expected": old, "pending": [],
            "states": {(s, v): EvaluationState() for s in (False, True)
                       for v in ("baseline", "variant")},
            "accounting": Counter(), "raw_relations": Counter(), "core_relations": Counter(),
            "recovery": Counter(), "class_duplicates": Counter(),
        })
    if len({c["threshold"] for c in conditions}) != 2 or headers[0] != headers[1]:
        raise DateExperimentError("threshold identity failed")
    total = Counter()
    class_totals: dict[str, Counter[str]] = {}
    streams = [_normalized_records(normalized), *(iter_records(c["path"]) for c in conditions)]
    for index, (row, *records) in enumerate(zip_longest(*streams)):
        if row is None or any(p is None or p.example_index != index for p in records):
            raise DateExperimentError("input alignment failed")
        text, truth = validate_normalized_example(row)
        _validate_native(row, text, mapping)
        candidates, counts, classes = generate(text)
        total.update(counts)
        total["examples"] += 1
        for kind, values in classes.items():
            class_totals.setdefault(kind, Counter()).update(values)
        gt = Counter(Span(e.start, e.end) for e in truth if e.entity_type == TARGET)
        for condition, record in zip(conditions, records, strict=True):
            variant, added, duplicates = merge(text, record.predictions, candidates, mapping)
            before = tuple(to_gateway(record.predictions, mapping).kept)
            after = tuple(to_gateway(variant, mapping).kept)
            condition["pending"].append((index, truth, before, after))
            baseline_counts = Counter(
                Span(e.start, e.end) for e in before if e.entity_type == TARGET
            )
            missing = gt - baseline_counts
            accounting = condition["accounting"]
            accounting.update({
                "model_predictions_preserved": len(record.predictions),
                "candidate_additions": len(added), "exact_duplicates": duplicates,
                "deleted_model_predictions": 0, "changed_model_predictions": 0,
            "changed_non_datetime_predictions": 0, "records_verified": 1,
            })
            for c in candidates:
                if c.core in baseline_counts:
                    condition["class_duplicates"][c.kind] += 1
            for c in added:
                for field, span in (("raw_relations", c.raw), ("core_relations", c.core)):
                    relation = classify_candidate(span, gt)
                    if relation == "no_phone_gt_overlap":
                        relation = "no_datetime_gt_overlap"
                    condition[field][relation] += 1
                was_missing = bool(missing[c.core])
                condition["recovery"]["recovered_fn" if was_missing else "added_fp"] += 1
                if was_missing:
                    missing[c.core] -= 1
                    boundary = any(overlaps(c.core, span) for span in baseline_counts)
                    category = (
                        "recovered_span_mismatch" if boundary
                        else "recovered_without_datetime_overlap"
                    )
                    condition["recovery"][category] += 1
                    if c.raw != c.core:
                        condition["recovery"]["refined_addition_tp"] += 1
                    if not gt[c.raw]:
                        condition["recovery"]["core_exact_raw_nonexact"] += 1
            if len(condition["pending"]) >= 64:
                flush(condition)
        if progress > 0 and total["examples"] % progress == 0:
            print(json.dumps({
                "event": "progress", "examples": total["examples"],
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }), flush=True)
    if not total["examples"] or any(sha256_file(p) != hashes[str(p)] for p in sources):
        raise DateExperimentError("input integrity failed")
    outputs = []
    for condition in sorted(conditions, key=lambda c: c["threshold"]):
        flush(condition)
        output = {k: dict(condition[k]) for k in (
            "accounting", "raw_relations", "core_relations", "recovery", "class_duplicates"
        )}
        output.update(threshold=condition["threshold"],
                      prediction_sha256=hashes[str(condition["path"])])
        for supported, view in ((False, "full_strict"), (True, "gateway_supported_only_strict")):
            output[view] = {v: _strict_summary(condition["states"][supported, v])
                            for v in ("baseline", "variant")}
            _same_metrics(output[view]["baseline"], condition["expected"][view])
            output[view]["delta"] = _delta(output[view]["baseline"], output[view]["variant"])
            before = condition["states"][supported, "baseline"]
            after = condition["states"][supported, "variant"]
            for label in set(before.entity_types()) | set(after.entity_types()):
                if label != TARGET and any(
                    getattr(before, f)[label] != getattr(after, f)[label]
                    for f in ("true_positives", "false_positives", "false_negatives")
                ):
                    raise DateExperimentError("non-target metric invariant failed")
        output["datetime"] = {v: metrics(condition["states"][False, v])
                              for v in ("baseline", "variant")}
        _same_metrics(output["datetime"]["baseline"],
                      next(m for m in condition["expected"]["inventory"] if m["type"] == TARGET))
        output["datetime"]["delta"] = _delta(
            output["datetime"]["baseline"], output["datetime"]["variant"]
        )
        delta = output["datetime"]["delta"]
        if (
            delta["tp"] != condition["recovery"]["recovered_fn"]
            or delta["fp"] != condition["recovery"]["added_fp"]
            or delta["fn"] != -delta["tp"]
            or sum(condition["core_relations"].values())
            != condition["accounting"]["candidate_additions"]
        ):
            raise DateExperimentError("recovery accounting failed")
        output["invariants"] = {"all_model_predictions_preserved": True,
                                "non_datetime_predictions_unchanged": True,
                                "existing_datetime_predictions_unchanged": True}
        outputs.append(output)
    return {"dataset": dataset, "examples": total["examples"], "candidates": dict(total),
            "candidate_classes": {k: dict(v) for k, v in class_totals.items()},
            "conditions": outputs, "normalized_sha256": hashes[str(normalized)],
            "elapsed_seconds": time.perf_counter() - started, "inputs_unchanged": True,
            "no_inference": True, "privacy": "aggregates only"}


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
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except (ValueError, TypeError, AttributeError, KeyError, StopIteration, OSError):
        print("error: datetime experiment failed input or invariant checks", file=sys.stderr)
        return 2
    print(json.dumps({"event": "complete", "examples": result["examples"],
                      "elapsed_seconds": round(result["elapsed_seconds"], 3)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())