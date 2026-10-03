from __future__ import annotations

import inspect
import json
from collections import Counter
from pathlib import Path

import pytest

from scripts.analyze_gliner_ood_errors import analyze
from scripts.evaluation_checkpoint import sha256_file
from scripts.gliner_datetime_anchored_refinement import (
    MAX_ANCHOR,
    RADIUS,
    AnchoredError,
    decide,
    diagnose,
    main,
    refine,
    run,
)
from scripts.gliner_datetime_boundary_refinement import run as previous_run
from scripts.gliner_evaluation import RawPrediction
from scripts.gliner_phone_hybrid_experiment import Span
from scripts.gliner_predictions import PredictionRecord, encode_record, iter_records
from tests.test_gliner_datetime_boundary_refinement import inputs


@pytest.mark.parametrize("value", [
    "2024-02-29", "February 29, 2024", "12:30", "12:30 PM",
    "12:30 UTC", "12:30+05:30", "2024-02-29T12:30:40.12Z",
])
def test_peripheral_trim(value: str) -> None:
    text = f"({value}),"
    d = decide(text, Span(0, len(text)))
    assert d.status == "replaced" and d.refined == Span(1, 1 + len(value))
    assert text[d.refined.start:d.refined.end] == value


@pytest.mark.parametrize("value", ["12:30 PM", "12:30 UTC", "12:30:45.125+05:30"])
def test_preserve_explicit_clock_attachments(value: str) -> None:
    d = decide(value, Span(0, 5))
    assert d.status == "replaced" and d.refined == Span(0, len(value))


@pytest.mark.parametrize("value", ["12:30Z", "12:30 PM", "2024-02-29T12:30Z"])
def test_complete_values_not_shortened(value: str) -> None:
    d = decide(value, Span(0, len(value)))
    assert d.status == "unchanged" and d.refined == d.original


@pytest.mark.parametrize("text,start,end", [
    ("2024-02-29T12:30", 11, 16),  # no time -> date+time conversion
    ("2024-02-29 12:30", 0, 10),  # no date-only -> composite conversion
    ("2024-02-29", 5, 10),         # no missing-year inference
    ("February 29, 2024", 9, 17),  # no missing-month inference
    ("12:30", 0, 2),              # no missing-minute inference
    ("On 2024-02-29", 0, 13),      # no removal of semantic prose
    ("2024-02-29 99", 0, 13),      # no deletion of digits
])
def test_identity_rejected(text: str, start: int, end: int) -> None:
    d = decide(text, Span(start, end))
    assert d.status == "rejected" and d.refined == d.original


@pytest.mark.parametrize("text", ["03/04/2024", "12:30 and 14:40", "12:30 XYZ"])
def test_ambiguity(text: str) -> None:
    d = decide(text, Span(0, len(text)))
    assert d.status == "rejected" and d.reason.startswith("ambiguous")


def test_no_anchor_no_candidate_and_original_object_identity() -> None:
    text = "Name 2024-02-29"
    p = RawPrediction("first_name", 0, 4, 0.8)
    original, revised, decisions = refine(text, [p])
    assert not decisions and revised == original and revised[0] is p


def test_one_to_one_and_collision_multiplicity() -> None:
    text = "(12:30)"
    p = RawPrediction("time", 0, len(text), 0.87)
    exact = RawPrediction("time", 1, 6, 0.73)
    original, revised, decisions = refine(text, [p, exact, p])
    assert len(original) == len(revised) == len(decisions) == 3
    assert original[0] is p and original[2] is p
    assert [(v.start, v.end) for v in revised] == [(1, 6)] * 3
    assert revised[0].score == 0.87 and revised[1] is exact


def diagnostics(text: str, predictions: list[RawPrediction], truth: Counter[Span]):
    original, revised, decisions = refine(text, predictions)
    c = {k: Counter() for k in (
        "accounting", "effects", "rejections", "baseline_relations",
        "anchored_relations", "changed_original_relations", "changed_relations",
    )}
    diagnose(original, revised, decisions, truth, c)
    return c


def test_valid_timezone_excluded_by_gt_is_not_removed_and_loss_is_counted() -> None:
    c = diagnostics("12:30 UTC", [RawPrediction("time", 0, 5, 0.9)], Counter({Span(0, 5): 1}))
    assert c["effects"]["lost_tp_gross"] == 1
    assert c["effects"]["recovered_fn_gross"] == 0
    assert c["effects"]["changed_became_fp_from_tp"] == 1
    assert c["changed_relations"]["gt_contained_by_candidate"] == 1


def test_duplicate_exact_boundary_does_not_claim_false_recovery() -> None:
    c = diagnostics("(12:30)", [
        RawPrediction("time", 0, 7, 0.9), RawPrediction("time", 1, 6, 0.8),
    ], Counter({Span(1, 6): 1}))
    assert c["effects"]["changed_boundary_exact"] == 1
    assert c["effects"]["changed_predictions_fp"] == 1
    assert c["effects"]["recovered_fn_gross"] == 0


def test_bounded_context_and_anchor_limit() -> None:
    text = "x" * (RADIUS + 3) + " 2024-02-29 " + "x" * (RADIUS + 3)
    d = decide(text, Span(0, 1))
    assert d.status == "rejected" and d.refined == Span(0, 1)
    huge = "x" * (MAX_ANCHOR + 1)
    assert decide(huge, Span(0, len(huge))).reason == "anchor_size_limit"


@pytest.mark.parametrize("text", ["2023-02-29", "12:99", "12:30 XYZ"])
def test_malformed_source_keeps_prediction(text: str) -> None:
    p = RawPrediction("date_time", 0, len(text), 0.9)
    original, revised, decisions = refine(text, [p])
    assert revised == original and revised[0] is p
    assert decisions[0].status == "rejected"


@pytest.mark.parametrize("bad", [None, 42, b"12:30"])
def test_fail_closed_source(bad) -> None:
    with pytest.raises(AnchoredError):
        refine(bad, [])  # type: ignore[arg-type]


def test_no_gt_parameter_and_source_offset_preservation() -> None:
    assert list(inspect.signature(decide).parameters) == ["text", "anchor"]
    text = "測試 (12:30)."
    original, revised, decisions = refine(text, [RawPrediction("time", 3, len(text), 0.8)])
    assert text[revised[0].start:revised[0].end] == "12:30"
    assert original[0].start == 3 and decisions[0].original != decisions[0].refined
    with pytest.raises(TypeError):
        decide(text, Span(4, 9), [])  # type: ignore[call-arg]


@pytest.mark.parametrize("start,end,score", [(-1, 3, 0.8), (0, 100, 0.8), (0, 3, float("nan"))])
def test_fail_closed_prediction(start: int, end: int, score: float) -> None:
    with pytest.raises(AnchoredError):
        refine("12:30", [RawPrediction("time", start, end, score)])


def fixture(tmp: Path):
    norm, paths, reference, fingerprints = inputs(tmp)
    text = json.loads(norm.read_text())["text"]
    start, end = text.index("2024") - 1, text.index("Z") + 2
    refs = json.loads(reference.read_text())
    pins = json.loads(fingerprints.read_text())
    for path, threshold in zip(paths, (0.3, 0.7), strict=True):
        record = next(iter_records(path))
        header = path.read_text().splitlines()[0]
        path.write_text(header + "\n" + encode_record(PredictionRecord(
            0, 1, 0,
            (record.predictions[0], RawPrediction("date_time", start, end, 0.9)),
        )) + "\n")
        base = analyze(norm, path)
        c = next(c for c in refs["datasets"][0]["conditions"] if c["threshold"] == threshold)
        c.update(full_strict=base["full_strict"],
                 gateway_supported_only_strict=base["gateway_supported_only_strict"],
                 inventory=[{"type": "DATE_TIME", **base["per_entity_type"]["DATE_TIME"]}])
        pin = next(c for c in pins["datasets"][0]["conditions"] if c["threshold"] == threshold)
        pin["input_predictions_sha256"] = sha256_file(path)
    reference.write_text(json.dumps(refs))
    fingerprints.write_text(json.dumps(pins))
    previous = tmp / "previous.json"
    previous.write_text(json.dumps({"datasets": [
        previous_run(norm, paths, reference, fingerprints, "synthetic", 0),
    ]}))
    return norm, paths, previous, fingerprints


def test_complete_scoring_replacement_and_privacy(tmp_path: Path) -> None:
    norm, paths, previous, pins = fixture(tmp_path)
    hashes = [sha256_file(p) for p in [norm, *paths, previous, pins]]
    result = run(norm, paths, previous, pins, "synthetic", 0)
    for c in result["conditions"]:
        assert c["datetime"]["delta_vs_baseline"]["tp"] == 1
        assert c["datetime"]["delta_vs_baseline"]["fp"] == -1
        assert c["datetime"]["delta_vs_baseline"]["fn"] == -1
        assert c["accounting"]["replacements"] == 1
        assert c["accounting"]["unchanged_predictions"] == 1
        assert c["accounting"]["deleted_predictions"] == 0
        assert c["changed_relations"]["exact_match"] == 1
        assert c["effects"]["recovered_fn_gross"] == 1
    encoded = json.dumps(result)
    for forbidden in ("2024-02-29", "5:45", '"start"', '"end"', '"example_index"', '"text"'):
        assert forbidden not in encoded
    assert hashes == [sha256_file(p) for p in [norm, *paths, previous, pins]]


def test_reject_hash_drift_and_existing_output(tmp_path: Path) -> None:
    norm, paths, previous, pins = fixture(tmp_path)
    assert main([
        "--normalized", str(norm), "--predictions", str(paths[0]),
        "--predictions", str(paths[1]), "--previous-report", str(previous),
        "--input-reference", str(pins), "--dataset", "synthetic", "--report", str(norm),
    ]) == 2
    paths[0].write_text(paths[0].read_text() + "\n")
    with pytest.raises(AnchoredError):
        run(norm, paths, previous, pins, "synthetic", 0)