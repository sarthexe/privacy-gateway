from __future__ import annotations

import inspect
import json
from collections import Counter
from pathlib import Path

import pytest

from scripts.analyze_gliner_ood_errors import analyze
from scripts.evaluation_checkpoint import sha256_file
from scripts.gliner_evaluation import RawPrediction, load_label_map
from scripts.gliner_location_anchored_refinement import (
    MAX_ANCHOR,
    RADIUS,
    LocationError,
    decide,
    diagnose,
    main,
    pair_counts,
    refine,
    run,
)
from scripts.gliner_phone_hybrid_experiment import Span
from scripts.gliner_predictions import PredictionRecord, build_header, encode_record


@pytest.mark.parametrize("text,expected", [
    ("(Rivergate),", "Rivergate"), ("[Rivergate]", "Rivergate"),
    ('"Rivergate"', "Rivergate"), ("‘Rivergate’", "Rivergate"),
    (" Rivergate! ", "Rivergate"), ("Rivergate)))", "Rivergate"),
    ("((Port Eastmere))", "Port Eastmere"), ("{Rivergate}", "Rivergate"),
])
def test_peripheral_punctuation(text: str, expected: str) -> None:
    d = decide(text, Span(0, len(text)))
    assert d.status == "replaced"
    assert text[d.refined.start:d.refined.end] == expected


@pytest.mark.parametrize("text,start,end", [
    ("Rivergate", 0, 5), ("Rivergate", 5, 9),
    ("East-Vale", 5, 9), ("East‑Vale", 0, 4),
    ("O’Vale", 2, 6), ("O'Vale", 0, 3),
    ("Port Eastmere", 0, 9), ("rivergate", 0, 5),
    ("Rivéregate", 0, 5),
])
def test_same_lexical_token_recovery(text: str, start: int, end: int) -> None:
    d = decide(text, Span(start, end))
    assert d.status == "replaced" and d.refined == Span(0, len(text))


@pytest.mark.parametrize("text", [
    "Port Eastmere", "Rivergate, Westmere", "Rivergate (Westmere)",
    "Rivergate's", "Port “Eastmere”",
])
def test_existing_multi_token_structure_preserved(text: str) -> None:
    d = decide(text, Span(0, len(text)))
    assert d.refined == d.original


def test_existing_internal_parenthetical_closer_only() -> None:
    text = "Rivergate (Westmere)"
    d = decide(text, Span(0, len(text) - 1))
    assert d.status == "replaced" and d.refined == Span(0, len(text))
    d = decide(text, Span(0, text.index("West") + 4))
    # Word-internal completion then an immediate structural closer is supported.
    assert d.refined == Span(0, len(text))


@pytest.mark.parametrize("text,end", [
    ("Rivergate's", 9), ("Rivergate’s", 9),
    ("Rivergate (Aro Nex)", 9), ("Rivergate, Eastmere", 9),
    ("Rivergate (West Mere)", len("Rivergate (West")),
    ("Rivergate/Eastmere", 9), ("Rivergate–Eastmere", 9),
    ("Rivergate_Station", 9), ("Rivergate123", 9),
    ("R. Eastmere", 2), ("Rivergate.", 10),
])
def test_ambiguous_expansion_rejected(text: str, end: int) -> None:
    d = decide(text, Span(0, end))
    assert d.status == "rejected" and d.reason.startswith("ambiguous")
    assert d.refined == d.original


@pytest.mark.parametrize("text", [
    '["Rivergate"]', "['Rivergate', 'Eastmere']", "{'city':'Rivergate'}",
])
def test_structured_collections_not_trimmed_into_names(text: str) -> None:
    d = decide(text, Span(0, len(text)))
    assert d.status == "rejected" and d.refined == d.original


@pytest.mark.parametrize("label,neighbor", [("first_name", "Aro"), ("company_name", "ForgeLabs")])
def test_nearby_person_organization_not_absorbed_or_suppressed(label: str, neighbor: str) -> None:
    text = "Rivergate " + neighbor
    place = RawPrediction("city", 0, 9, 0.8)
    other = RawPrediction(label, 10, len(text), 0.9)
    original, revised, decisions = refine(text, [place, other])
    assert original == revised and revised[0] is place and revised[1] is other
    assert decisions[0].status == "unchanged"


def test_no_anchor_no_detection() -> None:
    p = RawPrediction("first_name", 0, 3, 0.8)
    original, revised, decisions = refine("Aro at Rivergate", [p])
    assert original == revised and not decisions


def test_aliases_duplicates_one_to_one_trace_and_original_preservation() -> None:
    text = "(Rivergate)"
    p = RawPrediction("city", 0, len(text), 0.83)
    q = RawPrediction("state", 1, 10, 0.74)
    original, revised, decisions = refine(text, [p, q, p])
    assert original[0] is p and original[2] is p
    assert len(original) == len(revised) == len(decisions) == 3
    assert [(v.start, v.end) for v in revised] == [(1, 10)] * 3
    assert revised[0].label == p.label and revised[0].score == p.score
    assert revised[1] is q


def counters():
    return {k: Counter() for k in (
        "accounting", "effects", "truncation", "rejections", "baseline_pairs",
        "refined_pairs", "baseline_relations", "refined_relations", "changed_relations",
    )}


def test_truncation_cohort_recovery_and_capacity_accounting() -> None:
    original, revised, decisions = refine("Rivergate", [RawPrediction("city", 0, 5, 0.9)])
    c = counters()
    diagnose(original, revised, decisions, Counter({Span(0, 9): 1}), c)
    assert c["truncation"]["converted_to_original_longer_gt_exact"] == 1
    assert c["truncation"]["cohort_fn_recovered"] == 1
    assert c["baseline_pairs"]["prediction_contained_by_gt"] == 1
    assert c["refined_pairs"]["prediction_contained_by_gt"] == 0
    assert c["effects"]["recovered_fn_gross"] == 1


def test_duplicate_boundary_exact_does_not_invent_recovered_fn() -> None:
    original, revised, decisions = refine("(Rivergate)", [
        RawPrediction("city", 0, 11, 0.9), RawPrediction("city", 1, 10, 0.8),
    ])
    c = counters()
    diagnose(original, revised, decisions, Counter({Span(1, 10): 1}), c)
    assert c["effects"]["recovered_fn_gross"] == 0
    assert c["effects"]["changed_fp_instances"] == 1


def test_nonexclusive_pairs_preserve_multiplicity() -> None:
    p = Counter({Span(2, 4): 2})
    gt = Counter({Span(0, 6): 3})
    assert pair_counts(p, gt)["prediction_contained_by_gt"] == 6


def test_gt_independence_and_unicode_boundaries() -> None:
    assert list(inspect.signature(decide).parameters) == ["text", "anchor"]
    text = "測試 (Rivergate)"
    d = decide(text, Span(3, len(text)))
    assert text[d.refined.start:d.refined.end] == "Rivergate"
    with pytest.raises(TypeError):
        decide(text, Span(4, 13), [])  # type: ignore[call-arg]


def test_bounded_neighborhood_and_anchor_limit() -> None:
    text = "A" * (RADIUS + 40)
    d = decide(text, Span(0, 4))
    assert d.status == "rejected" and d.reason == "neighborhood_edge"
    assert decide("A" * (MAX_ANCHOR + 1), Span(0, MAX_ANCHOR + 1)).reason == "anchor_size_limit"


@pytest.mark.parametrize("bad", [None, 42, b"Rivergate"])
def test_malformed_source(bad) -> None:
    with pytest.raises(LocationError):
        refine(bad, [])  # type: ignore[arg-type]


@pytest.mark.parametrize("p", [
    RawPrediction("city", -1, 3, 0.9), RawPrediction("city", 0, 99, 0.9),
    RawPrediction("city", 0, 3, float("nan")), RawPrediction("unknown", 0, 3, 0.9),
])
def test_malformed_predictions(p: RawPrediction) -> None:
    with pytest.raises(LocationError):
        refine("Rivergate", [p])


def fixture(tmp: Path):
    mapping = load_label_map()
    rows = [
        ("At (Rivergate), with Aro.", (4, 13), (3, 15)),
        ("Port Eastmere", (0, 13), (0, 9)),
        ("Rivergate (Westmere)", (0, 20), (0, 19)),
    ]
    norm = tmp / "normalized.jsonl"
    normalized = []
    for text, gt, _ in rows:
        native = {"native_label": "city", "start": gt[0], "end": gt[1]}
        normalized.append({"text": text, "native_entities": [native],
                           "entities": [{"type": mapping.mapping["city"],
                                         "start": gt[0], "end": gt[1]}]})
    norm.write_text("".join(json.dumps(r) + "\n" for r in normalized))
    paths, baselines, pins = [], [], []
    for threshold in (0.3, 0.7):
        path = tmp / f"pred-{threshold}.jsonl"
        path.write_text(build_header({
            "threshold": threshold, "normalized_sha256": sha256_file(norm),
            "model": {"model_id": mapping.model_id, "model_revision": mapping.model_revision,
                      "max_width": 24},
        }) + "\n" + "".join(encode_record(PredictionRecord(
            i, 1, 0, (RawPrediction("city", pred[0], pred[1], 0.9),),
        )) + "\n" for i, (_, _, pred) in enumerate(rows)))
        base = analyze(norm, path)
        pairs = Counter()
        for _, gt, pred in rows:
            pairs.update(pair_counts(Counter({Span(*pred): 1}), Counter({Span(*gt): 1})))
        baselines.append({
            "threshold": threshold, "full_strict": base["full_strict"],
            "gateway_supported_only_strict": base["gateway_supported_only_strict"],
            "inventory": [{"type": "LOCATION", **base["per_entity_type"]["LOCATION"]}],
            "span_behavior": {"LOCATION": dict(pairs)},
        })
        pins.append({"threshold": threshold, "input_predictions_sha256": sha256_file(path)})
        paths.append(path)
    reference, fingerprints = tmp / "reference.json", tmp / "pins.json"
    reference.write_text(json.dumps({
        "datasets": [{"dataset": "synthetic", "conditions": baselines}],
    }))
    fingerprints.write_text(json.dumps({"datasets": [{
        "dataset": "synthetic", "normalized_sha256": sha256_file(norm), "conditions": pins,
    }]}))
    return norm, paths, reference, fingerprints


def test_six_view_scoring_invariants_and_aggregate_privacy(tmp_path: Path) -> None:
    norm, paths, reference, pins = fixture(tmp_path)
    hashes = [sha256_file(p) for p in [norm, *paths, reference, pins]]
    result = run(norm, paths, reference, pins, "synthetic", 0)
    for c in result["conditions"]:
        assert c["location"]["delta"]["tp"] == 3
        assert c["location"]["delta"]["fp"] == -3
        assert c["location"]["delta"]["fn"] == -3
        assert c["accounting"]["replacements"] == 3
        assert c["truncation"]["cohort_fn_recovered"] == 2
    encoded = json.dumps(result)
    for forbidden in ('"start"', '"end"', '"text"', '"example_index"', "Rivergate", "Westmere"):
        assert forbidden not in encoded
    assert hashes == [sha256_file(p) for p in [norm, *paths, reference, pins]]


def test_hash_drift_and_no_overwrite(tmp_path: Path) -> None:
    norm, paths, reference, pins = fixture(tmp_path)
    assert main([
        "--normalized", str(norm), "--predictions", str(paths[0]),
        "--predictions", str(paths[1]), "--baseline-report", str(reference),
        "--input-reference", str(pins), "--dataset", "synthetic", "--report", str(norm),
    ]) == 2
    paths[0].write_text(paths[0].read_text() + "\n")
    with pytest.raises(LocationError):
        run(norm, paths, reference, pins, "synthetic", 0)