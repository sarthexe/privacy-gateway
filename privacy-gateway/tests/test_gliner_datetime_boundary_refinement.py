from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_gliner_ood_errors import analyze
from scripts.evaluation_checkpoint import sha256_file
from scripts.gliner_datetime_boundary_refinement import (
    Candidate,
    DateExperimentError,
    generate,
    main,
    merge,
    run,
)
from scripts.gliner_evaluation import RawPrediction, load_label_map
from scripts.gliner_phone_hybrid_experiment import Span
from scripts.gliner_predictions import PredictionRecord, build_header, encode_record


@pytest.mark.parametrize("value", [
    "2024-02-29", "2024/2/29", "29/02/2024", "02/29/2024", "03/03/2024",
    "February 29, 2024", "29th Feb 2024", "12:34", "23:59:59",
    "5:45 PM", "5 p.m.", "2024-02-29T12:34:56.123Z",
    "2024-02-29 12:34:56 UTC", "12:34:00+05:30", "12:34 PST",
])
def test_structural_candidates(value: str) -> None:
    candidates, stats, _ = generate(value)
    assert len(candidates) == stats["structurally_valid"] == 1
    assert candidates[0].core == Span(0, len(value))


@pytest.mark.parametrize("value", [
    "2023-02-29", "2024-13-03", "2024-04-31", "2024-02/29",
    "24:30", "12:60", "12:30:61", "13:30 PM", "12:30+25:00",
    "11st May 2024", "2024-01-02T25:00", "2024-02-29 10:30:999",
])
def test_malformed_rejected(value: str) -> None:
    candidates, stats, _ = generate(value)
    assert not candidates and stats["rejected"] >= 1


@pytest.mark.parametrize("value", [
    "03/04/2024", "04-03-2024", "12:30Z UTC", "1/2024-02-29", "12:30 XYZ",
    "2001:db8::12:30",
])
def test_ambiguous_rejected(value: str) -> None:
    candidates, stats, _ = generate(value)
    assert not candidates and stats["ambiguous"] >= 1


@pytest.mark.parametrize("left,right", [("(", "),"), ("[", "]"), ("", ")))"), ('"', '".')])
def test_peripheral_and_unmatched_punctuation(left: str, right: str) -> None:
    value = "2024-02-29T12:34:56Z"
    text = left + value + right
    candidates, stats, _ = generate(text)
    c = candidates[0]
    assert c.core == Span(len(left), len(left) + len(value))
    assert stats["refined"] == 1
    assert text[c.core.start:c.core.end] == value


def test_combination_not_split_and_zone_not_trimmed() -> None:
    text = "2024-02-29T12:34:56+05:30"
    candidates, _, _ = generate(text)
    assert len(candidates) == 1 and candidates[0].core == Span(0, len(text))


def test_unicode_source_boundaries() -> None:
    text = "測試 (2024-02-29)."
    c = generate(text)[0][0]
    assert c.core == Span(4, 14)
    assert text[c.core.start:c.core.end] == "2024-02-29"


def test_no_broad_natural_language_or_gt_argument() -> None:
    assert not generate("tomorrow evening")[0]
    assert not generate("next Tuesday")[0]
    with pytest.raises(TypeError):
        generate("2024-02-29", [])  # type: ignore[call-arg]


def test_exact_duplicate_native_alias_and_candidate_handling() -> None:
    text = "2024-02-29"
    candidates = generate(text)[0]
    old = (RawPrediction("date", 0, len(text), 0.9),)
    result, added, duplicates = merge(text, old, candidates * 2)
    assert result == old and not added and duplicates == 1


def test_all_existing_objects_and_non_target_conflicts_preserved() -> None:
    text = "Name 2024-02-29T12:34Z"
    start = text.index("2024")
    name = RawPrediction("first_name", 0, 4, 0.81)
    partial = RawPrediction("date", start, start + 10, 0.9)
    conflict = RawPrediction("account_number", start, len(text), 0.8)
    old = (name, partial, conflict, name)
    result, added, _ = merge(text, old, generate(text)[0])
    assert result[:len(old)] == old and len(added) == 1
    assert all(a is b for a, b in zip(result[:len(old)], old, strict=True))


def test_fail_closed_input() -> None:
    with pytest.raises(DateExperimentError):
        generate(None)  # type: ignore[arg-type]
    with pytest.raises(DateExperimentError):
        merge("2024-02-29", (RawPrediction("date", -1, 5, 0.9),), ())
    with pytest.raises(DateExperimentError):
        merge("2024-02-29", (), [Candidate(Span(0, 10), Span(0, 10), "time")])


def inputs(tmp: Path):
    text = "At (2024-02-29T10:30:00Z), and 5:45 PM."
    start, clock = text.index("2024"), text.index("5:45")
    native = [
        {"native_label": "first_name", "start": 0, "end": 2},
        {"native_label": "date_time", "start": start, "end": text.index("Z") + 1},
        {"native_label": "time", "start": clock, "end": len(text) - 1},
    ]
    mapping = load_label_map()
    norm = tmp / "normalized.jsonl"
    norm.write_text(json.dumps({
        "text": text, "native_entities": native,
        "entities": [{"type": mapping.mapping[e["native_label"]],
                      "start": e["start"], "end": e["end"]} for e in native],
    }) + "\n")
    paths, conditions, pins = [], [], []
    for threshold in (0.3, 0.7):
        pred = tmp / f"pred-{threshold}.jsonl"
        pred.write_text(build_header({
            "threshold": threshold, "normalized_sha256": sha256_file(norm),
            "model": {"model_id": mapping.model_id, "model_revision": mapping.model_revision,
                      "max_width": 24},
        }) + "\n" + encode_record(PredictionRecord(0, 1, 0, (
            RawPrediction("first_name", 0, 2, 0.9),
            RawPrediction("date", start, start + 10, 0.9),
        ))) + "\n")
        baseline = analyze(norm, pred)
        conditions.append({
            "threshold": threshold, "full_strict": baseline["full_strict"],
            "gateway_supported_only_strict": baseline["gateway_supported_only_strict"],
            "inventory": [{"type": "DATE_TIME", **baseline["per_entity_type"]["DATE_TIME"]}],
        })
        pins.append({"threshold": threshold, "input_predictions_sha256": sha256_file(pred)})
        paths.append(pred)
    reference, fingerprints = tmp / "reference.json", tmp / "fingerprints.json"
    reference.write_text(json.dumps({"datasets": [{
        "dataset": "synthetic", "conditions": conditions,
    }]}))
    fingerprints.write_text(json.dumps({"datasets": [{
        "dataset": "synthetic", "normalized_sha256": sha256_file(norm), "conditions": pins,
    }]}))
    return norm, paths, reference, fingerprints


def test_complete_scoring_and_privacy(tmp_path: Path) -> None:
    norm, paths, reference, fingerprints = inputs(tmp_path)
    before = [sha256_file(p) for p in [norm, *paths, reference, fingerprints]]
    result = run(norm, paths, reference, fingerprints, "synthetic", 0)
    for c in result["conditions"]:
        assert c["datetime"]["delta"]["tp"] == 2
        assert c["datetime"]["delta"]["fp"] == 0
        assert c["datetime"]["delta"]["fn"] == -2
        assert c["recovery"]["recovered_span_mismatch"] == 1
        assert c["accounting"]["deleted_model_predictions"] == 0
        assert c["core_relations"]["exact_match"] == 2
    encoded = json.dumps(result)
    for forbidden in ("2024-02-29", "5:45", '"start"', '"end"', '"example_index"', '"text"'):
        assert forbidden not in encoded
    assert before == [sha256_file(p) for p in [norm, *paths, reference, fingerprints]]


def test_hash_failure_and_no_overwrite(tmp_path: Path) -> None:
    norm, paths, reference, fingerprints = inputs(tmp_path)
    assert main([
        "--normalized", str(norm), "--predictions", str(paths[0]),
        "--predictions", str(paths[1]), "--baseline-report", str(reference),
        "--input-reference", str(fingerprints), "--dataset", "synthetic",
        "--report", str(norm),
    ]) == 2
    paths[0].write_text(paths[0].read_text() + "\n")
    with pytest.raises(DateExperimentError):
        run(norm, paths, reference, fingerprints, "synthetic", 0)