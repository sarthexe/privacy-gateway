from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from scripts.evaluation_checkpoint import sha256_file
from scripts.gliner_evaluation import RawPrediction, load_label_map
from scripts.gliner_phone_hybrid_experiment import (
    BOUNDARIES,
    PhoneExperimentError,
    RegionPolicy,
    Span,
    classify_candidate,
    deduplicate_candidates,
    main,
    merge_phone_candidates,
    phone_candidates,
    run_experiment,
)
from scripts.gliner_predictions import PredictionRecord, build_header, encode_record

# NANP 555-0100 through 555-0199 is reserved for fictional use.
SYNTHETIC = "+1 202-555-0123"


def test_exact_international_match() -> None:
    result = phone_candidates(SYNTHETIC, RegionPolicy())
    assert result.raw == result.validated == (Span(0, len(SYNTHETIC)),)
    assert not result.rejections


def test_invalid_phone_rejected() -> None:
    result = phone_candidates("+1 000-000-0000", RegionPolicy())
    assert len(result.raw) == 1
    assert result.validated == ()
    assert result.rejections == Counter({"invalid_number": 1})


def test_candidate_deduplication() -> None:
    assert deduplicate_candidates([Span(0, 7), Span(0, 7), Span(9, 16)], 16) == (
        Span(0, 7), Span(9, 16)
    )


def test_exact_original_boundaries_unicode_prefix_and_punctuation() -> None:
    text = f"測試: {SYNTHETIC}, done."
    result = phone_candidates(text, RegionPolicy())
    begin = text.index("+")
    assert result.validated == (Span(begin, begin + len(SYNTHETIC)),)
    assert text[result.validated[0].start:result.validated[0].end] == SYNTHETIC


def test_explicit_region_required_for_national_number() -> None:
    national = "(202) 555-0123"
    assert not phone_candidates(national, RegionPolicy()).validated
    assert phone_candidates(national, RegionPolicy()).rejections == Counter(
        {"explicit_region_required": 1}
    )
    assert phone_candidates(national, RegionPolicy("US")).validated == (Span(0, len(national)),)


def test_no_implicit_locale_region_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("LANG", "LC_ALL", "PHONE_REGION", "DEFAULT_REGION"):
        monkeypatch.setenv(key, "US")
    text = "English USA phone: 202-555-0123"
    assert not phone_candidates(text, RegionPolicy()).validated


def test_international_number_ignores_configured_national_region() -> None:
    assert phone_candidates(SYNTHETIC, RegionPolicy("DE")).validated == (
        Span(0, len(SYNTHETIC)),
    )


@pytest.mark.parametrize("region", ["us", "ZZ", "", True, 123])
def test_invalid_region_fails(region: object) -> None:
    with pytest.raises(PhoneExperimentError):
        RegionPolicy(region)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("candidate", "gt", "expected"),
    [
        (Span(2, 10), Span(2, 10), "exact_match"),
        (Span(3, 9), Span(2, 10), "candidate_contained_by_gt"),
        (Span(1, 11), Span(2, 10), "gt_contained_by_candidate"),
        (Span(1, 5), Span(2, 10), "partial_overlap"),
        (Span(10, 12), Span(2, 10), "no_phone_gt_overlap"),
    ],
)
def test_overlap_classification(candidate: Span, gt: Span, expected: str) -> None:
    assert classify_candidate(candidate, [gt]) == expected


def test_multiple_gt_priority_is_exact_first() -> None:
    assert classify_candidate(Span(2, 10), [Span(1, 11), Span(2, 10)]) == BOUNDARIES[0]


def test_nonphone_predictions_and_baseline_duplicates_unchanged() -> None:
    mapping = load_label_map()
    baseline = (
        RawPrediction("first_name", 0, 2, 0.83),
        RawPrediction("first_name", 0, 2, 0.83),
        RawPrediction("url", 0, 3, 0.91),
    )
    result = merge_phone_candidates(SYNTHETIC, baseline, [Span(0, len(SYNTHETIC))], mapping)
    assert result.predictions[:3] == baseline
    assert result.added == (Span(0, len(SYNTHETIC)),)


@pytest.mark.parametrize("label", ["phone_number", "fax_number"])
def test_exact_model_phone_duplicate_not_added_twice(label: str) -> None:
    baseline = (RawPrediction(label, 0, len(SYNTHETIC), 0.8),)
    span = Span(0, len(SYNTHETIC))
    result = merge_phone_candidates(SYNTHETIC, baseline, [span, span], load_label_map())
    assert result.predictions == baseline
    assert result.duplicates_skipped == 1


def test_nonexact_phone_overlap_is_not_suppressed() -> None:
    baseline = (RawPrediction("phone_number", 3, len(SYNTHETIC), 0.9),)
    result = merge_phone_candidates(
        SYNTHETIC, baseline, [Span(0, len(SYNTHETIC))], load_label_map()
    )
    assert result.predictions[:1] == baseline
    assert len(result.predictions) == 2


@pytest.mark.parametrize("span", [Span(-1, 3), Span(3, 3), Span(0, 99), Span(True, 7)])
def test_malformed_candidate_fails_safely(span: Span) -> None:
    with pytest.raises(PhoneExperimentError, match="candidate offsets are invalid"):
        deduplicate_candidates([span], len(SYNTHETIC))


@pytest.mark.parametrize(
    "prediction",
    [
        RawPrediction("unknown", 0, 2, 0.8),
        RawPrediction("phone_number", 0, 99, 0.8),
        RawPrediction("phone_number", 0, 3, float("nan")),
        RawPrediction("phone_number", True, 3, 0.8),
    ],
)
def test_malformed_prediction_fails_safely(prediction: RawPrediction) -> None:
    with pytest.raises(PhoneExperimentError, match="baseline prediction is invalid"):
        merge_phone_candidates(SYNTHETIC, [prediction], [], load_label_map())


def test_bad_candidate_input_fails_safely() -> None:
    with pytest.raises(PhoneExperimentError):
        phone_candidates(None, RegionPolicy())  # type: ignore[arg-type]


def make_inputs(tmp_path: Path) -> tuple[Path, list[Path]]:
    normalized = tmp_path / "normalized.jsonl"
    normalized.write_text(json.dumps({
        "text": SYNTHETIC,
        "entities": [{"type": "PHONE_NUMBER", "start": 0, "end": len(SYNTHETIC)}],
        "native_entities": [
            {"native_label": "phone_number", "start": 0, "end": len(SYNTHETIC)}
        ],
    }) + "\n")
    mapping = load_label_map()
    paths = []
    for threshold in (0.3, 0.7):
        path = tmp_path / f"predictions-{threshold}.jsonl"
        identity = {
            "normalized_sha256": sha256_file(normalized), "threshold": threshold,
            "model": {
                "model_id": mapping.model_id,
                "model_revision": mapping.model_revision,
                "max_width": 24,
            },
        }
        path.write_text(build_header(identity) + "\n" + encode_record(
            PredictionRecord(0, 1, 0, ())
        ) + "\n")
        paths.append(path)
    return normalized, paths


def test_end_to_end_strict_scoring_and_aggregate_privacy(tmp_path: Path) -> None:
    normalized, paths = make_inputs(tmp_path)
    hashes = [sha256_file(p) for p in [normalized, *paths]]
    report = run_experiment(normalized, paths, RegionPolicy(), 0)
    for condition in report["conditions"]:
        assert condition["phone_number"]["baseline"]["fn"] == 1
        assert condition["phone_number"]["hybrid"]["tp"] == 1
        assert condition["phone_number"]["delta"]["fp"] == 0
        assert condition["regression"]["changed_non_phone_predictions"] == 0
        assert condition["recovered_fn_evidence"] == {
            "no_baseline_native_prediction_overlap": 1
        }
    encoded = json.dumps(report)
    assert SYNTHETIC not in encoded
    for forbidden in ('"start"', '"end"', '"text"', '"example_index"', '"native_entities"'):
        assert forbidden not in encoded
    assert hashes == [sha256_file(p) for p in [normalized, *paths]]


def test_misaligned_predictions_fail_safely(tmp_path: Path) -> None:
    normalized, paths = make_inputs(tmp_path)
    lines = paths[0].read_text().splitlines()
    record = json.loads(lines[1])
    record["i"] = 1
    paths[0].write_text(lines[0] + "\n" + json.dumps(record) + "\n")
    with pytest.raises(PhoneExperimentError, match="indices are not aligned"):
        run_experiment(normalized, paths, RegionPolicy(), 0)


def test_malformed_json_cli_fails_without_source_echo(tmp_path: Path, capsys: Any) -> None:
    normalized, paths = make_inputs(tmp_path)
    paths[0].write_text("not-json-secret\n")
    code = main([
        "--normalized", str(normalized),
        "--predictions", str(paths[0]), "--predictions", str(paths[1]),
        "--report", str(tmp_path / "report.json"),
    ])
    assert code == 2
    assert "not-json-secret" not in capsys.readouterr().err
    assert not (tmp_path / "report.json").exists()


def test_cli_refuses_overwriting_input(tmp_path: Path) -> None:
    normalized, paths = make_inputs(tmp_path)
    original = normalized.read_bytes()
    assert main([
        "--normalized", str(normalized),
        "--predictions", str(paths[0]), "--predictions", str(paths[1]),
        "--report", str(normalized),
    ]) == 2
    assert normalized.read_bytes() == original