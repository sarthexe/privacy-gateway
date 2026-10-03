from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import phonenumbers
import pytest

from scripts.evaluation_checkpoint import sha256_file
from scripts.gliner_evaluation import RawPrediction, load_label_map
from scripts.gliner_phone_boundary_refinement import (
    main,
    recovery_accounting,
    refine_candidate,
    run_experiment,
)
from scripts.gliner_phone_hybrid_experiment import (
    PhoneExperimentError,
    RegionPolicy,
    Span,
    merge_phone_candidates,
    phone_candidates,
)
from scripts.gliner_phone_hybrid_experiment import (
    run_experiment as run_a,
)
from scripts.gliner_predictions import PredictionRecord, build_header, encode_record

# Fictional NANP range; never real-person fixtures.
SYNTHETIC = "+1 202-555-0123"


def refine(value: str, region: str | None = None):
    return refine_candidate(value, Span(0, len(value)), RegionPolicy(region))


def test_unmatched_punctuation_trimmed_without_changing_identity() -> None:
    value = SYNTHETIC + ")"
    assert phone_candidates(value, RegionPolicy()).validated == (Span(0, len(value)),)
    result = refine(value)
    assert result.reason == "refined"
    assert result.span == Span(0, len(SYNTHETIC))


def test_original_absolute_boundaries_unicode_prefix_preserved() -> None:
    text = "測試: " + SYNTHETIC + ")"
    begin = text.index("+")
    result = refine_candidate(text, Span(begin, len(text)), RegionPolicy())
    assert result.span == Span(begin, begin + len(SYNTHETIC))
    assert text[result.span.start:result.span.end] == SYNTHETIC


@pytest.mark.parametrize("extension", [" ext. 42", " x42", " extension 42"])
def test_extension_is_preserved_not_dropped_to_shorten_span(extension: str) -> None:
    value = SYNTHETIC + extension
    result = refine(value)
    assert result.span == Span(0, len(value))
    assert result.reason == "unchanged"


def test_concatenated_numeric_material_is_not_reinterpreted() -> None:
    value = SYNTHETIC + " / 99"
    assert not phone_candidates(value, RegionPolicy()).validated
    assert refine(value).span is None


def test_even_parser_tolerated_outside_digits_are_never_discarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = SYNTHETIC + " / 99"
    identity = phonenumbers.parse(SYNTHETIC, None)
    monkeypatch.setattr(phonenumbers, "parse", lambda *a, **k: identity)
    monkeypatch.setattr(phonenumbers, "PhoneNumberMatcher", lambda *a, **k: [
        SimpleNamespace(start=0, end=len(SYNTHETIC), number=identity)
    ])
    assert refine(value).reason == "no_identity_preserving_boundary"


def test_unique_tightest_parser_valid_match_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    value = SYNTHETIC + ")"
    identity = phonenumbers.parse(value, None)
    monkeypatch.setattr(phonenumbers, "PhoneNumberMatcher", lambda *a, **k: [
        SimpleNamespace(start=0, end=len(value), number=identity),
        SimpleNamespace(start=0, end=len(SYNTHETIC), number=identity),
    ])
    assert refine(value).span == Span(0, len(SYNTHETIC))


def test_no_defensible_match_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(phonenumbers, "PhoneNumberMatcher", lambda *a, **k: [])
    assert refine(SYNTHETIC).reason == "no_identity_preserving_boundary"


def test_extension_identity_change_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    value = SYNTHETIC + " x42"
    no_extension = phonenumbers.parse(SYNTHETIC, None)
    monkeypatch.setattr(phonenumbers, "PhoneNumberMatcher", lambda *a, **k: [
        SimpleNamespace(start=0, end=len(value), number=no_extension)
    ])
    assert refine(value).reason == "no_identity_preserving_boundary"


def test_ambiguous_equal_length_boundaries_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    value = "(" + SYNTHETIC + ")"
    identity = phonenumbers.parse(value, "US")
    monkeypatch.setattr(phonenumbers, "PhoneNumberMatcher", lambda *a, **k: [
        SimpleNamespace(start=0, end=len(value) - 1, number=identity),
        SimpleNamespace(start=1, end=len(value), number=identity),
    ])
    assert refine(value, "US").reason == "ambiguous_tightest_boundary"


def test_international_policy_does_not_use_configured_national_region() -> None:
    assert refine(SYNTHETIC, "DE").span == Span(0, len(SYNTHETIC))


def test_explicit_region_policy_for_national_number() -> None:
    value = "202-555-0123"
    assert refine(value).reason == "explicit_region_required"
    assert refine(value, "US").span == Span(0, len(value))


def test_no_implicit_locale_or_environment_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("LC_ALL", "LANG", "PHONE_REGION"):
        monkeypatch.setenv(key, "US")
    assert refine("202-555-0123").span is None


def test_calling_code_indicator_is_not_discardable(monkeypatch: pytest.MonkeyPatch) -> None:
    identity = phonenumbers.parse(SYNTHETIC, None)
    monkeypatch.setattr(phonenumbers, "PhoneNumberMatcher", lambda *a, **k: [
        SimpleNamespace(start=1, end=len(SYNTHETIC), number=identity)
    ])
    assert refine(SYNTHETIC, "US").span is None


def test_refined_exact_duplicate_is_skipped_and_nonphones_unchanged() -> None:
    text = SYNTHETIC + ")"
    baseline = (
        RawPrediction("first_name", 0, 2, 0.9),
        RawPrediction("phone_number", 0, len(SYNTHETIC), 0.81),
        RawPrediction("url", 0, 3, 0.74),
    )
    result = refine(text)
    merged = merge_phone_candidates(text, baseline, [result.span], load_label_map())
    assert merged.predictions == baseline
    assert merged.duplicates_skipped == 1


@pytest.mark.parametrize("span", [Span(-1, 4), Span(0, 100), Span(2, 2), Span(True, 10)])
def test_malformed_span_fails_safely(span: Span) -> None:
    with pytest.raises(PhoneExperimentError):
        refine_candidate(SYNTHETIC, span, RegionPolicy())


def test_parser_failure_is_counted_without_source_echo(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        raise phonenumbers.NumberParseException(
            phonenumbers.NumberParseException.NOT_A_NUMBER, "private-source"
        )
    monkeypatch.setattr(phonenumbers, "parse", fail)
    result = refine(SYNTHETIC)
    assert result.span is None and result.reason == "parser_failure"
    assert "private-source" not in repr(result)


def test_recovery_retention_tracks_identity_not_only_tp_total() -> None:
    first, second = Span(0, 10), Span(20, 30)
    result = recovery_accounting(
        Counter({first: 1, second: 1}), Counter(), Counter({first: 1}), Counter({second: 1})
    )
    assert result["variant_a_recovered_fn"] == 1
    assert result["variant_b_retained_recoveries"] == 0
    assert result["variant_b_new_recoveries"] == 1


def test_fp_disappearance_and_new_fp_are_separate() -> None:
    old_fp, new_fp = Span(0, 10), Span(20, 30)
    result = recovery_accounting(Counter(), Counter(), Counter({old_fp: 1}), Counter({new_fp: 1}))
    assert result["variant_a_added_fp_disappeared"] == 1
    assert result["variant_b_new_fp"] == 1
    assert result["variant_b_total_added_fp"] == 1


def make_inputs(tmp_path: Path) -> tuple[Path, list[Path], Path]:
    text = "Call " + SYNTHETIC + ") then " + SYNTHETIC
    first, second = text.index("+"), text.rindex("+")
    entities = [{"type": "PERSON", "start": 0, "end": 4}] + [
        {"type": "PHONE_NUMBER", "start": pos, "end": pos + len(SYNTHETIC)}
        for pos in (first, second)
    ]
    native = [
        {"native_label": "first_name" if e["type"] == "PERSON" else "phone_number",
         "start": e["start"], "end": e["end"]}
        for e in entities
    ]
    normalized = tmp_path / "normalized.jsonl"
    normalized.write_text(json.dumps({
        "text": text, "entities": entities, "native_entities": native,
    }) + "\n")
    mapping = load_label_map()
    paths = []
    for threshold in (0.3, 0.7):
        path = tmp_path / f"p-{threshold}.jsonl"
        identity = {
            "threshold": threshold, "normalized_sha256": sha256_file(normalized),
            "model": {
                "model_id": mapping.model_id, "model_revision": mapping.model_revision,
                "max_width": 24,
            },
        }
        path.write_text(build_header(identity) + "\n" + encode_record(PredictionRecord(
            0, 1, 0, (RawPrediction("first_name", 0, 4, 0.9),)
        )) + "\n")
        paths.append(path)
    frozen = tmp_path / "a.json"
    frozen.write_text(json.dumps({"datasets": [run_a(normalized, paths, RegionPolicy(), 0)]}))
    return normalized, paths, frozen


def test_end_to_end_three_way_scoring_retention_and_privacy(tmp_path: Path) -> None:
    normalized, paths, frozen = make_inputs(tmp_path)
    old_hashes = [sha256_file(p) for p in [normalized, *paths, frozen]]
    report = run_experiment(normalized, paths, frozen, RegionPolicy(), 0)
    for condition in report["conditions"]:
        phone = condition["phone_number"]
        assert phone["baseline"]["tp"] == 0
        assert phone["variant_a"]["tp"] == 1 and phone["variant_a"]["fp"] == 1
        assert phone["variant_b"]["tp"] == 2 and phone["variant_b"]["fp"] == 0
        retention = condition["recovery_retention"]
        assert retention["variant_b_retained_recoveries"] == 1
        assert retention["retention_pct"] == 100
        assert retention["variant_a_added_fp_disappeared"] == 1
        assert retention["variant_b_new_fp"] == 0
        assert condition["regression"]["existing_predictions_modified"] == 0
    encoded = json.dumps(report)
    assert SYNTHETIC not in encoded
    for forbidden in ('"start"', '"end"', '"text"', '"example_index"'):
        assert forbidden not in encoded
    assert old_hashes == [sha256_file(p) for p in [normalized, *paths, frozen]]


def test_changed_frozen_inputs_fail_safely(tmp_path: Path) -> None:
    normalized, paths, frozen = make_inputs(tmp_path)
    paths[0].write_text(paths[0].read_text() + "\n")
    with pytest.raises(PhoneExperimentError, match="identity differs"):
        run_experiment(normalized, paths, frozen, RegionPolicy(), 0)


def test_cli_does_not_overwrite_inputs(tmp_path: Path) -> None:
    normalized, paths, frozen = make_inputs(tmp_path)
    before = normalized.read_bytes()
    assert main([
        "--normalized", str(normalized),
        "--predictions", str(paths[0]), "--predictions", str(paths[1]),
        "--variant-a-report", str(frozen), "--report", str(normalized),
    ]) == 2
    assert normalized.read_bytes() == before