from __future__ import annotations

import ast
import inspect
import json
import tempfile
from pathlib import Path

import pytest

from scripts.gliner_email_hybrid import recognize_email_addresses
from scripts.gliner_evaluation import GlinerLabelMap, RawPrediction
from scripts.gliner_person_email_suppression import (
    EMAIL_RECOGNIZER,
    Classification,
    PersonEmailExperimentError,
    account_suppressed,
    classify_predictions,
    removal_is_subsequence,
    run_condition,
    scoring_views,
    strictly_inside,
)
from scripts.gliner_predictions import PredictionRecord, build_header, encode_record
from scripts.nemotron_pii import is_within_repository
from scripts.presidio_evaluation import ExactMatch

LABEL_MAP = GlinerLabelMap(
    "test",
    "test",
    ("email", "first_name", "last_name", "user_name", "company_name", "city", "phone_number"),
    {
        "email": "EMAIL_ADDRESS",
        "first_name": "PERSON",
        "last_name": "PERSON",
        "user_name": "USERNAME",
        "company_name": "ORGANIZATION",
        "city": "LOCATION",
        "phone_number": "PHONE_NUMBER",
    },
    {},
)


def _email(text: str) -> tuple[int, int]:
    found = recognize_email_addresses(text)
    assert len(found) == 1
    return found[0].start, found[0].end


def _classify(text: str, predictions: list[RawPrediction]) -> Classification:
    return classify_predictions(text, predictions, LABEL_MAP)


def test_email_recognizer_is_the_frozen_helper() -> None:
    assert EMAIL_RECOGNIZER is recognize_email_addresses
    assert "truth" not in inspect.signature(classify_predictions).parameters
    assert "entities" not in inspect.signature(classify_predictions).parameters


def test_person_strictly_inside_valid_email_is_suppressed() -> None:
    text = "write ada.q@example.com today"
    start, end = _email(text)
    inner = text.index("example")
    result = _classify(text, [RawPrediction("first_name", inner, inner + 7, 0.9)])
    assert strictly_inside(inner, inner + 7, start, end)
    assert result.kept == ()
    assert len(result.strict) == 1


def test_person_not_fully_contained_is_kept() -> None:
    text = "ada wrote ada.q@example.com"
    email_start, _ = _email(text)
    result = _classify(text, [RawPrediction("first_name", 0, email_start + 2, 0.9)])
    assert result.strict == ()
    assert [item.start for item in result.kept] == [0]


def test_person_touching_email_boundary_is_kept() -> None:
    text = "mail ada.q@example.com now"
    start, end = _email(text)
    touching_start = RawPrediction("first_name", start, start + 3, 0.91)
    touching_end = RawPrediction("last_name", end - 3, end, 0.92)
    result = _classify(text, [touching_start, touching_end])
    assert result.strict == ()
    assert result.kept == (touching_start, touching_end)
    assert len(result.loose_only) == 2


def test_person_overlapping_email_but_extending_outside_is_kept() -> None:
    text = "xx ada.q@example.com yy"
    start, end = _email(text)
    prediction = RawPrediction("first_name", start - 1, end - 1, 0.9)
    result = _classify(text, [prediction])
    assert result.strict == ()
    assert result.kept == (prediction,)
    assert result.loose_only == (prediction,)


def test_invalid_email_causes_no_suppression() -> None:
    text = "reach ada@localhost today"
    assert recognize_email_addresses(text) == []
    prediction = RawPrediction("first_name", text.index("ada"), text.index("ada") + 3, 0.9)
    result = _classify(text, [prediction])
    assert result.strict == () and result.loose == ()
    assert result.kept == (prediction,)


def test_nearby_email_does_not_suppress_unrelated_person() -> None:
    text = "Ada mailed bob@example.com"
    prediction = RawPrediction("first_name", 0, 3, 0.95)
    result = _classify(text, [prediction])
    assert result.strict == () and result.loose == ()
    assert result.kept == (prediction,)


def test_multiple_emails_suppress_only_contained_people() -> None:
    text = "ada@example.com and quinn.b@example.org"
    first, second = recognize_email_addresses(text)
    inside_first = RawPrediction("first_name", first.start + 1, first.start + 3, 0.9)
    inside_second = RawPrediction("last_name", second.start + 1, second.start + 4, 0.8)
    outside = RawPrediction("first_name", first.end + 1, first.end + 4, 0.7)
    result = _classify(text, [inside_first, outside, inside_second])
    assert result.strict == (inside_first, inside_second)
    assert result.kept == (outside,)


def test_multiple_eligible_person_predictions_are_all_removed() -> None:
    text = "see ada.quinn@example.com"
    start, _ = _email(text)
    predictions = [
        RawPrediction("first_name", start + 1, start + 4, 0.9),
        RawPrediction("last_name", start + 5, start + 10, 0.8),
        RawPrediction("first_name", 0, 3, 0.4),
    ]
    result = _classify(text, predictions)
    assert result.strict == tuple(predictions[:2])
    assert result.kept == (predictions[2],)


def test_punctuation_around_email_stays_outside_the_candidate() -> None:
    text = "(ada.q@example.com),"
    start, end = _email(text)
    assert text[start:end] == "ada.q@example.com"
    assert start > 0 and end < len(text)
    inner = RawPrediction("first_name", start + 1, start + 4, 0.9)
    result = _classify(text, [inner])
    assert result.strict == (inner,)


def test_strict_containment_is_narrower_than_loose_overlap() -> None:
    text = "ada.q@example.com"
    start, end = _email(text)
    inside = RawPrediction("first_name", start + 1, start + 4, 0.9)
    overlap = RawPrediction("last_name", start, start + 3, 0.9)
    result = _classify(text, [inside, overlap])
    assert result.strict == (inside,)
    assert result.loose == (inside, overlap)
    assert result.loose_only == (overlap,)


def test_true_positive_risk_is_suppressed_without_consulting_truth() -> None:
    text = "mail ada.q@example.com"
    start, _ = _email(text)
    prediction = RawPrediction("first_name", start + 1, start + 4, 0.99)
    result = _classify(text, [prediction])
    truth = (
        ExactMatch("PERSON", prediction.start, prediction.end),
        ExactMatch("EMAIL_ADDRESS", start, _email(text)[1]),
    )
    account = account_suppressed(result.strict, truth, LABEL_MAP)
    assert result.kept == ()
    assert account["exact_person_gt_matches"] == 1
    assert account["overlapping_email_gt"] == 1
    assert account["no_gt_overlap"] == 0


def test_non_person_predictions_inside_email_stay() -> None:
    text = "mail ada.q@example.com"
    start, end = _email(text)
    predictions = [
        RawPrediction("user_name", start + 1, start + 4, 0.9),
        RawPrediction("company_name", start + 5, start + 6, 0.9),
        RawPrediction("city", start + 1, end - 1, 0.9),
        RawPrediction("phone_number", start + 1, start + 4, 0.9),
    ]
    result = _classify(text, predictions)
    assert result.strict == () and result.loose == ()
    assert result.kept == tuple(predictions)


def test_email_address_predictions_are_not_added_or_removed() -> None:
    text = "mail ada.q@example.com"
    start, end = _email(text)
    email = RawPrediction("email", start, end, 0.4)
    person = RawPrediction("first_name", start + 1, start + 4, 0.9)
    result = _classify(text, [email, person])
    assert result.kept == (email,)
    assert result.strict == (person,)


def test_duplicate_predictions_are_removed_independently() -> None:
    text = "mail ada.q@example.com"
    start, _ = _email(text)
    one = RawPrediction("first_name", start + 1, start + 4, 0.9)
    two = RawPrediction("first_name", start + 1, start + 4, 0.9)
    outside = RawPrediction("first_name", 0, 4, 0.9)
    result = _classify(text, [one, two, outside])
    assert result.strict == (one, two)
    assert result.kept == (outside,)
    assert removal_is_subsequence((one, two, outside), result.kept) == 2


def test_malformed_input_fails_closed() -> None:
    text = "ada.q@example.com"
    with pytest.raises(PersonEmailExperimentError, match="outside source text"):
        _classify(text, [RawPrediction("first_name", 0, len(text) + 1, 0.9)])
    with pytest.raises(PersonEmailExperimentError, match="outside the pinned label map"):
        _classify(text, [RawPrediction("not_a_label", 1, 3, 0.9)])
    with pytest.raises(PersonEmailExperimentError, match="source text"):
        classify_predictions(None, [], LABEL_MAP)  # type: ignore[arg-type]


def test_equal_score_to_threshold_is_not_scored_or_suppressed() -> None:
    text = "mail ada.q@example.com"
    start, _ = _email(text)
    low = RawPrediction("first_name", start + 1, start + 4, 0.3)
    kept_score = RawPrediction("first_name", start + 1, start + 4, 0.3000001)
    baseline, classified = scoring_views(text, [low, kept_score], LABEL_MAP, 0.3)
    assert baseline == (kept_score,)
    assert classified.strict == (kept_score,)
    assert _prediction_unchanged([low, kept_score])


def _prediction_unchanged(predictions: list[RawPrediction]) -> bool:
    return predictions == [
        RawPrediction("first_name", predictions[0].start, predictions[0].end, 0.3),
        RawPrediction("first_name", predictions[1].start, predictions[1].end, 0.3000001),
    ]


def test_ground_truth_does_not_change_eligibility() -> None:
    text = "mail ada.q@example.com"
    start, _ = _email(text)
    prediction = RawPrediction("first_name", start + 1, start + 4, 0.9)
    first = _classify(text, [prediction])
    second = _classify(text, [prediction])
    truth = (ExactMatch("PERSON", prediction.start, prediction.end),)
    matching = account_suppressed(first.strict, truth, LABEL_MAP)
    absent = account_suppressed(first.strict, (), LABEL_MAP)
    assert first == second
    assert matching["exact_person_gt_matches"] == 1
    assert absent["no_gt_overlap"] == 1
    assert absent["exact_person_gt_matches"] == 0


def test_module_does_not_import_inference_or_production_detector() -> None:
    source = Path(__file__).parents[1] / "scripts" / "gliner_person_email_suppression.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert "gliner" not in imported
    assert "torch" not in imported
    assert "app.detection" not in imported
    assert "scripts.gliner_predict" not in imported
    assert "scripts.gliner_ood_evaluate" not in imported


def _write_inputs(
    root: Path, rows: list[tuple[str, list[dict[str, int | str]], list[RawPrediction]]]
) -> tuple[Path, Path]:
    normalized = root / "ai4privacy_validation.jsonl"
    predictions = root / "predictions.jsonl"
    identity = {"model": {"max_width": 24}, "protocol": {"max_width": 24}}
    with (
        normalized.open("w", encoding="utf-8", newline="\n") as data,
        predictions.open("w", encoding="utf-8", newline="\n") as preds,
    ):
        preds.write(build_header(identity) + "\n")
        for index, (text, entities, items) in enumerate(rows):
            data.write(
                json.dumps(
                    {"entities": entities, "example_index": index, "text": text},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            preds.write(encode_record(PredictionRecord(index, 1, 0, tuple(items))) + "\n")
    return normalized, predictions


def test_runner_reports_aggregates_without_mutating_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "mail ada.q@example.com"
    start, end = _email(text)
    person = RawPrediction("first_name", start + 1, start + 4, 0.95)
    email = RawPrediction("email", start, end, 0.5)
    city = RawPrediction("city", 0, 4, 0.6)
    rows = [
        (
            text,
            [
                {"end": person.end, "start": person.start, "type": "PERSON"},
                {"end": end, "start": start, "type": "EMAIL_ADDRESS"},
            ],
            [person, email, city],
        )
    ]
    production = Path(__file__).parents[1] / "scripts" / "gliner_email_hybrid.py"
    before = production.read_bytes()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        assert not is_within_repository(root)
        normalized, predictions = _write_inputs(root, rows)
        normalized_hash = normalized.read_bytes()
        predictions_hash = predictions.read_bytes()
        monkeypatch.setattr(
            "scripts.gliner_person_email_suppression.EXPECTED_NORMALIZED_SHA256",
            {"ai4privacy": __import__("hashlib").sha256(normalized_hash).hexdigest()},
        )
        report = run_condition("ai4privacy", normalized, predictions, 0.3, LABEL_MAP)
        assert normalized.read_bytes() == normalized_hash
        assert predictions.read_bytes() == predictions_hash
        payload = json.dumps(report)
        assert "ada.q@example.com" not in payload
        assert "example_index" not in payload
        assert "false_negative_samples" not in payload
        assert report["invariants"] == {
            "original_total_predictions": 3,
            "scored_predictions": 3,
            "suppressed_predictions": 1,
            "added_predictions": 0,
            "modified_predictions": 0,
            "deleted_predictions": 1,
        }
        assert report["person"]["baseline"]["tp"] == 1
        assert report["person"]["strict_suppression"]["tp"] == 0
        assert report["person"]["delta"]["tp"] == -1
        assert report["person"]["delta"]["fp"] == 0
        assert report["tp_loss_table"]["suppressed_tp"] == 1
        assert report["tp_loss_table"]["suppressed_fp"] == 0
        assert report["accounting"]["overlapping_email_gt"] == 1
        assert report["full_strict"]["delta"]["tp"] == -1
        assert report["input"]["predictions_sha256"] == report["input"]["predictions_sha256_after"]
    assert production.read_bytes() == before


def test_runner_fails_closed_on_bad_width(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        normalized = root / "data.jsonl"
        predictions = root / "predictions.jsonl"
        report = root / "report.json"
        normalized.write_text('{"entities": [], "text": "a"}\n', encoding="utf-8", newline="\n")
        predictions.write_text(
            build_header({"model": {"max_width": 12}}) + "\n", encoding="utf-8", newline="\n"
        )
        monkeypatch.setattr(
            "scripts.gliner_person_email_suppression.EXPECTED_NORMALIZED_SHA256",
            {"gretel": __import__("hashlib").sha256(normalized.read_bytes()).hexdigest()},
        )
        with pytest.raises(PersonEmailExperimentError, match="max_width=24"):
            run_condition("gretel", normalized, predictions, 0.7, LABEL_MAP)
        assert not report.exists()
