from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts import analyze_gliner_ood_errors as analysis
from scripts.evaluation_checkpoint import sha256_file
from scripts.gliner_evaluation import RawPrediction, load_label_map
from scripts.gliner_predictions import PredictionRecord, build_header, encode_record


def make_inputs(
    tmp_path: Path,
    truth: list[tuple[str, int, int]],
    predictions: list[tuple[str, int, int]],
    *,
    threshold: float = 0.3,
    text: str = "synthetic_private_sentinel\u2028more\u2029text\u0085" + "x" * 30,
    source_index: int = 17,
) -> tuple[Path, Path]:
    label_map = load_label_map()
    normalized, saved = tmp_path / "normalized.jsonl", tmp_path / "predictions.jsonl"
    row = {
        "example_index": source_index,
        "text": text,
        "entities": [{"type": kind, "start": start, "end": end} for kind, start, end in truth],
        "native_entities": [
            {"native_label": "UNMAPPED", "start": start, "end": end}
            for _, start, end in truth
        ],
        "metadata": {"dataset": "private_metadata_sentinel"},
    }
    normalized.write_text("\n" + json.dumps(row, ensure_ascii=False) + "\n \t\n", encoding="utf-8")
    identity = {
        "normalized_sha256": sha256_file(normalized),
        "model": {"model_id": label_map.model_id, "model_revision": label_map.model_revision},
        "threshold": threshold,
        "untrusted_extra": "private_header_sentinel",
    }
    record = PredictionRecord(0, 1, 0, tuple(
        RawPrediction(kind, start, end, 0.9) for kind, start, end in predictions
    ))
    saved.write_text(build_header(identity) + "\n" + encode_record(record) + "\n", encoding="utf-8")
    return normalized, saved


def test_exact_tp_uses_gateway_mapping_and_keeps_source_indices(tmp_path: Path) -> None:
    normalized, saved = make_inputs(tmp_path, [("PERSON", 2, 8)], [("first_name", 2, 8)])
    report = analysis.analyze(normalized, saved)
    assert report["overall"] == {"tp": 1, "fp": 0, "fn": 0}
    assert report["record_count"] == 1
    assert report["per_entity_type"]["PERSON"]["precision"] == 1
    assert report["per_entity_type"]["PERSON"]["recall"] == 1
    assert report["per_entity_type"]["PERSON"]["f1"] == 1
    assert report["false_negatives"]["by_failure_category"] == {}


@pytest.mark.parametrize("threshold", [0.3, 0.7])
def test_same_type_span_mismatch(tmp_path: Path, threshold: float) -> None:
    paths = make_inputs(tmp_path, [("PERSON", 4, 10)], [("first_name", 2, 8)], threshold=threshold)
    report = analysis.analyze(*paths)
    assert report["threshold"] == threshold
    assert report["overall"] == {"tp": 0, "fp": 1, "fn": 1}
    assert report["false_negatives"]["by_entity_type"] == {"PERSON": {"span_mismatch": 1}}
    assert report["false_positives"]["by_entity_type"] == {"PERSON": {"span_mismatch": 1}}
    patterns = report["span_mismatches"]["boundary_patterns"]
    assert patterns["prediction_starts_before_gt"] == 1
    assert patterns["prediction_ends_before_gt"] == 1


def test_detector_miss_and_disjoint_false_positive(tmp_path: Path) -> None:
    report = analysis.analyze(*make_inputs(
        tmp_path, [("PERSON", 2, 8)], [("first_name", 12, 18)]
    ))
    assert report["overall"] == {"tp": 0, "fp": 1, "fn": 1}
    assert report["false_negatives"]["by_failure_category"] == {"detector_miss": 1}
    assert report["false_positives"]["by_failure_category"] == {"detector_recognizer": 1}
    assert report["likely_type_confusions"]["overlap_pairs"] == 0


def test_type_confusion_is_supplementary_to_fp_fn(tmp_path: Path) -> None:
    report = analysis.analyze(*make_inputs(
        tmp_path, [("PERSON", 2, 8)], [("email", 2, 8)]
    ))
    assert report["overall"] == {"tp": 0, "fp": 1, "fn": 1}
    assert report["likely_type_confusions"]["overlap_pairs"] == 1
    assert report["likely_type_confusions"]["by_ground_truth_and_predicted_type"] == {
        "PERSON": {"EMAIL_ADDRESS": 1}
    }
    assert report["span_mismatches"]["overlap_pairs"] == 0
    assert report["per_entity_type"]["EMAIL_ADDRESS"]["fp"] == 1
    assert report["per_entity_type"]["PERSON"]["fn"] == 1


def test_unmapped_and_unsupported_ground_truth(tmp_path: Path) -> None:
    report = analysis.analyze(*make_inputs(
        tmp_path, [("UNMAPPED", 2, 8), ("private_source_label", 12, 18)], []
    ))
    assert report["overall"] == {"tp": 0, "fp": 0, "fn": 2}
    assert report["unmapped_ground_truth"] == 1
    assert report["false_negatives"]["by_failure_category"] == {
        "unmapped_source_label": 1, "unsupported_entity": 1
    }
    assert "private_source_label" not in json.dumps(report)


@pytest.mark.parametrize(("span", "flags"), [
    ((2, 12), {
        "prediction_starts_before_gt", "prediction_ends_after_gt", "prediction_contains_gt"
    }),
    ((6, 8), {"prediction_starts_after_gt", "prediction_ends_before_gt", "gt_contains_prediction"}),
    ((4, 12), {"prediction_ends_after_gt", "prediction_contains_gt"}),
    ((4, 8), {"prediction_ends_before_gt", "gt_contains_prediction"}),
    ((6, 12), {"prediction_starts_after_gt", "prediction_ends_after_gt"}),
])
def test_boundary_and_containment_patterns(
    tmp_path: Path, span: tuple[int, int], flags: set[str],
) -> None:
    report = analysis.analyze(*make_inputs(
        tmp_path, [("PERSON", 4, 10)], [("first_name", *span)]
    ))
    patterns = report["span_mismatches"]["boundary_patterns"]
    assert {name for name, count in patterns.items() if count} == flags
    assert report["span_mismatches"]["overlap_pairs"] == 1
    assert set(report["span_mismatches"]["by_entity_type"]["PERSON"]) == flags


def test_duplicate_errors_preserve_multiset_counts(tmp_path: Path) -> None:
    report = analysis.analyze(*make_inputs(
        tmp_path, [("PERSON", 2, 8), ("PERSON", 2, 8)],
        [("first_name", 12, 18), ("first_name", 12, 18)],
    ))
    assert report["overall"] == {"tp": 0, "fp": 2, "fn": 2}
    assert report["false_negatives"]["by_failure_category"] == {"detector_miss": 2}
    assert report["false_positives"]["by_failure_category"] == {"detector_recognizer": 2}


def test_partial_duplicate_exact_match_keeps_remaining_fn(tmp_path: Path) -> None:
    report = analysis.analyze(*make_inputs(
        tmp_path, [("PERSON", 2, 8), ("PERSON", 2, 8)], [("first_name", 2, 8)],
    ))
    assert report["overall"] == {"tp": 1, "fp": 0, "fn": 1}
    assert report["false_negatives"]["by_failure_category"] == {"detector_miss": 1}
    assert report["per_entity_type"]["PERSON"]["recall"] == 0.5


def test_touching_spans_do_not_overlap(tmp_path: Path) -> None:
    report = analysis.analyze(*make_inputs(
        tmp_path, [("PERSON", 2, 8)], [("first_name", 8, 12)],
    ))
    assert report["false_negatives"]["by_failure_category"] == {"detector_miss": 1}
    assert report["false_positives"]["by_failure_category"] == {"detector_recognizer": 1}
    assert report["span_mismatches"]["overlap_pairs"] == 0


def test_empty_entities_are_valid(tmp_path: Path) -> None:
    report = analysis.analyze(*make_inputs(tmp_path, [], []))
    assert report["overall"] == {"tp": 0, "fp": 0, "fn": 0}
    assert report["per_entity_type"] == {}


@pytest.mark.parametrize("extra_predictions", [False, True])
def test_record_count_mismatch(tmp_path: Path, extra_predictions: bool) -> None:
    normalized, saved = make_inputs(tmp_path, [], [])
    if extra_predictions:
        with saved.open("a", encoding="utf-8") as handle:
            handle.write(encode_record(PredictionRecord(1, 1, 0, ())) + "\n")
    else:
        saved.write_text(saved.read_text().split("\n")[0] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="record count"):
        analysis.analyze(normalized, saved)


def test_prediction_order_is_checked(tmp_path: Path) -> None:
    normalized, saved = make_inputs(tmp_path, [], [])
    header = saved.read_text().split("\n")[0]
    saved.write_text(header + "\n" + encode_record(PredictionRecord(17, 1, 0, ())) + "\n")
    with pytest.raises(ValueError, match="indices"):
        analysis.analyze(normalized, saved)


@pytest.mark.parametrize("change", ["hash", "model", "threshold", "label", "offset", "native"])
def test_invalid_inputs_fail_closed(tmp_path: Path, change: str) -> None:
    normalized, saved = make_inputs(tmp_path, [("PERSON", 2, 8)], [])
    identity = json.loads(saved.read_text().split("\n")[0])["identity"]
    record = PredictionRecord(0, 1, 0, ())
    if change == "hash":
        identity["normalized_sha256"] = "incorrect"
    elif change == "model":
        identity["model"]["model_revision"] = "incorrect"
    elif change == "threshold":
        identity["threshold"] = True
    elif change in {"label", "offset"}:
        record = PredictionRecord(0, 1, 0, (
            RawPrediction("unknown_label" if change == "label" else "email", 0, 1000, 0.9),
        ))
    else:
        row = json.loads(next(line for line in normalized.read_text().split("\n") if line.strip()))
        row["native_entities"][0]["end"] = 9
        normalized.write_text(json.dumps(row) + "\n")
        identity["normalized_sha256"] = sha256_file(normalized)
    saved.write_text(build_header(identity) + "\n" + encode_record(record) + "\n")
    with pytest.raises(ValueError):
        analysis.analyze(normalized, saved)


def test_report_and_cli_are_privacy_safe_and_inputs_unchanged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The CLI must work even when model runtimes cannot be imported.
    monkeypatch.setitem(sys.modules, "gliner", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    normalized, saved = make_inputs(tmp_path, [("PERSON", 2, 8)], [])
    before = normalized.read_bytes(), saved.read_bytes()
    output = tmp_path / "report.json"
    args = ["--normalized", str(normalized), "--predictions", str(saved), "--output", str(output)]
    assert analysis.main(args) == 0
    report_text = output.read_text()
    for forbidden in (
        "synthetic_private_sentinel", "private_metadata_sentinel", "private_header_sentinel",
        '"text"', '"start"', '"end"', '"example_index"', '"native_entities"',
    ):
        assert forbidden not in report_text
    assert capsys.readouterr().out == ""
    assert before == (normalized.read_bytes(), saved.read_bytes())
    assert analysis.main(args) == 2  # Never overwrite a previous report.
    assert output.read_text() == report_text


def test_cli_does_not_echo_invalid_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    normalized, saved = make_inputs(tmp_path, [], [])
    normalized.write_text('{"text": "private_error_sentinel", broken}\n')
    identity = json.loads(saved.read_text().split("\n")[0])["identity"]
    identity["normalized_sha256"] = sha256_file(normalized)
    saved.write_text(
        build_header(identity) + "\n" + encode_record(PredictionRecord(0, 1, 0, ())) + "\n"
    )
    output = tmp_path / "report.json"
    assert analysis.main([
        "--normalized", str(normalized), "--predictions", str(saved), "--output", str(output)
    ]) == 2
    captured = capsys.readouterr()
    assert "private_error_sentinel" not in captured.err
    assert not output.exists()


def test_repository_output_is_rejected(tmp_path: Path) -> None:
    normalized, saved = make_inputs(tmp_path, [], [])
    output = Path(__file__).resolve().parents[1] / "analysis_report.json"
    with pytest.raises(ValueError, match="outside repository"):
        analysis.run(normalized, saved, output)