from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from scripts import evaluate_gliner_numeric_finetune as evaluate
from scripts.batch_evaluation import Batch, score_batch
from scripts.evaluation_checkpoint import EvaluationState, sha256_file
from scripts.gliner_evaluation import RawPrediction, to_gateway
from scripts.gliner_numeric_common import (
    OTHER_NUMERIC,
    NumericExperimentError,
    extended_label_map,
    load_config,
)
from scripts.gliner_predictions import PredictionRecord, PredictionWriter
from scripts.nemotron_pii import REPOSITORY_ROOT

CONFIG = load_config()
TEXT = "Card 4111 1111 1111 1111 for Ana at 10.0.0.1 ref 0042"
CARD = (TEXT.index("4111"), TEXT.index(" for"))
NAME = (TEXT.index("Ana"), TEXT.index("Ana") + 3)
IP = (TEXT.index("10.0"), TEXT.index(" ref"))
REF = (TEXT.index("0042"), len(TEXT))


def normalized(tmp_path: Path, rows: int = 3, with_other: bool = False) -> Path:
    path = tmp_path / "normalized.jsonl"
    native = [("credit_debit_card", *CARD), ("first_name", *NAME), ("ipv4", *IP)]
    if with_other:
        native.append((OTHER_NUMERIC, *REF))
    record = {
        "text": TEXT,
        "entities": [
            {
                "type": "UNMAPPED"
                if label == OTHER_NUMERIC
                else CONFIG.label_map.gateway_type(label),
                "start": start,
                "end": end,
            }
            for label, start, end in native
        ],
        "native_entities": [
            {"native_label": label, "start": start, "end": end} for label, start, end in native
        ],
    }
    path.write_text("".join(json.dumps(record) + "\n" for _ in range(rows)), encoding="utf-8")
    return path


def identity(data: Path, labels: str = "native", **changes: Any) -> dict[str, Any]:
    base = {
        "normalized_sha256": sha256_file(data),
        "model": {
            "model_id": CONFIG.label_map.model_id,
            "model_revision": CONFIG.label_map.model_revision,
            "dtype": "fp32",
        },
        "protocol": {
            "max_width": 24,
            "batch_size": 1,
            "thresholds": [0.3, 0.7],
            "email_hybrid": "not_run",
            "label_set": labels,
        },
        "threshold": 0.3,
    }
    base.update(changes)
    return base


def write(path: Path, header: dict[str, Any], per_row: list[list[tuple[str, int, int]]]) -> Path:
    writer = PredictionWriter(path, header, range(len(per_row)), False, False)
    for index, spans in enumerate(per_row):
        writer.append(
            [
                PredictionRecord(
                    index,
                    1,
                    0,
                    tuple(RawPrediction(label, start, end, 0.9) for label, start, end in spans),
                )
            ]
        )
    return path


ALL_RIGHT = [("credit_debit_card", *CARD), ("first_name", *NAME), ("ipv4", *IP)]


def test_other_numeric_is_never_scored_and_absorption_is_flagged(tmp_path: Path) -> None:
    data = normalized(tmp_path)
    baseline = write(tmp_path / "base.jsonl", identity(data), [ALL_RIGHT] * 3)
    absorbed = [(OTHER_NUMERIC, *CARD), ("first_name", *NAME), ("ipv4", *IP)]
    candidate = write(
        tmp_path / "cand.jsonl", identity(data, "native+other_numeric"), [absorbed] * 3
    )
    report = evaluate.compare(CONFIG, data, baseline, candidate)
    cand = report["candidate"]
    assert "OTHER_NUMERIC" not in cand["full_strict"]["per_type"]
    assert OTHER_NUMERIC not in cand["native_label_space"]["per_type"]
    assert cand["full_strict"]["per_type"]["CREDIT_CARD"]["tp"] == 0
    assert cand["full_strict"]["fp"] == 0
    assert cand["other_numeric"]["predictions"] == 3
    assert cand["other_numeric"]["mapped_gt_exact_span"] == 3
    gate = report["safety_gate"]
    assert gate["tp_lost_to_other_numeric"] == 3
    flags = {flag["flag"] for flag in gate["flags"]}
    assert {
        "other_numeric_absorbs_supported_entities",
        "true_positives_lost_to_other_numeric",
        "supported_only_f1_drop",
    } <= flags
    assert gate["verdict"] == "flagged"


def test_scores_match_the_shared_strict_scorer(tmp_path: Path) -> None:
    data = normalized(tmp_path, rows=1)
    spans = [("credit_debit_card", *CARD), ("last_name", *NAME), ("ipv4", IP[0], IP[1] - 1)]
    baseline = write(tmp_path / "base.jsonl", identity(data), [spans])
    candidate = write(tmp_path / "cand.jsonl", identity(data), [spans])
    report = evaluate.compare(CONFIG, data, baseline, candidate)
    assert report["baseline"] == report["candidate"]
    state = EvaluationState()
    row = json.loads(data.read_text(encoding="utf-8"))
    truth = [
        (evaluate.validate_normalized_example(row)[1][i].entity_type, e["start"], e["end"])
        for i, e in enumerate(row["entities"])
    ]
    kept = to_gateway(
        [RawPrediction(label, s, e, 0.9) for label, s, e in spans], CONFIG.label_map
    ).kept
    score_batch(
        state,
        Batch([(0, b"")], 0, False),
        [(tuple(truth), tuple((k.entity_type, k.start, k.end) for k in kept))],
    )
    totals = state.aggregate()
    full = report["baseline"]["full_strict"]
    assert (full["tp"], full["fp"], full["fn"]) == (totals["tp"], totals["fp"], totals["fn"])
    assert report["baseline"]["native_label_space"]["per_type"]["first_name"]["fn"] == 1


def test_forgetting_table_lists_every_native_label(tmp_path: Path) -> None:
    data = normalized(tmp_path)
    baseline = write(tmp_path / "base.jsonl", identity(data), [ALL_RIGHT] * 3)
    candidate = write(tmp_path / "cand.jsonl", identity(data), [ALL_RIGHT] * 3)
    report = evaluate.compare(CONFIG, data, baseline, candidate)
    assert list(report["forgetting"]) == list(CONFIG.native_labels)
    assert len(report["forgetting"]) == 55
    assert report["forgetting"]["credit_debit_card"]["f1_delta"] == 0.0
    assert report["forgetting"]["ssn"]["f1_delta"] is None
    assert report["safety_gate"]["verdict"] == "no_target_improvement"


def test_improvement_without_flags_still_needs_review(tmp_path: Path) -> None:
    data = normalized(tmp_path)
    missing_card = [("first_name", *NAME), ("ipv4", *IP)]
    baseline = write(tmp_path / "base.jsonl", identity(data), [missing_card] * 3)
    candidate = write(tmp_path / "cand.jsonl", identity(data), [ALL_RIGHT] * 3)
    gate = evaluate.compare(CONFIG, data, baseline, candidate)["safety_gate"]
    assert gate["flags"] == []
    assert gate["improved_target_types"] == ["CREDIT_CARD"]
    assert gate["verdict"] == "no_gate_flags_review_required"


def test_credit_card_gain_does_not_offset_a_regression(tmp_path: Path) -> None:
    data = normalized(tmp_path)
    baseline = write(
        tmp_path / "base.jsonl", identity(data), [[("first_name", *NAME), ("ipv4", *IP)]] * 3
    )
    candidate = write(tmp_path / "cand.jsonl", identity(data), [[("credit_debit_card", *CARD)]] * 3)
    gate = evaluate.compare(CONFIG, data, baseline, candidate)["safety_gate"]
    assert "CREDIT_CARD" in gate["improved_target_types"]
    assert gate["verdict"] == "flagged"


def test_other_numeric_ground_truth_is_reported_separately(tmp_path: Path) -> None:
    data = normalized(tmp_path, with_other=True)
    baseline = write(tmp_path / "base.jsonl", identity(data), [ALL_RIGHT] * 3)
    candidate = write(
        tmp_path / "cand.jsonl",
        identity(data, "native+other_numeric"),
        [[*ALL_RIGHT, (OTHER_NUMERIC, *REF)]] * 3,
    )
    report = evaluate.compare(CONFIG, data, baseline, candidate)
    assert report["candidate"]["other_numeric"]["vs_other_numeric_gt"]["tp"] == 3
    assert report["candidate"]["other_numeric"]["mapped_gt_overlapped"] == 0
    assert report["baseline"]["native_label_space"] == report["candidate"]["native_label_space"]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"threshold": 0.7}, "thresholds differ"),
        ({"normalized_sha256": "0" * 64}, "another dataset"),
        ({"protocol": {"max_width": 12, "batch_size": 1}}, "another protocol"),
        ({"model": {"model_id": "nvidia/gliner-PII", "model_revision": "main"}}, "pinned lineage"),
    ],
)
def test_incomparable_prediction_files_are_refused(
    tmp_path: Path, changes: dict[str, Any], message: str
) -> None:
    data = normalized(tmp_path)
    baseline = write(tmp_path / "base.jsonl", identity(data), [ALL_RIGHT] * 3)
    candidate = write(tmp_path / "cand.jsonl", identity(data, **changes), [ALL_RIGHT] * 3)
    with pytest.raises(NumericExperimentError, match=message):
        evaluate.compare(CONFIG, data, baseline, candidate)


def test_record_count_mismatch_is_refused(tmp_path: Path) -> None:
    data = normalized(tmp_path)
    baseline = write(tmp_path / "base.jsonl", identity(data), [ALL_RIGHT] * 3)
    candidate = write(tmp_path / "cand.jsonl", identity(data), [ALL_RIGHT] * 2)
    with pytest.raises(NumericExperimentError, match="counts differ"):
        evaluate.compare(CONFIG, data, baseline, candidate)


def test_gzip_baseline_without_label_set_is_read_as_native(tmp_path: Path) -> None:
    data = normalized(tmp_path)
    legacy = identity(data)
    legacy["protocol"] = {
        key: value for key, value in legacy["protocol"].items() if key != "label_set"
    }
    plain = write(tmp_path / "base.jsonl", legacy, [ALL_RIGHT] * 3)
    packed = tmp_path / "base.jsonl.gz"
    with plain.open("rb") as source, gzip.open(packed, "wb") as target:
        shutil.copyfileobj(source, target)
    assert evaluate.label_set(evaluate.read_prediction_identity(packed)) == "native"
    candidate = write(tmp_path / "cand.jsonl", identity(data), [ALL_RIGHT] * 3)
    report = evaluate.compare(CONFIG, data, packed, candidate)
    assert report["deltas"]["full_strict"] == 0.0


def test_native_label_files_reject_other_numeric_predictions(tmp_path: Path) -> None:
    data = normalized(tmp_path)
    baseline = write(tmp_path / "base.jsonl", identity(data), [ALL_RIGHT] * 3)
    candidate = write(
        tmp_path / "cand.jsonl", identity(data), [[*ALL_RIGHT, (OTHER_NUMERIC, *REF)]] * 3
    )
    with pytest.raises(ValueError, match="prediction label"):
        evaluate.compare(CONFIG, data, baseline, candidate)


def test_reports_and_summary_hold_no_text(tmp_path: Path) -> None:
    data = normalized(tmp_path)
    baseline = write(tmp_path / "base.jsonl", identity(data), [ALL_RIGHT] * 3)
    candidate = write(tmp_path / "cand.jsonl", identity(data), [ALL_RIGHT] * 3)
    report = evaluate.compare(CONFIG, data, baseline, candidate)
    summary, table = evaluate.summarize([report, report])
    for blob in (json.dumps(report), json.dumps(summary), table):
        for value in ("4111", "Ana", "10.0.0.1", TEXT):
            assert value not in blob
    assert "accepted" not in json.dumps(summary)
    assert table.count("\n") == 4


def test_label_sets_and_prompts() -> None:
    assert evaluate.resolve_label_set("auto", "baseline") == "native"
    assert evaluate.resolve_label_set("auto", "finetuned") == "native+other_numeric"
    extended = evaluate.prompt_label_map(CONFIG, "native+other_numeric")
    assert extended.labels == extended_label_map(CONFIG.label_map).labels
    assert extended.labels[-1] == OTHER_NUMERIC and len(extended.labels) == 56
    assert evaluate.prompt_label_map(CONFIG, "native").labels == CONFIG.native_labels


def test_predict_refuses_repository_output_before_loading(tmp_path: Path) -> None:
    data = normalized(tmp_path)
    args = evaluate.parse_args(
        [
            "predict",
            "--normalized",
            str(data),
            "--model",
            "base",
            "--output-dir",
            str(REPOSITORY_ROOT / "predictions"),
        ]
    )
    with pytest.raises(NumericExperimentError, match="outside the repository"):
        evaluate.predict(args)


def test_fine_tuned_lineage_is_checked(tmp_path: Path) -> None:
    model_dir = tmp_path / "final_model"
    model_dir.mkdir()
    with pytest.raises(NumericExperimentError, match="no lineage"):
        evaluate.load_lineage(model_dir, CONFIG)
    base = CONFIG.raw["base_model"]
    lineage = {
        "experiment": CONFIG.raw["experiment"],
        "base_model": {"model_id": base["model_id"], "revision": base["revision"]},
        "training_labels": list(CONFIG.training_labels),
    }
    (model_dir / evaluate.LINEAGE_FILE).write_text(json.dumps(lineage), encoding="utf-8")
    assert evaluate.load_lineage(model_dir, CONFIG)["experiment"] == CONFIG.raw["experiment"]
    lineage["base_model"]["revision"] = "0" * 40
    (model_dir / evaluate.LINEAGE_FILE).write_text(json.dumps(lineage), encoding="utf-8")
    with pytest.raises(NumericExperimentError, match="lineage"):
        evaluate.load_lineage(model_dir, CONFIG)
