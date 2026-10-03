from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_gliner_ood_errors import analyze
from scripts.evaluation_checkpoint import sha256_file
from scripts.gliner_credit_card_experiment import (
    CARD,
    Candidate,
    CardExperimentError,
    generate_candidates,
    luhn,
    main,
    merge_candidates,
    run_experiment,
    validation,
)
from scripts.gliner_evaluation import RawPrediction, load_label_map
from scripts.gliner_phone_hybrid_experiment import Span
from scripts.gliner_predictions import PredictionRecord, build_header, encode_record

# Published processor sandbox fixtures, not real account numbers.
VALID = "4111111111111111"
INVALID = "4111111111111112"


@pytest.mark.parametrize("value", [VALID, "4111 1111 1111 1111", "4111-1111-1111-1111"])
def test_synthetic_valid_cards_and_separator_normalization(value: str) -> None:
    result = generate_candidates(value)
    assert result.accepted == (Candidate(Span(0, len(value)), "checksum_valid", False),)
    assert result.stats["length_valid"] == result.stats["checksum_valid"] == 1


def test_amex_grouping_supported() -> None:
    assert validation("3782 822463 10005")[0] == "checksum_valid"


@pytest.mark.parametrize("length", [1, 3, 12, 20, 129])
def test_length_rejected(length: int) -> None:
    assert not generate_candidates("0" * length).accepted
    assert validation("0" * length)[0] == "unable_to_validate"


def test_luhn_invalid_requires_explicit_card_context() -> None:
    assert luhn(VALID) and not luhn(INVALID)
    assert not generate_candidates(INVALID).accepted
    contextual = generate_candidates("credit card number: " + INVALID)
    assert contextual.accepted[0].checksum == "checksum_invalid"
    assert contextual.stats["accepted_checksum_invalid"] == 1


@pytest.mark.parametrize("value", [
    "4111--1111-1111-1111", "4111 1111-1111 1111", "+4111111111111111",
    "*4111111111111111", "1/4111111111111111", "4111111111111111.2",
])
def test_malformed_or_connected_fragments_rejected(value: str) -> None:
    assert not generate_candidates(value).accepted


def test_context_does_not_cross_line_or_use_generic_card_word() -> None:
    for prefix in ["credit card:\n", "ID card: ", "card: ", "credit card earlier words "]:
        assert not generate_candidates(prefix + INVALID).accepted


def test_context_window_cannot_start_inside_a_word() -> None:
    label = "credit card: "
    prefix = "mis" + label + " " * (64 - len(label))
    valid = generate_candidates(prefix + VALID)
    assert len(valid.accepted) == 1 and not valid.accepted[0].explicit_context
    assert not generate_candidates(prefix + INVALID).accepted


def test_boundary_preservation_unicode_and_punctuation() -> None:
    text = "測試 credit card: " + VALID + ", done"
    candidate = generate_candidates(text).accepted[0]
    start = text.index(VALID)
    assert candidate.span == Span(start, start + len(VALID))
    assert text[candidate.span.start:candidate.span.end] == VALID


def merge(text: str, old: tuple[RawPrediction, ...], arbitration: bool = False):
    return merge_candidates(
        text, old, generate_candidates(text).accepted, load_label_map(), arbitration
    )


@pytest.mark.parametrize("label", ["credit_debit_card", "cvv"])
def test_exact_card_alias_duplicate_skipped(label: str) -> None:
    old = (RawPrediction(label, 0, len(VALID), 0.8),)
    result = merge(VALID, old)
    assert result.predictions == old and result.stats["exact_duplicates_skipped"] == 1


def test_duplicate_candidate_objects_removed() -> None:
    candidate = generate_candidates(VALID).accepted[0]
    result = merge_candidates(VALID, (), [candidate, candidate], load_label_map())
    assert len(result.added) == 1
    assert result.stats["candidate_duplicates_removed"] == 1


@pytest.mark.parametrize("label", [
    "account_number", "national_id", "customer_id", "employee_id",
])
def test_exact_contextual_bank_and_id_conflicts(label: str) -> None:
    text = "credit card: " + VALID
    span = generate_candidates(text).accepted[0].span
    old = (RawPrediction(label, span.start, span.end, 0.9),)
    a, b = merge(text, old), merge(text, old, True)
    assert a.predictions[0] == old[0] and not a.suppressed
    assert b.suppressed == old
    assert b.stats["predictions_suppressed"] == b.stats["conflicts_resolved"] == 1
    assert len(b.added) == len(a.added) == 1
    assert b.predictions[0].label == "credit_debit_card"


@pytest.mark.parametrize("label", ["phone_number", "bank_routing_number", "swift_bic", "pin"])
def test_protected_direct_conflicts_reject_arbitration(label: str) -> None:
    text = "credit card: " + VALID
    span = generate_candidates(text).accepted[0].span
    old = (RawPrediction(label, span.start, span.end, 0.9),)
    result = merge(text, old, True)
    assert result.predictions[:1] == old and not result.suppressed
    assert result.stats["rejected_as_ambiguous"] == 1


def test_partial_overlap_is_ambiguous_and_no_partial_suppression() -> None:
    text = "credit card: " + VALID
    span = generate_candidates(text).accepted[0].span
    old = (
        RawPrediction("account_number", span.start, span.end, 0.9),
        RawPrediction("national_id", span.start + 1, span.end, 0.8),
    )
    result = merge(text, old, True)
    assert not result.suppressed and result.predictions[:2] == old
    assert result.stats["rejected_as_ambiguous"] == 1


def test_luhn_without_context_is_not_enough_to_suppress_bank() -> None:
    old = (RawPrediction("account_number", 0, len(VALID), 0.9),)
    result = merge(VALID, old, True)
    assert not result.suppressed and result.predictions[:1] == old


def test_contextual_invalid_checksum_is_not_enough_to_suppress_id() -> None:
    text = "credit card: " + INVALID
    span = generate_candidates(text).accepted[0].span
    old = (RawPrediction("national_id", span.start, span.end, 0.9),)
    result = merge(text, old, True)
    assert not result.suppressed and len(result.added) == 1


def test_unrelated_objects_scores_order_and_multiplicity_preserved() -> None:
    text = "Name credit card: " + VALID
    span = generate_candidates(text).accepted[0].span
    name = RawPrediction("first_name", 0, 4, 0.73)
    old = (name, RawPrediction("account_number", span.start, span.end, 0.9), name)
    result = merge(text, old, True)
    assert result.predictions[:2] == (name, name)
    assert result.predictions[0] is name and result.predictions[1] is name


@pytest.mark.parametrize("bad", [
    RawPrediction("account_number", -1, 3, 0.9),
    RawPrediction("account_number", 0, 16, float("nan")),
    RawPrediction("unknown", 0, 3, 0.9),
    RawPrediction("account_number", True, 3, 0.9),
])
def test_malformed_model_input_fails_closed(bad: RawPrediction) -> None:
    with pytest.raises(CardExperimentError):
        merge(VALID, (bad,))


def test_malformed_text_and_forged_evidence_fail_closed() -> None:
    with pytest.raises(CardExperimentError):
        generate_candidates(None)  # type: ignore[arg-type]
    with pytest.raises(CardExperimentError):
        merge_candidates(
            VALID, (), [Candidate(Span(0, len(VALID)), "checksum_invalid", False)],
            load_label_map(),
        )
    with pytest.raises(CardExperimentError):
        luhn("synthetic-not-digits")


def inputs(tmp: Path, zero_gt: bool = False):
    text = "credit card: " + VALID + "\ncredit card: " + INVALID + "\nCVV: 123"
    first, second = text.index(VALID), text.index(INVALID)
    native = (
        [{"native_label": "account_number", "start": first, "end": first + len(VALID)}]
        if zero_gt else [
            {"native_label": "credit_debit_card", "start": first, "end": first + len(VALID)},
            {"native_label": "credit_debit_card", "start": second, "end": second + len(INVALID)},
            {"native_label": "cvv", "start": len(text) - 3, "end": len(text)},
        ]
    )
    mapping = load_label_map()
    path = tmp / "normalized.jsonl"
    path.write_text(json.dumps({
        "text": text, "native_entities": native,
        "entities": [{"type": mapping.mapping[e["native_label"]],
                      "start": e["start"], "end": e["end"]} for e in native],
    }) + "\n")
    paths, conditions, pinned = [], [], []
    for threshold in (0.3, 0.7):
        pred = tmp / f"pred-{threshold}.jsonl"
        pred.write_text(build_header({
            "threshold": threshold, "normalized_sha256": sha256_file(path),
            "model": {"model_id": mapping.model_id, "model_revision": mapping.model_revision,
                      "max_width": 24},
        }) + "\n" + encode_record(PredictionRecord(
            0, 1, 0, (RawPrediction("account_number", first, first + len(VALID), 0.9),)
        )) + "\n")
        old = analyze(path, pred)
        card = old["per_entity_type"].get(CARD, {
            "tp": 0, "fp": 0, "fn": 0, "precision": 0, "recall": 0, "f1": 0,
        })
        conditions.append({
            "threshold": threshold, "full_strict": old["full_strict"],
            "gateway_supported_only_strict": old["gateway_supported_only_strict"],
            "inventory": [{"type": CARD, **card}],
        })
        pinned.append({"threshold": threshold, "input_predictions_sha256": sha256_file(pred)})
        paths.append(pred)
    reference = tmp / "baseline.json"
    reference.write_text(json.dumps({"datasets": [{
        "dataset": "synthetic", "conditions": conditions,
    }]}))
    fingerprints = tmp / "fingerprints.json"
    fingerprints.write_text(json.dumps({"datasets": [{
        "dataset": "synthetic", "normalized_sha256": sha256_file(path), "conditions": pinned,
    }]}))
    return path, paths, reference, fingerprints


@pytest.mark.parametrize("zero_gt", [False, True])
def test_three_way_scoring_checksum_coverage_and_zero_support(
    tmp_path: Path, zero_gt: bool,
) -> None:
    path, preds, reference, fingerprints = inputs(tmp_path, zero_gt)
    hashes = [sha256_file(p) for p in [path, *preds, reference, fingerprints]]
    report = run_experiment(path, preds, reference, fingerprints, "synthetic", 0)
    assert report["zero_credit_card_gt_support"] == zero_gt
    assert report["gt_checksum_partition"] == (
        {"checksum_valid": 0, "checksum_invalid": 0, "unable_to_validate": 0} if zero_gt
        else {"checksum_valid": 1, "checksum_invalid": 1, "unable_to_validate": 1}
    )
    for c in report["conditions"]:
        assert c["arbitration"]["predictions_suppressed"] == 1
        assert c["arbitration"]["predictions_changed"] == 0
        assert c["credit_card"]["variant_a"] == c["credit_card"]["variant_b"]
        assert c["credit_card"]["variant_a"]["tp"] == (0 if zero_gt else 2)
        assert c["credit_card"]["variant_a"]["fp"] == (2 if zero_gt else 0)
        assert c["credit_card_luhn_only_additions_ablation"]["tp"] == (0 if zero_gt else 1)
        assert c["added_candidate_boundaries"].get(
            "no_credit_card_gt_overlap", 0
        ) == (2 if zero_gt else 0)
        if zero_gt:
            assert c["suppression_gt_consequences"]["BANK_ACCOUNT_exact_gt_matches_removed"] == 1
        else:
            assert c["credit_card"]["variant_a"]["fn"] == 1  # CVV remains in GT.
    encoded = json.dumps(report)
    assert VALID not in encoded and INVALID not in encoded
    for forbidden in ('"text"', '"start"', '"end"', '"example_index"'):
        assert forbidden not in encoded
    assert hashes == [sha256_file(p) for p in [path, *preds, reference, fingerprints]]


def test_prediction_hash_mismatch_rejected(tmp_path: Path) -> None:
    path, preds, reference, fingerprints = inputs(tmp_path)
    preds[0].write_text(preds[0].read_text() + "\n")
    with pytest.raises(CardExperimentError):
        run_experiment(path, preds, reference, fingerprints, "synthetic", 0)


def test_cli_does_not_overwrite_input(tmp_path: Path) -> None:
    path, preds, reference, fingerprints = inputs(tmp_path)
    before = path.read_bytes()
    assert main([
        "--dataset", "synthetic", "--normalized", str(path),
        "--predictions", str(preds[0]), "--predictions", str(preds[1]),
        "--baseline-report", str(reference), "--input-reference", str(fingerprints),
        "--report", str(path),
    ]) == 2
    assert path.read_bytes() == before