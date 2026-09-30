from __future__ import annotations

from scripts.evaluation_checkpoint import EvaluationState
from scripts.gliner_email_hybrid import _flush_scores, hybrid_predictions, recognize_email_addresses
from scripts.gliner_evaluation import GlinerLabelMap, RawPrediction
from scripts.presidio_evaluation import ExactMatch


LABEL_MAP = GlinerLabelMap(
    model_id="test",
    model_revision="test",
    labels=("email", "first_name", "user_name", "company_name", "url"),
    mapping={
        "email": "EMAIL_ADDRESS",
        "first_name": "PERSON",
        "user_name": "USERNAME",
        "company_name": "ORGANIZATION",
        "url": "UNMAPPED",
    },
    deviations={},
)


def spans(text: str) -> list[tuple[int, int]]:
    return [(item.start, item.end) for item in recognize_email_addresses(text)]


def test_email_recognizer_offsets_and_punctuation() -> None:
    text = "(first.last+tag@example.co.uk),"
    assert spans(text) == [(1, len(text) - 2)]
    assert text[slice(*spans(text)[0])] == "first.last+tag@example.co.uk"


def test_email_recognizer_handles_simple_subdomain_multiple_and_boundaries() -> None:
    text = "a@b.io then x@y.mail.example.org"
    assert [text[start:end] for start, end in spans(text)] == ["a@b.io", "x@y.mail.example.org"]
    assert spans("start@x.io") == [(0, 10)]
    assert spans("end@x.io") == [(0, 8)]


def test_hybrid_suppresses_only_contained_person_username_organization() -> None:
    text = "write ann.smith@example.com now"
    start = text.index("ann.smith@example.com")
    predictions = [
        RawPrediction("first_name", start, start + 3, 0.9),
        RawPrediction("user_name", start + 4, start + 9, 0.9),
        RawPrediction("company_name", start + 10, start + 17, 0.9),
        RawPrediction("url", start - 1, start + 4, 0.9),
    ]
    result, duplicates = hybrid_predictions(text, predictions, LABEL_MAP)
    assert duplicates == 0
    assert [(item.label, item.start, item.end) for item in result] == [
        ("url", start - 1, start + 4),
        ("email", start, start + 21),
    ]


def test_partial_overlap_is_not_suppressed_and_exact_predictions_are_deduplicated() -> None:
    text = "ab@example.com."
    email_end = len(text) - 1
    original = [
        RawPrediction("first_name", 0, len(text), 0.9),  # overlaps but ends outside the email
        RawPrediction("email", 0, email_end, 0.2),
        RawPrediction("email", 0, email_end, 0.8),
    ]
    result, duplicates = hybrid_predictions(text, original, LABEL_MAP)
    assert [(item.label, item.start, item.end, item.score) for item in result] == [
        ("email", 0, email_end, 1.0),
        ("first_name", 0, len(text), 0.9),
    ]
    assert duplicates == 2


def test_flush_scores_adapts_exact_matches_to_score_batch_tuple_api() -> None:
    original, hybrid = EvaluationState(), EvaluationState()
    pending = [
        (
            0,
            (ExactMatch("EMAIL_ADDRESS", 0, 8),),
            (ExactMatch("EMAIL_ADDRESS", 0, 8),),
            (ExactMatch("EMAIL_ADDRESS", 0, 8),),
        )
    ]

    _flush_scores(original, hybrid, pending, reached_eof=True)

    assert pending == []
    assert original.aggregate() == {"tp": 1, "fp": 0, "fn": 0}
    assert hybrid.aggregate() == {"tp": 1, "fp": 0, "fn": 0}
