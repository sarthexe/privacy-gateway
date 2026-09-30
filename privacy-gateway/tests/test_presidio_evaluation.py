"""Tests for the isolated Presidio evaluation adapter and strict harness."""

from __future__ import annotations

from app.models.privacy import EntitySpan, EntityType
from scripts.presidio_evaluation import (
    ExactMatch,
    PresidioEvaluationAdapter,
    build_error_summary,
    harness_strict_metrics,
    metric_rows,
    validate_normalized_example,
)


class FixedDetector:
    """Detector fake that exposes no entity value outside the model object."""

    def detect(self, text: str) -> list[EntitySpan]:
        return [
            EntitySpan(
                entity_type=EntityType.EMAIL_ADDRESS,
                start=0,
                end=4,
                value=text[:4],
            )
        ]


def test_adapter_preserves_detector_offsets_without_values() -> None:
    predictions = PresidioEvaluationAdapter(FixedDetector()).predict("safe text")
    assert predictions == [ExactMatch("EMAIL_ADDRESS", 0, 4)]


def test_normalized_date_is_compared_to_phase_one_date_time() -> None:
    _, truth = validate_normalized_example(
        {"text": "date", "entities": [{"type": "DATE", "start": 0, "end": 4}]}
    )
    assert truth == [ExactMatch("DATE_TIME", 0, 4)]


def test_harness_strict_matching_requires_type_and_offsets() -> None:
    truth = [[ExactMatch("PERSON", 0, 4)]]
    predicted = [[ExactMatch("PERSON", 0, 3), ExactMatch("EMAIL_ADDRESS", 5, 9)]]
    metrics = harness_strict_metrics(truth, predicted, ["PERSON", "EMAIL_ADDRESS"])
    assert metrics["PERSON"] == {"correct": 0, "incorrect": 1, "missed": 0, "spurious": 0}
    rows = metric_rows({"PERSON": 1}, {"PERSON": 1, "EMAIL_ADDRESS": 1}, {"PERSON": 0})
    assert rows["PERSON"]["false_negatives"] == 1
    assert rows["EMAIL_ADDRESS"]["false_positives"] == 1


def test_error_records_never_contain_source_text_or_values() -> None:
    false_negatives, false_positives = build_error_summary(
        7, [ExactMatch("PERSON", 1, 5)], [ExactMatch("EMAIL_ADDRESS", 1, 5)]
    )
    assert false_negatives == [
        {
            "example_index": 7,
            "entity_type": "PERSON",
            "start": 1,
            "end": 5,
            "failure_group": "supported_direct",
        }
    ]
    assert false_positives[0]["failure_group"] == "detector_recognizer"
