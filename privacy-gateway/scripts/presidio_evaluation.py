"""Privacy-safe exact-span evaluation adapter for Presidio and nervaluate."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from nervaluate import Evaluator

from app.models.privacy import EntityType

if TYPE_CHECKING:
    # Type-only: scoring must not import Presidio/spaCy, so detector-independent
    # adapters (e.g. GLiNER) can reuse this harness without the Phase 1 runtime.
    from app.detection import PIIDetector

NormalizedEntity = dict[str, int | str]

# Nemotron's normalized DATE and TIME categories have no direct Phase 1 entity
# type. This adapter preserves their offsets and scores them against Presidio's
# combined DATE_TIME output without changing gateway detector behavior.
EVALUATION_TYPE_MAP = {"DATE": "DATE_TIME", "TIME": "DATE_TIME"}
SUPPORTED_TYPES = {item.value for item in EntityType}


@dataclass(frozen=True)
class ExactMatch:
    """A hashable, privacy-safe entity used only for offset/type comparisons."""

    entity_type: str
    start: int
    end: int


def evaluation_type(entity_type: str) -> str:
    """Map normalized dataset types to Phase 1's comparable entity types."""
    return EVALUATION_TYPE_MAP.get(entity_type, entity_type)


def adapter_category(entity_type: str) -> str:
    """Explain whether a label is direct, adapter-mapped, or unsupported."""
    if entity_type == "UNMAPPED":
        return "unmapped_source_label"
    if entity_type in EVALUATION_TYPE_MAP:
        return "adapter_mapped_date_time"
    if entity_type in SUPPORTED_TYPES:
        return "supported_direct"
    return "unsupported_entity"


def validate_normalized_example(example: Mapping[str, object]) -> tuple[str, list[ExactMatch]]:
    """Validate a normalized JSONL record without logging its plaintext text."""
    text = example.get("text")
    entities = example.get("entities")
    if not isinstance(text, str) or not isinstance(entities, list):
        raise ValueError("each record must contain text and an entities list")
    normalized: list[ExactMatch] = []
    for entity in entities:
        if not isinstance(entity, Mapping):
            raise ValueError("each entity must be an object")
        entity_type, start, end = entity.get("type"), entity.get("start"), entity.get("end")
        if (
            not isinstance(entity_type, str)
            or isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end <= start
            or end > len(text)
        ):
            raise ValueError("entity type or offsets are invalid")
        normalized.append(ExactMatch(evaluation_type(entity_type), start, end))
    return text, sorted(normalized, key=lambda item: (item.start, item.end, item.entity_type))


class PresidioEvaluationAdapter:
    """Adapt the runtime detector to the normalized offset-only representation."""

    def __init__(self, detector: PIIDetector) -> None:
        self._detector = detector

    def predict(self, text: str) -> list[ExactMatch]:
        """Return type/start/end predictions, never persisting detected values."""
        return [
            ExactMatch(span.entity_type.value, span.start, span.end)
            for span in self._detector.detect(text)
        ]


def as_harness_entities(entities: Sequence[ExactMatch]) -> list[dict[str, int | str]]:
    """Convert half-open offsets to nervaluate's dictionary-loader shape."""
    return [
        {"label": entity.entity_type, "start": entity.start, "end": entity.end}
        for entity in entities
    ]


def build_error_summary(
    example_index: int, truth: Sequence[ExactMatch], predicted: Sequence[ExactMatch]
) -> tuple[list[dict[str, int | str]], list[dict[str, int | str]]]:
    """Create offset-only false-negative/false-positive records for one example."""
    truth_set, predicted_set = set(truth), set(predicted)
    false_negatives: list[dict[str, int | str]] = []
    false_positives: list[dict[str, int | str]] = []
    for entity in sorted(
        truth_set - predicted_set, key=lambda item: (item.start, item.end, item.entity_type)
    ):
        has_overlap = any(
            candidate.entity_type == entity.entity_type
            and candidate.start < entity.end
            and entity.start < candidate.end
            for candidate in predicted_set
        )
        failure_group = "span_mismatch" if has_overlap else adapter_category(entity.entity_type)
        false_negatives.append(
            {
                "example_index": example_index,
                "entity_type": entity.entity_type,
                "start": entity.start,
                "end": entity.end,
                "failure_group": failure_group,
            }
        )
    for entity in sorted(
        predicted_set - truth_set, key=lambda item: (item.start, item.end, item.entity_type)
    ):
        has_overlap = any(
            candidate.entity_type == entity.entity_type
            and candidate.start < entity.end
            and entity.start < candidate.end
            for candidate in truth_set
        )
        failure_group = "span_mismatch" if has_overlap else "detector_recognizer"
        false_positives.append(
            {
                "example_index": example_index,
                "entity_type": entity.entity_type,
                "start": entity.start,
                "end": entity.end,
                "failure_group": failure_group,
            }
        )
    return false_negatives, false_positives


def harness_strict_metrics(
    truth: list[list[ExactMatch]], predicted: list[list[ExactMatch]], tags: list[str]
) -> dict[str, dict[str, int | float]]:
    """Run nervaluate strict matching on a bounded batch of independent documents."""
    result = Evaluator(
        [as_harness_entities(entities) for entities in truth],
        [as_harness_entities(entities) for entities in predicted],
        tags=tags,
        loader="dict",
    ).evaluate()
    strict = result["entities"]
    return {
        entity_type: {
            "correct": metrics["strict"].correct,
            "incorrect": metrics["strict"].incorrect,
            "missed": metrics["strict"].missed,
            "spurious": metrics["strict"].spurious,
        }
        for entity_type, metrics in strict.items()
    }


def metric_rows(
    truth_by_type: Counter[str], predicted_by_type: Counter[str], matches_by_type: Counter[str]
) -> dict[str, dict[str, int | float]]:
    """Calculate exact-match metrics, including types with no predictions."""
    rows: dict[str, dict[str, int | float]] = {}
    for entity_type in sorted(set(truth_by_type) | set(predicted_by_type)):
        support = truth_by_type.get(entity_type, 0)
        predictions = predicted_by_type.get(entity_type, 0)
        matches = matches_by_type.get(entity_type, 0)
        false_positives = predictions - matches
        false_negatives = support - matches
        precision = matches / predictions if predictions else 0.0
        recall = matches / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows[entity_type] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "exact_matches": matches,
            "predictions": predictions,
        }
    return rows


def summarize_groups(records: Iterable[Mapping[str, object]]) -> dict[str, int]:
    """Aggregate privacy-safe failures by their diagnostic category."""
    return dict(sorted(Counter(str(record["failure_group"]) for record in records).items()))
