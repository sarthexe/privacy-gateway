"""Tests for privacy-safe Nemotron-PII parsing and normalization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.nemotron_pii import (
    AnnotationError,
    inspect_split,
    load_ontology,
    normalize_example,
    parse_spans,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "pii" / "synthetic_nemotron.json"


def fixture_records() -> list[dict[str, object]]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_normalization_preserves_character_spans_deterministically() -> None:
    records = fixture_records()
    ontology = load_ontology()
    first = normalize_example(records[0], ontology)
    assert first == normalize_example(records[0], ontology)
    assert first["entities"] == [
        {"type": "PERSON", "start": 0, "end": 8},
        {"type": "EMAIL_ADDRESS", "start": 17, "end": 38},
    ]


def test_unknown_source_label_is_retained_as_unmapped() -> None:
    normalized = normalize_example(fixture_records()[2], load_ontology())
    assert normalized["entities"] == [{"type": "UNMAPPED", "start": 17, "end": 22}]


@pytest.mark.parametrize(
    "record",
    [
        {"text": "short", "spans": [{"start": 0, "end": 6, "label": "PERSON"}]},
        {"text": "short", "spans": [{"start": 2, "end": 2, "label": "PERSON"}]},
        {"text": "short", "spans": "invalid"},
    ],
)
def test_malformed_annotations_fail(record: dict[str, object]) -> None:
    with pytest.raises(AnnotationError):
        normalize_example(record, load_ontology())


def test_inspection_reports_structure_and_labels_without_text() -> None:
    report = inspect_split(fixture_records(), ["text", "spans"])
    assert report["rows"] == 5
    assert report["entity_label_counts"]["first_name"] == 1
    assert report["annotation_structure"] == {"end": "int", "label": "str", "start": "int"}


def test_dataset_serialized_span_schema_is_parsed() -> None:
    spans = parse_spans("[{'start': 0, 'end': 4, 'text': 'safe', 'label': 'first_name'}]")
    assert spans[0]["label"] == "first_name"
