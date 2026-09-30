from __future__ import annotations

import pytest

from scripts.gliner_evaluation import GlinerLabelMap
from scripts.ood_pii import OodNormalizationError, normalize


MODEL = GlinerLabelMap(
    "test", "test", ("email", "phone_number", "first_name", "date", "ipv4"),
    {"email": "EMAIL_ADDRESS", "phone_number": "PHONE_NUMBER", "first_name": "PERSON", "date": "DATE", "ipv4": "IP_ADDRESS"}, {},
)


def test_ai4privacy_uses_verified_character_offsets_and_language() -> None:
    record = {"source_text": "mail a@b.co", "language": "English", "privacy_mask": [{"value": "a@b.co", "start": 5, "end": 11, "label": "EMAIL"}]}
    normalized = normalize("ai4privacy", record, 3, "validation", MODEL)
    assert normalized["example_index"] == 3 and normalized["entities"] == [{"type": "EMAIL_ADDRESS", "start": 5, "end": 11}]
    assert normalized["native_entities"][0]["native_label"] == "email" and normalized["metadata"]["language"] == "English"


def test_gretel_reconstructs_only_a_unique_exact_value_and_maps_label() -> None:
    normalized = normalize("gretel", {"text": "Call 555-0100", "entities": [{"entity": "555-0100", "types": ["phone_number"]}]}, 0, "test", MODEL)
    assert normalized["entities"] == [{"type": "PHONE_NUMBER", "start": 5, "end": 13}]
    with pytest.raises(OodNormalizationError, match="uniquely"):
        normalize("gretel", {"text": "x x", "entities": [{"entity": "x", "types": ["first_name"]}]}, 0, "test", MODEL)


def test_argilla_uses_pii_suggestion_offsets_and_unmaps_unknown_labels() -> None:
    normalized = normalize("argilla", {"source-text": "Ada 1.2.3.4", "language": "en", "pii.suggestion": [{"start": 0, "end": 3, "label": "FIRSTNAME"}, {"start": 4, "end": 11, "label": "IPV4"}, {"start": 0, "end": 3, "label": "JOBAREA"}]}, 1, "train", MODEL)
    assert normalized["entities"] == [{"type": "PERSON", "start": 0, "end": 3}, {"type": "IP_ADDRESS", "start": 4, "end": 11}, {"type": "UNMAPPED", "start": 0, "end": 3}]


def test_ai4privacy_fails_closed_when_span_value_disagrees() -> None:
    with pytest.raises(OodNormalizationError, match="does not match"):
        normalize("ai4privacy", {"source_text": "abc", "privacy_mask": [{"value": "z", "start": 0, "end": 1}]}, 0, "train", MODEL)
