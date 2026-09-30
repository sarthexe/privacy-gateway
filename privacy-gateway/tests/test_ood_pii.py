from __future__ import annotations

import pytest

from scripts.gliner_evaluation import GlinerLabelMap
from scripts.ood_pii import OodNormalizationError, normalize
from scripts.prepare_ood_pii import format_normalization_error


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
    normalized = normalize(
        "gretel",
        {
            "text": "Call 555-0100",
            "entities": "[{'entity': '555-0100', 'types': ['phone_number']}]",
        },
        0,
        "test",
        MODEL,
    )
    assert normalized["entities"] == [{"type": "PHONE_NUMBER", "start": 5, "end": 13}]


def test_gretel_rejects_malformed_or_non_list_serialized_entities() -> None:
    with pytest.raises(OodNormalizationError, match="serialized entities are invalid"):
        normalize("gretel", {"text": "a", "entities": "not a list"}, 0, "test", MODEL)
    with pytest.raises(OodNormalizationError, match="entities must be a list"):
        normalize("gretel", {"text": "a", "entities": "{'entity': 'a'}"}, 0, "test", MODEL)


def test_gretel_fails_closed_for_repeated_or_absent_entity_value() -> None:
    with pytest.raises(OodNormalizationError, match="uniquely"):
        normalize(
            "gretel",
            {"text": "x x", "entities": "[{'entity': 'x', 'types': ['first_name']}]"},
            0,
            "test",
            MODEL,
        )
    with pytest.raises(OodNormalizationError, match="uniquely"):
        normalize(
            "gretel",
            {"text": "present", "entities": "[{'entity': 'absent', 'types': ['first_name']}]"},
            0,
            "test",
            MODEL,
        )


def test_argilla_uses_pii_suggestion_offsets_and_unmaps_unknown_labels() -> None:
    normalized = normalize("argilla", {"source-text": "Ada 1.2.3.4", "language": "en", "pii.suggestion": [{"start": 0, "end": 3, "label": "FIRSTNAME"}, {"start": 4, "end": 11, "label": "IPV4"}, {"start": 0, "end": 3, "label": "JOBAREA"}]}, 1, "train", MODEL)
    assert normalized["entities"] == [{"type": "PERSON", "start": 0, "end": 3}, {"type": "IP_ADDRESS", "start": 4, "end": 11}, {"type": "UNMAPPED", "start": 0, "end": 3}]


def test_ai4privacy_fails_closed_when_span_value_disagrees() -> None:
    with pytest.raises(OodNormalizationError, match="does not match"):
        normalize("ai4privacy", {"source_text": "abc", "privacy_mask": [{"value": "z", "start": 0, "end": 1}]}, 0, "train", MODEL)


def test_normalization_error_message_has_safe_dataset_split_and_index_context() -> None:
    message = format_normalization_error(
        "ai4privacy", "validation", 12, OodNormalizationError("offsets are outside text")
    )
    assert message == (
        "error: normalization failed dataset=ai4privacy split=validation "
        "example_index=12: offsets are outside text"
    )
    assert "source_text" not in message and "value" not in message
