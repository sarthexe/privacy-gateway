"""Tests for reversible span based privacy transformations."""

import re

import pytest

from app.detection import PIIDetector, PresidioPIIDetector
from app.models.privacy import EntitySpan, EntityType, TokenMapping
from app.reconstruction import Detokenizer
from app.tokenization import SpanTokenizer


class FixedDetector:
    """A predictable detector fixture that keeps PII out of test output."""

    def __init__(self, spans: list[EntitySpan]) -> None:
        self._spans = spans

    def detect(self, text: str) -> list[EntitySpan]:
        del text
        return self._spans


def span(text: str, value: str, entity_type: EntityType, start: int = 0) -> EntitySpan:
    return EntitySpan(
        entity_type=entity_type,
        start=start,
        end=start + len(value),
        value=value,
    )


def transform(text: str, spans: list[EntitySpan]) -> tuple[str, dict[str, TokenMapping]]:
    result = SpanTokenizer(FixedDetector(spans)).tokenize(text)
    return result.text, result.mappings


def test_no_pii_preserves_text() -> None:
    text = "Ordinary text, punctuation!\n第二行."
    tokenized, mappings = transform(text, [])
    assert tokenized == text
    assert mappings == {}


def test_single_entity_round_trip() -> None:
    text = "Contact Casey."
    tokenized, mappings = transform(text, [span(text, "Casey", EntityType.PERSON, 8)])
    assert re.fullmatch(r"Contact <PII_PERSON:[0-9a-f]{32}>\.", tokenized)
    assert Detokenizer().detokenize(tokenized, mappings) == text


def test_multiple_entity_types_and_punctuation() -> None:
    text = "Email: a@b.com, phone: 555-0100."
    spans = [
        span(text, "a@b.com", EntityType.EMAIL_ADDRESS, 7),
        span(text, "555-0100", EntityType.PHONE_NUMBER, 23),
    ]
    tokenized, mappings = transform(text, spans)
    assert "Email: <PII_EMAIL_ADDRESS:" in tokenized
    assert ", phone: <PII_PHONE_NUMBER:" in tokenized
    assert tokenized.endswith(".")
    assert Detokenizer().detokenize(tokenized, mappings) == text


def test_repeated_same_entity_reuses_identity() -> None:
    text = "Alex met Alex."
    spans = [span(text, "Alex", EntityType.PERSON, 0), span(text, "Alex", EntityType.PERSON, 9)]
    tokenized, mappings = transform(text, spans)
    tokens = re.findall(r"<PII_PERSON:[0-9a-f]{32}>", tokenized)
    assert len(tokens) == 2
    assert tokens[0] == tokens[1]
    assert len(mappings) == 1
    assert Detokenizer().detokenize(tokenized, mappings) == text


def test_different_entities_get_different_tokens() -> None:
    text = "Amy met Bob."
    spans = [span(text, "Amy", EntityType.PERSON), span(text, "Bob", EntityType.PERSON, 8)]
    tokenized, mappings = transform(text, spans)
    assert len(mappings) == 2
    assert len(set(mappings)) == 2
    assert Detokenizer().detokenize(tokenized, mappings) == text


def test_multiline_and_unicode_round_trip() -> None:
    text = "🙂 Ada\nmet 東京, email: 名@example.jp"
    spans = [
        span(text, "Ada", EntityType.PERSON, 2),
        span(text, "東京", EntityType.LOCATION, 10),
        span(text, "名@example.jp", EntityType.EMAIL_ADDRESS, 21),
    ]
    tokenized, mappings = transform(text, spans)
    assert "🙂 <PII_PERSON:" in tokenized
    assert "met <PII_LOCATION:" in tokenized
    assert Detokenizer().detokenize(tokenized, mappings) == text


def test_adjacent_entities() -> None:
    text = "AdaBob"
    tokenized, mappings = transform(
        text,
        [span(text, "Ada", EntityType.PERSON), span(text, "Bob", EntityType.PERSON, 3)],
    )
    assert re.fullmatch(r"<PII_PERSON:[0-9a-f]{32}><PII_PERSON:[0-9a-f]{32}>", tokenized)
    assert len(mappings) == 2
    assert Detokenizer().detokenize(tokenized, mappings) == text


@pytest.mark.parametrize(
    "text,start,value",
    [("Casey ends", 0, "Casey"), ("ends Casey", 5, "Casey")],
)
def test_beginning_and_end_boundaries(text: str, start: int, value: str) -> None:
    tokenized, mappings = transform(text, [span(text, value, EntityType.PERSON, start)])
    assert Detokenizer().detokenize(tokenized, mappings) == text


def test_unknown_valid_token_is_left_unchanged() -> None:
    token = "<PII_PERSON:00000000000000000000000000000000>"
    assert Detokenizer().detokenize(f"hello {token}", {}) == f"hello {token}"


def test_malformed_token_is_left_unchanged() -> None:
    malformed = "<PII_PERSON:not-a-valid-id>"
    assert Detokenizer().detokenize(malformed, {}) == malformed


def test_overlapping_spans_choose_longest_at_first_offset() -> None:
    text = "abcde"
    tokenized, mappings = transform(
        text,
        [span(text, "abc", EntityType.PERSON), span(text, "abcde", EntityType.LOCATION)],
    )
    assert tokenized.startswith("<PII_LOCATION:")
    assert len(mappings) == 1
    assert Detokenizer().detokenize(tokenized, mappings) == text


def test_presidio_detector_uses_configured_entity_types() -> None:
    class Analyzer:
        def analyze(self, *, text: str, language: str, entities: list[str]) -> list[object]:
            assert language == "en"
            assert set(entities) == {item.value for item in EntityType}
            return []

    detector: PIIDetector = PresidioPIIDetector(analyzer=Analyzer())  # type: ignore[arg-type]
    assert detector.detect("no entities") == []
