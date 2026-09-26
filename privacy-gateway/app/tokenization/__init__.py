"""Span-based, reversible PII tokenization."""

import secrets
from typing import Protocol

from app.detection import PIIDetector
from app.models.privacy import EntitySpan, EntityType, TokenizationResult, TokenMapping


class Tokenizer(Protocol):
    """Interface for replacing detected entities with reversible tokens."""

    def tokenize(self, text: str) -> TokenizationResult:
        """Replace PII spans while preserving all other source characters."""


class SpanTokenizer:
    """Tokenize detector spans by rebuilding text from original offsets."""

    def __init__(self, detector: PIIDetector) -> None:
        self._detector = detector

    def tokenize(self, text: str) -> TokenizationResult:
        spans = self._select_non_overlapping(self._detector.detect(text), text)
        mappings: dict[str, TokenMapping] = {}
        value_tokens: dict[tuple[EntityType, str], str] = {}
        output: list[str] = []
        cursor = 0

        for span in spans:
            output.append(text[cursor : span.start])
            key = (span.entity_type, span.value)
            token = value_tokens.get(key)
            if token is None:
                token = f"<PII_{span.entity_type.value}:{secrets.token_hex(16)}>"
                value_tokens[key] = token
                mappings[token] = TokenMapping(
                    token=token,
                    entity_type=span.entity_type,
                    value=span.value,
                )
            output.append(token)
            cursor = span.end

        output.append(text[cursor:])
        return TokenizationResult(text="".join(output), mappings=mappings)

    @staticmethod
    def _select_non_overlapping(spans: list[EntitySpan], text: str) -> list[EntitySpan]:
        valid = [
            span
            for span in spans
            if span.end <= len(text) and text[span.start : span.end] == span.value
        ]
        ordered = sorted(valid, key=lambda span: (span.start, -(span.end - span.start)))
        selected: list[EntitySpan] = []
        cursor = 0
        for span in ordered:
            if span.start < cursor:
                continue
            selected.append(span)
            cursor = span.end
        return selected
