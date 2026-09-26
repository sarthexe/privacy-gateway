"""PII detection interfaces and Presidio implementation."""

from typing import Protocol

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

from app.models.privacy import EntitySpan, EntityType


class PIIDetector(Protocol):
    """Interface for detectors that return source-relative entity spans."""

    def detect(self, text: str) -> list[EntitySpan]:
        """Return detected entities using half-open character offsets."""


class PresidioPIIDetector:
    """PII detector backed by Microsoft Presidio Analyzer."""

    _SUPPORTED = {item.value for item in EntityType}

    def __init__(self, analyzer: AnalyzerEngine | None = None) -> None:
        if analyzer is None:
            provider = NlpEngineProvider(
                nlp_configuration={
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
                }
            )
            analyzer = AnalyzerEngine(nlp_engine=provider.create_engine())
        self._analyzer = analyzer

    def detect(self, text: str) -> list[EntitySpan]:
        if not text:
            return []
        results = self._analyzer.analyze(
            text=text,
            language="en",
            entities=sorted(self._SUPPORTED),
        )
        spans = [
            EntitySpan(
                entity_type=EntityType(result.entity_type),
                start=result.start,
                end=result.end,
                value=text[result.start : result.end],
            )
            for result in results
            if result.entity_type in self._SUPPORTED
        ]
        return sorted(spans, key=lambda span: (span.start, span.end, span.entity_type.value))
