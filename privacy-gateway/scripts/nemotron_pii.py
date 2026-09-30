"""Shared, privacy-safe helpers for the Nemotron-PII dataset scripts."""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

DATASET_ID = "nvidia/Nemotron-PII"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class AnnotationError(ValueError):
    """Raised when a source annotation cannot safely be normalized."""


def load_ontology(path: Path | None = None) -> dict[str, str]:
    """Load and validate the reviewed source-label-to-gateway-label mapping."""
    ontology_path = path or REPOSITORY_ROOT / "configs" / "pii_ontology.yaml"
    loaded = yaml.safe_load(ontology_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or not isinstance(loaded.get("labels"), dict):
        raise ValueError(f"Ontology at {ontology_path} must contain a labels mapping")
    labels = loaded["labels"]
    if not all(
        isinstance(source, str) and isinstance(target, str) for source, target in labels.items()
    ):
        raise ValueError(f"Ontology at {ontology_path} contains a non-string label")
    return dict(labels)


def parse_spans(raw_spans: object) -> list[Mapping[str, Any]]:
    """Parse Nemotron's Python-literal span column into annotation objects."""
    spans = raw_spans
    if isinstance(spans, str):
        try:
            spans = ast.literal_eval(spans)
        except (SyntaxError, ValueError) as error:
            raise AnnotationError("spans string is not a valid literal") from error
    if not isinstance(spans, list):
        raise AnnotationError("record spans must be a list or a serialized list")
    if not all(isinstance(span, Mapping) for span in spans):
        raise AnnotationError("each span must be an object")
    return spans


def normalize_example(record: Mapping[str, Any], ontology: Mapping[str, str]) -> dict[str, Any]:
    """Convert a Nemotron record while preserving validated character offsets."""
    text = record.get("text")
    if not isinstance(text, str):
        raise AnnotationError("record text must be a string")
    spans = parse_spans(record.get("spans"))

    entities: list[dict[str, Any]] = []
    for index, span in enumerate(spans):
        start, end, label = span.get("start"), span.get("end"), span.get("label")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
        ):
            raise AnnotationError(f"span {index} start and end must be integers")
        if not isinstance(label, str) or not label:
            raise AnnotationError(f"span {index} label must be a non-empty string")
        if start < 0 or end <= start or end > len(text):
            raise AnnotationError(f"span {index} offsets are outside the text")
        entities.append({"type": ontology.get(label, "UNMAPPED"), "start": start, "end": end})

    entities.sort(key=lambda entity: (entity["start"], entity["end"], entity["type"]))
    return {"text": text, "entities": entities}


def inspect_split(split: Iterable[Mapping[str, Any]], column_names: list[str]) -> dict[str, Any]:
    """Return structural inspection information without reading text values into output."""
    labels: Counter[str] = Counter()
    annotation_shape: dict[str, str] | None = None
    row_count = 0
    for record in split:
        row_count += 1
        spans = parse_spans(record.get("spans", []))
        for span in spans:
            label = span.get("label")
            if isinstance(label, str):
                labels[label] += 1
            if annotation_shape is None:
                annotation_shape = {
                    key: type(value).__name__ for key, value in sorted(span.items())
                }
    return {
        "rows": row_count,
        "columns": column_names,
        "entity_label_counts": dict(sorted(labels.items())),
        "annotation_structure": annotation_shape or {},
    }


def is_within_repository(path: Path) -> bool:
    """Return whether a resolved path is in the repository tree."""
    try:
        path.resolve().relative_to(REPOSITORY_ROOT.resolve())
    except ValueError:
        return False
    return True
