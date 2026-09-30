"""Thin, privacy-safe adapter from GLiNER output to the strict exact-span harness.

GLiNER output -> validated ``RawPrediction`` (label, offsets, score; the entity
text is dropped immediately) -> gateway ``ExactMatch`` via the reviewed label map
and the same ``evaluation_type`` mapping applied to the ground truth. Metric logic
is not reimplemented here; scoring reuses ``scripts.presidio_evaluation`` and
``scripts.batch_evaluation``. This module imports neither torch nor gliner, so
it is unit-testable with mocked model output.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeGuard

import yaml

from scripts.nemotron_pii import REPOSITORY_ROOT, load_ontology
from scripts.presidio_evaluation import ExactMatch, evaluation_type

UNMAPPED = "UNMAPPED"
DEFAULT_LABEL_MAP = REPOSITORY_ROOT / "configs" / "gliner_label_map.yaml"

# Words (as split by the model's own splitter) are the unit of GLiNER's input
# limit (``max_len``) and of its longest predictable span (``max_width``).
WordSpan = tuple[int, int]
WordSplitter = Callable[[str], Iterable[tuple[str, int, int]]]
# gliner's WhitespaceTokenSplitter (``words_splitter_type: whitespace``), restated so
# the scorer never imports gliner; a test pins it to the library's splitter.
WHITESPACE_WORD_PATTERN = re.compile(r"\w+(?:[-_]\w+)*|\S")


def whitespace_words(text: str) -> list[WordSpan]:
    return [match.span() for match in WHITESPACE_WORD_PATTERN.finditer(text)]


class GlinerOutputError(ValueError):
    """Malformed model output. Messages are fixed and never contain entity text."""


@dataclass(frozen=True)
class RawPrediction:
    """One native GLiNER prediction with absolute half-open character offsets."""

    label: str
    start: int
    end: int
    score: float


@dataclass(frozen=True)
class GlinerLabelMap:
    """Reviewed prompt vocabulary and native-label-to-gateway-type mapping."""

    model_id: str
    model_revision: str
    labels: tuple[str, ...]
    mapping: Mapping[str, str]
    deviations: Mapping[str, str]

    def gateway_type(self, label: str) -> str:
        return self.mapping[label]

    def as_identity(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "labels": list(self.labels),
            "mapping": dict(sorted(self.mapping.items())),
            "deviations": dict(sorted(self.deviations.items())),
        }


def load_label_map(
    path: Path = DEFAULT_LABEL_MAP, ontology: Mapping[str, str] | None = None
) -> GlinerLabelMap:
    """Load the label map, rejecting undocumented divergence from the ontology."""
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or not isinstance(loaded.get("labels"), dict):
        raise ValueError("GLiNER label map must contain a labels mapping")
    labels, deviations = loaded["labels"], loaded.get("deviations") or {}
    model_id, revision = loaded.get("model_id"), loaded.get("model_revision")
    if not isinstance(model_id, str) or not isinstance(revision, str):
        raise ValueError("GLiNER label map must pin model_id and model_revision")
    if not labels or not all(
        isinstance(label, str) and label and isinstance(target, str) and target
        for label, target in labels.items()
    ):
        raise ValueError("GLiNER label map entries must be non-empty strings")
    if not isinstance(deviations, dict) or not all(
        isinstance(reason, str) and reason.strip() for reason in deviations.values()
    ):
        raise ValueError("every GLiNER label map deviation needs a written reason")
    ontology = load_ontology() if ontology is None else ontology
    allowed_targets = set(ontology.values()) | {UNMAPPED}
    for label, target in labels.items():
        if target not in allowed_targets:
            raise ValueError(f"GLiNER label {label!r} maps to unknown gateway type {target!r}")
        expected = ontology.get(label)
        if expected is not None and expected != target and label not in deviations:
            raise ValueError(
                f"GLiNER label {label!r} maps to {target!r} but the ground-truth ontology "
                f"uses {expected!r}; document it under deviations or align it"
            )
    unknown_deviations = set(deviations) - set(labels)
    if unknown_deviations:
        raise ValueError("deviations reference labels that are not in the label map")
    return GlinerLabelMap(
        model_id=model_id,
        model_revision=revision,
        labels=tuple(labels),
        mapping=dict(labels),
        deviations=dict(deviations),
    )


# --- Raw output validation ---------------------------------------------------------------


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def parse_model_output(
    entities: object, text_length: int, allowed_labels: Iterable[str]
) -> list[RawPrediction]:
    """Validate one text's GLiNER output; keep only label, offsets, and score.

    The ``text`` field GLiNER returns is the detected value itself; it is never
    read. Any structural problem raises instead of silently dropping a prediction.
    """
    if not isinstance(entities, list):
        raise GlinerOutputError("model output for a text must be a list")
    allowed = set(allowed_labels)
    predictions: list[RawPrediction] = []
    for entity in entities:
        if not isinstance(entity, Mapping):
            raise GlinerOutputError("each model prediction must be an object")
        label, start, end = entity.get("label"), entity.get("start"), entity.get("end")
        score = entity.get("score")
        if not isinstance(label, str) or label not in allowed:
            raise GlinerOutputError("model returned a label outside the prompt vocabulary")
        if not _is_int(start) or not _is_int(end):
            raise GlinerOutputError("model prediction offsets must be integers")
        if start < 0 or end <= start or end > text_length:
            raise GlinerOutputError("model prediction offsets are outside the text")
        if isinstance(score, bool) or not isinstance(score, int | float):
            raise GlinerOutputError("model prediction score must be a number")
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise GlinerOutputError("model prediction score must be within [0, 1]")
        predictions.append(RawPrediction(label, start, end, float(score)))
    return predictions


def canonical_order(predictions: Iterable[RawPrediction]) -> list[RawPrediction]:
    """Deterministic order independent of model/batch emission order."""
    return sorted(predictions, key=lambda p: (p.start, p.end, p.label, -p.score))


def deduplicate(predictions: Iterable[RawPrediction]) -> tuple[list[RawPrediction], int]:
    """Collapse identical (label, start, end) predictions, keeping the best score.

    Flat-NER decoding never emits duplicates, so the returned count is expected to
    be zero; it is reported rather than hidden.
    """
    best: dict[tuple[str, int, int], RawPrediction] = {}
    total = 0
    for prediction in predictions:
        total += 1
        key = (prediction.label, prediction.start, prediction.end)
        current = best.get(key)
        if current is None or prediction.score > current.score:
            best[key] = prediction
    return canonical_order(best.values()), total - len(best)


def apply_threshold(predictions: Iterable[RawPrediction], threshold: float) -> list[RawPrediction]:
    """Keep ``score > threshold``, the same strict comparison GLiNER's decoder uses."""
    return [prediction for prediction in predictions if prediction.score > threshold]


def overlapping_pairs(predictions: Sequence[RawPrediction]) -> int:
    """Count pairs of predictions whose half-open spans overlap (nested included)."""
    ordered = sorted(predictions, key=lambda p: (p.start, p.end))
    pairs = 0
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if right.start >= left.end:
                break
            pairs += 1
    return pairs


@dataclass(frozen=True)
class GatewayPredictions:
    """Predictions after ontology mapping, with what the filter removed."""

    kept: list[ExactMatch]
    removed_by_label: Counter[str]


def to_gateway(
    predictions: Iterable[RawPrediction], label_map: GlinerLabelMap
) -> GatewayPredictions:
    """Map native labels to gateway types; UNMAPPED predictions are counted, not scored."""
    kept: list[ExactMatch] = []
    removed: Counter[str] = Counter()
    for prediction in predictions:
        gateway = label_map.gateway_type(prediction.label)
        if gateway == UNMAPPED:
            removed[prediction.label] += 1
            continue
        kept.append(ExactMatch(evaluation_type(gateway), prediction.start, prediction.end))
    kept.sort(key=lambda item: (item.start, item.end, item.entity_type))
    return GatewayPredictions(kept, removed)


# --- Long-text chunking -------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """A contiguous word window and the region whose predictions it owns.

    ``char_start``/``char_end`` delimit the text passed to the model. A prediction
    from this chunk is kept only if its absolute start lies in
    ``[own_start, own_end)``; owned regions partition the text.
    """

    char_start: int
    char_end: int
    own_start: int
    own_end: int
    word_count: int


def plan_chunks(
    words: Sequence[WordSpan], text_length: int, max_words: int, margin: int
) -> list[Chunk]:
    """Split a text into overlapping windows of at most ``max_words`` words.

    A text within the model's limit is one chunk: exactly the unchunked library
    path. Longer texts get windows overlapping by ``2 * margin`` words; each owns
    its centre. With ``margin >= max_width`` any predictable entity starting in an
    owned region lies wholly inside that window, so no entity is cut by chunking.
    """
    if max_words < 1 or margin < 0 or 2 * margin >= max_words:
        raise ValueError("chunking requires max_words > 2 * margin >= 0")
    count = len(words)
    if count <= max_words:
        return [Chunk(0, text_length, 0, text_length, count)]
    stride = max_words - 2 * margin
    starts = list(range(0, count - max_words, stride)) + [count - max_words]
    chunks: list[Chunk] = []
    for index, first in enumerate(starts):
        last = first + max_words  # exclusive word index
        own_first = 0 if index == 0 else first + margin
        if index + 1 < len(starts):
            # Hand over to the next window where its own region begins.
            own_last = max(own_first, starts[index + 1] + margin)
        else:
            own_last = count
        own_start = 0 if own_first == 0 else words[own_first][0]
        own_end = text_length if own_last >= count else words[own_last][0]
        chunks.append(
            Chunk(
                char_start=words[first][0],
                char_end=words[last - 1][1],
                own_start=own_start,
                own_end=own_end,
                word_count=last - first,
            )
        )
    return chunks


def merge_chunk_predictions(
    chunks: Sequence[Chunk], chunk_predictions: Sequence[Sequence[RawPrediction]]
) -> list[RawPrediction]:
    """Shift chunk-relative offsets to the full text and keep each chunk's owned region."""
    if len(chunks) != len(chunk_predictions):
        raise GlinerOutputError("chunk prediction count does not match the chunk plan")
    merged: list[RawPrediction] = []
    for chunk, predictions in zip(chunks, chunk_predictions, strict=True):
        for prediction in predictions:
            start, end = prediction.start + chunk.char_start, prediction.end + chunk.char_start
            if end > chunk.char_end:
                raise GlinerOutputError("chunk prediction extends beyond its chunk")
            if chunk.own_start <= start < chunk.own_end:
                merged.append(RawPrediction(prediction.label, start, end, prediction.score))
    return merged


# --- Batched prediction -------------------------------------------------------------------


class GlinerModel(Protocol):
    """The subset of ``gliner.GLiNER`` used here (``inference`` backs ``predict_entities``)."""

    def inference(
        self,
        texts: list[str],
        labels: list[str],
        flat_ner: bool = ...,
        threshold: float = ...,
        multi_label: bool = ...,
        batch_size: int = ...,
    ) -> list[list[dict[str, Any]]]: ...


@dataclass(frozen=True)
class InferenceConfig:
    threshold: float
    batch_size: int
    max_words: int
    margin: int
    flat_ner: bool = True
    multi_label: bool = False

    def as_identity(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "max_words": self.max_words,
            "chunk_margin_words": self.margin,
            "flat_ner": self.flat_ner,
            "multi_label": self.multi_label,
        }


@dataclass(frozen=True)
class TextPrediction:
    """Native predictions for one text plus privacy-safe structural counters."""

    predictions: list[RawPrediction]
    chunks: int
    duplicates_removed: int


class GlinerPredictor:
    """Chunk, batch, and validate GLiNER inference for a list of texts."""

    def __init__(
        self,
        model: GlinerModel,
        label_map: GlinerLabelMap,
        splitter: WordSplitter,
        config: InferenceConfig,
    ) -> None:
        self._model = model
        self._labels = list(label_map.labels)
        self._splitter = splitter
        self._config = config

    def words(self, text: str) -> list[WordSpan]:
        return [(start, end) for _, start, end in self._splitter(text)]

    def predict(self, texts: Sequence[str]) -> list[TextPrediction]:
        plans = [
            plan_chunks(self.words(text), len(text), self._config.max_words, self._config.margin)
            for text in texts
        ]
        requests = [
            (text_index, chunk_index, texts[text_index][chunk.char_start : chunk.char_end])
            for text_index, chunks in enumerate(plans)
            for chunk_index, chunk in enumerate(chunks)
        ]
        # Length-sorted batches minimise padding; results are restored to input order.
        order = sorted(range(len(requests)), key=lambda i: (-len(requests[i][2]), i))
        outputs = (
            self._model.inference(
                [requests[i][2] for i in order],
                self._labels,
                flat_ner=self._config.flat_ner,
                threshold=self._config.threshold,
                multi_label=self._config.multi_label,
                batch_size=self._config.batch_size,
            )
            if requests
            else []
        )
        if not isinstance(outputs, list) or len(outputs) != len(requests):
            raise GlinerOutputError("model returned a different number of outputs than inputs")
        per_chunk: dict[tuple[int, int], list[RawPrediction]] = {}
        for position, output in zip(order, outputs, strict=True):
            text_index, chunk_index, chunk_text = requests[position]
            per_chunk[(text_index, chunk_index)] = parse_model_output(
                output, len(chunk_text), self._labels
            )
        results: list[TextPrediction] = []
        for text_index, chunks in enumerate(plans):
            merged = merge_chunk_predictions(
                chunks, [per_chunk[(text_index, index)] for index in range(len(chunks))]
            )
            unique, duplicates = deduplicate(merged)
            results.append(TextPrediction(unique, len(chunks), duplicates))
        return results


# --- Structural reachability ----------------------------------------------------------------


def unreachable_reasons(
    words: Sequence[WordSpan], spans: Iterable[tuple[int, int]], max_width: int
) -> list[str | None]:
    """Why each exact span cannot be produced by a word-span model, if it cannot.

    GLiNER predicts spans of whole words (per its splitter) at most ``max_width``
    words long; ground truth that breaks either rule is structurally unreachable.
    """
    starts = {span[0]: index for index, span in enumerate(words)}
    ends = {span[1]: index for index, span in enumerate(words)}
    reasons: list[str | None] = []
    for start, end in spans:
        first, last = starts.get(start), ends.get(end)
        if first is None or last is None or last < first:
            reasons.append("misaligned_word_boundary")
        elif last - first + 1 > max_width:
            reasons.append("wider_than_max_width")
        else:
            reasons.append(None)
    return reasons
