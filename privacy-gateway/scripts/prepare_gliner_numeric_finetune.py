"""Build the offline GLiNER numeric fine-tuning dataset (no model, no inference).

Reads only non-benchmark training splits, keeps their character offsets, converts
them to gliner 0.2.29's span training record, adds deterministic synthetic
``other_numeric`` hard negatives, and writes train/dev JSONL plus an aggregate-only
manifest outside the repository.

gliner 0.2.29 training record (``UniEncoderSpanProcessor``)::

    {"tokenized_text": [word, ...],
     "ner": [[first_word, last_word_inclusive, label], ...],
     "ner_labels": [prompt label, ...]}

``ner_labels`` fixes the prompted label set for the example, so the processor does
not sample label types at random. Extra keys (``source``, ``kind``, ``group``) are
ignored by the processor and dropped by the trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluation_checkpoint import canonical_sha256, sha256_file
from scripts.gliner_evaluation import (
    WHITESPACE_WORD_PATTERN,
    GlinerLabelMap,
    unreachable_reasons,
    whitespace_words,
)
from scripts.gliner_numeric_common import (
    DEFAULT_CONFIG,
    EXPERIMENT_ID,
    FORBIDDEN_TRAINING_SPLITS,
    OOD_BENCHMARK_SHA256,
    OTHER_NUMERIC,
    REQUIRED_GLINER_VERSION,
    SOURCE_DATASETS,
    NumericConfig,
    NumericExperimentError,
    check_output_path,
    is_forbidden_split,
    lf_sha256,
    load_config,
)
from scripts.gliner_predictions import DEV_SELECTION_SEED, select_dev_indices
from scripts.nemotron_pii import (
    REPOSITORY_ROOT,
    AnnotationError,
    load_ontology,
    normalize_example,
    parse_spans,
)
from scripts.ood_pii import (
    AI4PRIVACY_LABELS,
    DATASETS,
    GRETEL_LABELS,
    OodNormalizationError,
    normalize,
)
from scripts.prepare_ood_pii import audit_error

DEFAULT_OUTPUT_DIR = Path.home() / ".cache" / "privacy-gateway" / "gliner-numeric-finetune" / "data"
THRESHOLD_SELECTION_SIZE = 2_000
SHUFFLE_BUCKETS = 64
FILES = ("train.jsonl", "dev.jsonl", "dev_normalized.jsonl")
OOD_SOURCE_LABELS = {"ai4privacy": AI4PRIVACY_LABELS, "gretel": GRETEL_LABELS}

Entity = tuple[str, int, int]
SourceRows = Callable[[], Iterable[Mapping[str, Any]]]


class Rejection(Exception):
    """An example cannot be represented exactly; the reason is a fixed identifier."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# --- Span alignment ---------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    """One training example: model words plus word-index entities and the char view."""

    tokens: tuple[str, ...]
    ner: tuple[tuple[int, int, str], ...]
    text: str
    char_entities: tuple[Entity, ...]


def plan_windows(
    word_count: int, word_entities: Sequence[tuple[int, int, str]], max_words: int
) -> list[tuple[int, int]]:
    """Consecutive word windows of at most ``max_words`` that never cut an entity."""
    if word_count == 0:
        return [(0, 0)]
    windows: list[tuple[int, int]] = []
    start = 0
    while start < word_count:
        end = min(start + max_words, word_count)
        if end < word_count:
            crossing = [first for first, last, _ in word_entities if first < end <= last]
            if crossing:
                end = min(crossing)
        if end <= start:
            raise Rejection("entity_wider_than_window")
        windows.append((start, end))
        start = end
    return windows


def align_example(
    text: str, entities: Sequence[Entity], max_words: int, max_width: int
) -> list[Window]:
    """Map char-offset entities onto whole model words without moving any boundary."""
    if not isinstance(text, str):
        raise Rejection("text_missing")
    unique = sorted(set(entities), key=lambda item: (item[1], item[2], item[0]))
    by_span: dict[tuple[int, int], str] = {}
    for label, start, end in unique:
        if by_span.setdefault((start, end), label) != label:
            raise Rejection("conflicting_labels_same_span")
    words = whitespace_words(text)
    reasons = unreachable_reasons(words, [(start, end) for _, start, end in unique], max_width)
    for reason in reasons:
        if reason is not None:
            raise Rejection(reason)
    first_word = {start: index for index, (start, _) in enumerate(words)}
    last_word = {end: index for index, (_, end) in enumerate(words)}
    word_entities = [(first_word[start], last_word[end], label) for label, start, end in unique]
    windows = []
    for begin, finish in plan_windows(len(words), word_entities, max_words):
        if begin == finish:
            windows.append(Window((), (), "", ()))
            continue
        offset, stop = words[begin][0], words[finish - 1][1]
        inside = [item for item in word_entities if begin <= item[0] and item[1] < finish]
        ner = tuple((first - begin, last - begin, label) for first, last, label in inside)
        char_entities = tuple(
            (label, words[first][0] - offset, words[last][1] - offset)
            for first, last, label in inside
        )
        windows.append(
            Window(
                tokens=tuple(text[start:end] for start, end in words[begin:finish]),
                ner=ner,
                text=text[offset:stop],
                char_entities=char_entities,
            )
        )
    return windows


# --- Source adapters --------------------------------------------------------------------


@dataclass(frozen=True)
class SourceExample:
    source: str
    row_index: int
    stable_id: str | None
    text: str
    entities: tuple[Entity, ...]
    prompt_labels: tuple[str, ...]


def nemotron_example(
    record: Mapping[str, Any],
    row_index: int,
    ontology: Mapping[str, str],
    native_labels: Sequence[str],
) -> SourceExample:
    """Reuse the Nemotron normalizer's checks; keep the native label for each span."""
    normalize_example(record, ontology)
    allowed = set(native_labels)
    entities = []
    for span in parse_spans(record.get("spans")):
        label = span["label"]
        if label not in allowed:
            raise AnnotationError("span label is outside the GLiNER vocabulary")
        entities.append((str(label), int(span["start"]), int(span["end"])))
    uid = record.get("uid")
    if not isinstance(uid, str) or not uid:
        raise AnnotationError("record uid must be a non-empty string")
    return SourceExample(
        source="nemotron_train",
        row_index=row_index,
        stable_id=f"nemotron:uid:{uid}",
        text=str(record["text"]),
        entities=tuple(entities),
        prompt_labels=tuple(native_labels),
    )


def ood_prompt_labels(dataset: str, native_labels: Sequence[str]) -> tuple[str, ...]:
    """Native labels this source can express; others would be false negatives."""
    expressible = set(OOD_SOURCE_LABELS[dataset].values())
    return tuple(label for label in native_labels if label in expressible)


def ood_example(
    dataset: str,
    split: str,
    record: Mapping[str, Any],
    row_index: int,
    label_map: GlinerLabelMap,
) -> SourceExample:
    """Reuse the pinned OOD adapter; drop whole examples that contain UNMAPPED spans."""
    if is_forbidden_split(dataset, split):
        raise NumericExperimentError("refusing to read a frozen benchmark split")
    try:
        normalized = normalize(dataset, record, row_index, split, label_map)
    except OodNormalizationError as error:
        raise Rejection("adapter: " + audit_error(error)) from None
    native = normalized["native_entities"]
    if any(item["native_label"] == "UNMAPPED" for item in native):
        raise Rejection("unmapped_source_label")
    raw_id = next((record[key] for key in ("uid", "id") if record.get(key) not in (None, "")), None)
    return SourceExample(
        source=f"{dataset}_train",
        row_index=row_index,
        stable_id=f"{dataset}:id:{raw_id}" if raw_id is not None else None,
        text=str(normalized["text"]),
        entities=tuple(
            (str(item["native_label"]), int(item["start"]), int(item["end"])) for item in native
        ),
        prompt_labels=ood_prompt_labels(dataset, label_map.labels),
    )


def text_key(text: str) -> str:
    """Near-duplicate key: casefolded, whitespace-collapsed text hash."""
    return hashlib.sha256(" ".join(text.casefold().split()).encode("utf-8")).hexdigest()


def load_benchmark_text_keys(paths: Sequence[Path]) -> tuple[set[str], list[str]]:
    """Text keys of frozen OOD benchmark rows. Only the pinned files are accepted."""
    keys: set[str] = set()
    digests = []
    allowed = set(OOD_BENCHMARK_SHA256.values())
    for path in paths:
        digest = sha256_file(path)
        if digest not in allowed:
            raise NumericExperimentError("benchmark file is not a pinned OOD benchmark")
        digests.append(digest)
        with path.open("r", encoding="utf-8", newline="\n") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                        raise NumericExperimentError("benchmark record has no text")
                    keys.add(text_key(row["text"]))
    return keys, sorted(digests)


# --- Deterministic grouping and split ---------------------------------------------------


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, key: str) -> str:
        self.parent.setdefault(key, key)
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != root:
            self.parent[key], key = root, self.parent[key]
        return root

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def _unit(seed: int, purpose: str, key: str) -> float:
    digest = hashlib.sha256(f"{seed}|{purpose}|{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_splits(
    example_keys: Sequence[Sequence[str]],
    reserved: Sequence[bool],
    seed: int,
    dev_fraction: float,
) -> tuple[list[str], list[str]]:
    """Return (split, group) per example. Examples sharing any key share a split.

    Keys are stable source ids and near-duplicate text keys. The group name is the
    smallest key in its component, so the result does not depend on input order.
    """
    forest = _UnionFind()
    for keys in example_keys:
        if not keys:
            raise NumericExperimentError("every example needs at least one grouping key")
        for key in keys:
            forest.union(keys[0], key)
    groups = [forest.find(keys[0]) for keys in example_keys]
    reserved_groups = {group for group, flag in zip(groups, reserved, strict=True) if flag}
    splits = []
    for group in groups:
        if group in reserved_groups:
            splits.append("reserved")
        elif _unit(seed, "dev", group) < dev_fraction:
            splits.append("dev")
        else:
            splits.append("train")
    return splits, groups


# --- Synthetic other_numeric hard negatives ---------------------------------------------


IP_LOOKALIKE_CONTEXTS = ("firmware version", "build number", "software release", "package version")
DATE_LOOKALIKE_CONTEXTS = ("batch number", "lot number", "production run")
POSTCODE_LOOKALIKE_CONTEXTS = ("error code", "part number", "item code", "model number")
PREFIXES = ("INV", "ORD", "TXN", "REF", "TKT", "TRK", "CNF", "CASE", "DOC", "REC", "PO", "SO")
LEADS = (
    "Please quote the",
    "We have updated the",
    "For reference, the",
    "The system generated the",
    "Our records show the",
    "Kindly confirm the",
    "Attached you will find the",
)
TAILS = (
    "when you contact support.",
    "for this request.",
    "in the next message.",
    "on the attached form.",
    "before the end of the week.",
    "in all future replies.",
)
DELIMITERS = (": ", " ", " is ", " #", ":  ", ":\t", " - ", " no. ", "\n")
MID_TRAILS = ("", ",", " -", " ,")
END_TRAILS = (".", "!", " .", ";", "")
WRAPS = (("", ""), ("(", ")"), ('"', '"'), ("[", "]"), ("'", "'"))
QUANTITY_TEMPLATES = (
    "We packed {a} boxes and {b} crates for the shipment.",
    "The report lists {a} open items and {b} closed items.",
    "Only {a} of the {b} printers were working this morning.",
    "The queue held {a} requests, down from {b} last week.",
)


def luhn_check_digit(payload: str) -> str:
    total = 0
    for index, char in enumerate(reversed(payload)):
        value = int(char)
        if index % 2 == 0:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return str((10 - total % 10) % 10)


def luhn_valid(digits: str) -> bool:
    return luhn_check_digit(digits[:-1]) == digits[-1]


def _digits(rng: random.Random, length: int, leading_zero: bool = False) -> str:
    body = "".join(str(rng.randrange(10)) for _ in range(length))
    if leading_zero:
        zeros = rng.randint(1, max(1, min(4, length - 1)))
        return "0" * zeros + body[zeros:]
    return str(rng.randint(1, 9)) + body[1:] if length > 1 else body


def _group(digits: str, sizes: Sequence[int], separator: str) -> str:
    parts, index = [], 0
    for size in sizes:
        if index >= len(digits):
            break
        parts.append(digits[index : index + size])
        index += size
    if index < len(digits):
        parts.append(digits[index:])
    return separator.join(parts)


def reference_value(rng: random.Random) -> tuple[str, str]:
    style = rng.choice(("plain", "leading_zero", "long", "grouped", "prefixed", "alnum"))
    if style == "plain":
        return style, _digits(rng, rng.randint(4, 12))
    if style == "leading_zero":
        return style, _digits(rng, rng.randint(5, 12), leading_zero=True)
    if style == "long":
        return style, _digits(rng, rng.randint(20, 30))
    if style == "grouped":
        size = rng.choice((3, 4))
        digits = _digits(rng, rng.randint(8, 16))
        return style, _group(digits, [size] * 8, rng.choice((" ", "-")))
    if style == "prefixed":
        separator = rng.choice(("-", "", "/", "_"))
        return style, f"{rng.choice(PREFIXES)}{separator}{_digits(rng, rng.randint(4, 10), True)}"
    letters = "".join(rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(rng.randint(1, 3)))
    middle = "".join(rng.choice("0123456789ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(6))
    return style, f"{rng.randint(1, 9)}{letters}{middle}{_digits(rng, rng.randint(6, 10))}"


def card_shaped_value(rng: random.Random, luhn_ok: bool) -> str:
    length = rng.choice((13, 15, 16, 16, 16, 19))
    payload = str(rng.randint(2, 6)) + "".join(str(rng.randrange(10)) for _ in range(length - 2))
    check = luhn_check_digit(payload)
    if not luhn_ok:
        check = str((int(check) + rng.randint(1, 9)) % 10)
    digits = payload + check
    sizes = {15: (4, 6, 5)}.get(length, (4, 4, 4, 4, 4))
    return _group(digits, sizes, rng.choice((" ", "-", "")))


def ip_shaped_value(rng: random.Random) -> str:
    parts = rng.choice((3, 4, 4, 4, 5))
    return ".".join(str(rng.randint(0, 255)) for _ in range(parts))


def date_shaped_value(rng: random.Random) -> str:
    year, month, day = rng.randint(2015, 2030), rng.randint(1, 12), rng.randint(1, 28)
    serial = _digits(rng, rng.randint(2, 4), leading_zero=rng.random() < 0.5)
    return rng.choice(
        (
            f"{year}{month:02d}{day:02d}-{serial}",
            f"{year}-{month:02d}-{day:02d}-{serial}",
            f"{year % 100:02d}{month:02d}{day:02d}/{serial}",
        )
    )


def postcode_shaped_value(rng: random.Random) -> str:
    return rng.choice(
        (
            _digits(rng, 5),
            f"{_digits(rng, 5)}-{_digits(rng, 4, leading_zero=rng.random() < 0.3)}",
            _digits(rng, 4, leading_zero=True),
        )
    )


def _cased(rng: random.Random, phrase: str) -> str:
    return rng.choice((phrase, phrase[:1].upper() + phrase[1:], phrase.upper()))


@dataclass
class _Builder:
    """Accumulates text while recording exact char offsets of inserted values."""

    parts: list[str] = field(default_factory=list)
    entities: list[Entity] = field(default_factory=list)
    length: int = 0

    def add(self, piece: str) -> None:
        self.parts.append(piece)
        self.length += len(piece)

    def value(self, value: str, label: str) -> None:
        self.entities.append((label, self.length, self.length + len(value)))
        self.add(value)

    def text(self) -> str:
        return "".join(self.parts)


def _clause(rng: random.Random, builder: _Builder, context: str, value: str, label: str) -> None:
    opening, closing = rng.choice(WRAPS)
    builder.add(_cased(rng, context) + rng.choice(DELIMITERS) + opening)
    builder.value(value, label)
    builder.add(closing)


def _single(rng: random.Random, context: str, value: str, label: str) -> tuple[str, list[Entity]]:
    builder = _Builder()
    position = rng.choice(("start", "middle", "end"))
    if position != "start":
        builder.add(rng.choice(LEADS) + rng.choice((" ", "  ", "\t")))
    _clause(rng, builder, context, value, label)
    if position == "end":
        builder.add(rng.choice(END_TRAILS))
    else:
        builder.add(rng.choice(MID_TRAILS) + " " + rng.choice(TAILS))
    return builder.text(), builder.entities


@dataclass(frozen=True)
class SyntheticExample:
    family: str
    text: str
    entities: tuple[Entity, ...]
    card_shaped_luhn: bool | None = None


def _lookalike(rng: random.Random, contexts: Sequence[str]) -> tuple[str, str, str, bool | None]:
    roll = rng.random()
    if roll < 0.55:
        style, value = reference_value(rng)
        return f"reference_{style}", rng.choice(contexts), value, None
    if roll < 0.70:
        return "ip_lookalike", rng.choice(IP_LOOKALIKE_CONTEXTS), ip_shaped_value(rng), None
    if roll < 0.85:
        return "date_lookalike", rng.choice(DATE_LOOKALIKE_CONTEXTS), date_shaped_value(rng), None
    return (
        "postcode_lookalike",
        rng.choice(POSTCODE_LOOKALIKE_CONTEXTS),
        (postcode_shaped_value(rng)),
        None,
    )


def generate_hard_negatives(
    count: int, seed: int, stream: str, settings: Mapping[str, Any]
) -> list[SyntheticExample]:
    """Deterministic contextual numeric examples. Labels never depend on Luhn."""
    contexts = tuple(settings["other_numeric_contexts"])
    native_contexts = dict(settings["native_contexts"])
    cuts = []
    total = 0.0
    for key in (
        "unlabeled_quantity_fraction",
        "card_shaped_fraction",
        "multi_span_fraction",
        "native_context_fraction",
    ):
        total += float(settings[key])
        cuts.append(total)
    examples: list[SyntheticExample] = []
    index = 0
    while len(examples) < count:
        rng = random.Random(f"{EXPERIMENT_ID}|{seed}|{stream}|{index}")
        index += 1
        roll = rng.random()
        if roll < cuts[0]:
            template = rng.choice(QUANTITY_TEMPLATES)
            text = template.format(a=rng.randint(2, 40), b=rng.randint(41, 90))
            examples.append(SyntheticExample("unlabeled_quantity", text, ()))
        elif roll < cuts[1]:
            luhn_ok = rng.random() < 0.5
            value = card_shaped_value(rng, luhn_ok)
            text, entities = _single(rng, rng.choice(contexts), value, OTHER_NUMERIC)
            examples.append(SyntheticExample("card_shaped", text, tuple(entities), luhn_ok))
        elif roll < cuts[2]:
            builder = _Builder()
            if rng.random() < 0.5:
                builder.add(rng.choice(LEADS) + " ")
            for clause in range(rng.randint(2, 3)):
                if clause:
                    builder.add(rng.choice((" and ", "; ", ", ", " plus ", " / ")))
                _, context, value, _ = _lookalike(rng, contexts)
                _clause(rng, builder, context, value, OTHER_NUMERIC)
            builder.add(rng.choice(END_TRAILS))
            examples.append(SyntheticExample("multi_span", builder.text(), tuple(builder.entities)))
        elif roll < cuts[3] and native_contexts and count - len(examples) >= 2:
            # The same value under a reference context and an identifier context.
            if rng.random() < 0.5:
                value = card_shaped_value(rng, rng.random() < 0.5)
            else:
                value = reference_value(rng)[1]
            phrase = rng.choice(sorted(native_contexts))
            text, entities = _single(rng, rng.choice(contexts), value, OTHER_NUMERIC)
            examples.append(SyntheticExample("contrast_reference", text, tuple(entities)))
            text, entities = _single(rng, phrase, value, native_contexts[phrase])
            examples.append(SyntheticExample("contrast_native", text, tuple(entities)))
        else:
            family, context, value, _ = _lookalike(rng, contexts)
            text, entities = _single(rng, context, value, OTHER_NUMERIC)
            examples.append(SyntheticExample(family, text, tuple(entities)))
    return examples[:count]


# --- Records ----------------------------------------------------------------------------


def _shuffled_labels(labels: Sequence[str], seed: int, key: str) -> list[str]:
    ordered = list(labels)
    random.Random(f"{EXPERIMENT_ID}|{seed}|labels|{key}").shuffle(ordered)
    return ordered


def gliner_record(
    window: Window,
    prompt_labels: Sequence[str],
    seed: int,
    key: str,
    source: str,
    kind: str,
    group: str,
) -> dict[str, Any]:
    return {
        "tokenized_text": list(window.tokens),
        "ner": [[first, last, label] for first, last, label in window.ner],
        "ner_labels": _shuffled_labels(prompt_labels, seed, key),
        "source": source,
        "kind": kind,
        "group": hashlib.sha256(group.encode("utf-8")).hexdigest()[:16],
    }


def normalized_record(window: Window, label_map: GlinerLabelMap) -> dict[str, Any]:
    """The evaluation view of a dev window, in the OOD normalized JSONL shape."""

    def gateway(label: str) -> str:
        return "UNMAPPED" if label == OTHER_NUMERIC else label_map.gateway_type(label)

    return {
        "text": window.text,
        "entities": [
            {"type": gateway(label), "start": start, "end": end}
            for label, start, end in window.char_entities
        ],
        "native_entities": [
            {"native_label": label, "start": start, "end": end}
            for label, start, end in window.char_entities
        ],
    }


class _ShuffleWriter:
    """External bucket shuffle: lines are globally ordered by a keyed hash."""

    def __init__(self, directory: Path, name: str) -> None:
        self.directory = directory / f".{name}.buckets"
        self.directory.mkdir(parents=True)
        self.handles = [
            (self.directory / f"{bucket:02x}").open("w", encoding="utf-8", newline="\n")
            for bucket in range(SHUFFLE_BUCKETS)
        ]

    def write(self, order_key: str, *payloads: str) -> None:
        digest = hashlib.sha256(order_key.encode("utf-8")).hexdigest()
        bucket = int(digest[:2], 16) % SHUFFLE_BUCKETS
        self.handles[bucket].write("\t".join((digest, order_key, *payloads)) + "\n")

    def finish(self, outputs: Sequence[Path]) -> int:
        for handle in self.handles:
            handle.close()
        written = 0
        targets = [path.open("x", encoding="utf-8", newline="\n") for path in outputs]
        try:
            for bucket in range(SHUFFLE_BUCKETS):
                path = self.directory / f"{bucket:02x}"
                with path.open("r", encoding="utf-8", newline="\n") as handle:
                    rows = sorted(line.rstrip("\n").split("\t") for line in handle)
                for row in rows:
                    for target, payload in zip(targets, row[2:], strict=True):
                        target.write(payload + "\n")
                    written += 1
        finally:
            for target in targets:
                target.close()
        shutil.rmtree(self.directory)
        return written


def _dumps(value: object) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if "\t" in text or "\n" in text:
        raise NumericExperimentError("serialized record contains a raw separator")
    return text


# --- Balancing --------------------------------------------------------------------------


@dataclass(frozen=True)
class BalancePlan:
    copies: dict[str, int]
    synthetic_kept: int
    focus_mentions: int
    total_mentions: int

    @property
    def share(self) -> float:
        return self.focus_mentions / self.total_mentions if self.total_mentions else 0.0


def balance_plan(
    natural_focus: int,
    natural_total: int,
    focus_windows: Sequence[tuple[str, int, int]],
    synthetic: Sequence[tuple[int, int]],
    factor: int,
    max_share: float,
    seed: int,
) -> BalancePlan:
    """Choose extra focus-window copies and the synthetic prefix under the share cap.

    Share = focus-label and ``other_numeric`` mentions over all labeled mentions in
    train. Natural windows are never removed. Synthetic examples are trimmed from
    the end only if natural data plus synthetic alone exceeds the cap; then copies
    of focus windows (``(key, focus_mentions, total_mentions)``) are added round
    by round, in a seeded order, until the next copy would cross the cap.
    """
    focus, total = natural_focus, natural_total
    kept = list(synthetic)
    focus += sum(item[0] for item in kept)
    total += sum(item[1] for item in kept)
    while kept and total and focus / total > max_share:
        removed_focus, removed_total = kept.pop()
        focus, total = focus - removed_focus, total - removed_total
    ranked = sorted(
        focus_windows,
        key=lambda item: hashlib.sha256(f"{seed}|copy|{item[0]}".encode()).digest(),
    )
    copies: dict[str, int] = {}
    for _ in range(factor - 1):
        for key, window_focus, window_total in ranked:
            if (focus + window_focus) / (total + window_total) > max_share:
                return BalancePlan(copies, len(kept), focus, total)
            copies[key] = copies.get(key, 0) + 1
            focus, total = focus + window_focus, total + window_total
    return BalancePlan(copies, len(kept), focus, total)


# --- Build ------------------------------------------------------------------------------


@dataclass
class _Scan:
    source: str
    row_index: int
    keys: list[str]
    reserved: bool
    windows: int
    # Per window: (focus-label mentions, all mentions).
    mentions: list[tuple[int, int]]


@dataclass
class SourceSpec:
    name: str
    dataset: str
    split: str
    revision: str
    rows: SourceRows
    total_rows: int


def _example(
    spec: SourceSpec,
    record: Mapping[str, Any],
    row_index: int,
    config: NumericConfig,
    ontology: Mapping[str, str],
) -> SourceExample:
    if spec.dataset == "nemotron":
        return nemotron_example(record, row_index, ontology, config.native_labels)
    return ood_example(spec.dataset, spec.split, record, row_index, config.label_map)


def _iter_rows(spec: SourceSpec, limit: int | None) -> Iterator[tuple[int, Mapping[str, Any]]]:
    for row_index, record in enumerate(spec.rows()):
        if limit is not None and row_index >= limit:
            return
        yield row_index, record


def build_dataset(
    config: NumericConfig,
    sources: Sequence[SourceSpec],
    output_dir: Path,
    benchmark_paths: Sequence[Path] = (),
    limit_source_rows: int | None = None,
    synthetic_counts: tuple[int, int] | None = None,
    ontology: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Two passes: scan and group, then write shuffled train/dev files and a manifest."""
    check_output_path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise NumericExperimentError("dataset output directory is not empty")
    for spec in sources:
        if is_forbidden_split(spec.dataset, spec.split):
            raise NumericExperimentError("refusing to read a frozen benchmark split")
    ood_enabled = any(spec.dataset != "nemotron" for spec in sources)
    if ood_enabled and not benchmark_paths:
        raise NumericExperimentError("OOD training sources require --benchmark-normalized")
    ontology = load_ontology() if ontology is None else ontology
    data = config.data
    seed, max_words, max_width = config.seed, int(data["max_words"]), int(data["max_width"])
    focus = set(config.focus_labels)
    benchmark_keys, benchmark_digests = load_benchmark_text_keys(benchmark_paths)

    # Pass 1: validate, align, and collect grouping keys only.
    scans: list[_Scan] = []
    source_stats: dict[str, dict[str, Any]] = {}
    for spec in sources:
        stats: dict[str, Any] = {
            "dataset": spec.dataset,
            "split": spec.split,
            "revision": spec.revision,
            "rows_read": 0,
            "examples_accepted": 0,
            "windows": 0,
            "rejected": Counter(),
            "benchmark_overlap_excluded": 0,
        }
        reserved_rows: set[int] = set()
        if spec.dataset == "nemotron" and data["reserve_threshold_selection_subset"]:
            reserved_rows = set(select_dev_indices(spec.total_rows, THRESHOLD_SELECTION_SIZE))
        for row_index, record in _iter_rows(spec, limit_source_rows):
            stats["rows_read"] += 1
            try:
                example = _example(spec, record, row_index, config, ontology)
                windows = align_example(example.text, example.entities, max_words, max_width)
            except Rejection as rejection:
                stats["rejected"][rejection.reason] += 1
                continue
            except AnnotationError:
                raise NumericExperimentError(
                    f"malformed annotation in {spec.name} row {row_index}"
                ) from None
            key = text_key(example.text)
            if key in benchmark_keys:
                stats["benchmark_overlap_excluded"] += 1
                continue
            stats["examples_accepted"] += 1
            stats["windows"] += len(windows)
            scans.append(
                _Scan(
                    source=spec.name,
                    row_index=row_index,
                    keys=[f"text:{key}"] + ([example.stable_id] if example.stable_id else []),
                    reserved=row_index in reserved_rows,
                    windows=len(windows),
                    mentions=[
                        (sum(label in focus for _, _, label in w.ner), len(w.ner)) for w in windows
                    ],
                )
            )
        stats["rejected"] = dict(sorted(stats["rejected"].items()))
        source_stats[spec.name] = stats

    splits, groups = assign_splits(
        [scan.keys for scan in scans],
        [scan.reserved for scan in scans],
        seed,
        float(data["dev_fraction"]),
    )
    assignment = {
        (scan.source, scan.row_index): (split, group)
        for scan, split, group in zip(scans, splits, groups, strict=True)
    }

    hard = config.data["hard_negatives"]
    train_count, dev_count = synthetic_counts or (int(hard["train_count"]), int(hard["dev_count"]))
    synthetic_train = generate_hard_negatives(train_count, seed, "train", hard)
    train_texts = {text_key(example.text) for example in synthetic_train}
    synthetic_dev_all = generate_hard_negatives(dev_count, seed, "dev", hard)
    synthetic_dev = [ex for ex in synthetic_dev_all if text_key(ex.text) not in train_texts]

    train_scans = [scan for scan, split in zip(scans, splits, strict=True) if split == "train"]
    natural_train = sum(scan.windows for scan in train_scans)
    focus_windows = [
        (f"{scan.source}:{scan.row_index}:{window}", focus_count, total_count)
        for scan in train_scans
        for window, (focus_count, total_count) in enumerate(scan.mentions)
        if focus_count
    ]
    balancing = data["balancing"]
    focus_or_other = focus | {OTHER_NUMERIC}
    plan = balance_plan(
        natural_focus=sum(f for scan in train_scans for f, _ in scan.mentions),
        natural_total=sum(t for scan in train_scans for _, t in scan.mentions),
        focus_windows=focus_windows,
        synthetic=[
            (sum(label in focus_or_other for label, _, _ in ex.entities), len(ex.entities))
            for ex in synthetic_train
        ],
        factor=int(balancing["focus_oversample_factor"]),
        max_share=float(balancing["max_focus_share"]),
        seed=seed,
    )
    synthetic_train = synthetic_train[: plan.synthetic_kept]
    copies = plan.copies
    extra = sum(copies.values())

    # Pass 2: rebuild deterministically and write.
    output_dir.mkdir(parents=True, exist_ok=True)
    train_writer = _ShuffleWriter(output_dir, "train")
    dev_writer = _ShuffleWriter(output_dir, "dev")
    counts: dict[str, Counter[str]] = {
        name: Counter()
        for name in (
            "train",
            "dev",
            "train_natural",
            "train_oversampled",
            "train_synthetic",
            "dev_synthetic",
        )
    }
    examples_with: dict[str, Counter[str]] = {"train": Counter(), "dev": Counter()}
    kinds: Counter[str] = Counter()
    label_map = config.label_map

    def emit(split: str, record: dict[str, Any], window: Window, order_key: str) -> None:
        labels = [label for _, _, label in window.ner]
        counts[split].update(labels)
        examples_with[split].update(set(labels))
        kinds[f"{split}:{record['kind']}"] += 1
        if split == "train":
            counts[f"train_{record['kind']}"].update(labels)
            train_writer.write(order_key, _dumps(record))
        else:
            if record["kind"] == "synthetic":
                counts["dev_synthetic"].update(labels)
            dev_writer.write(
                order_key, _dumps(record), _dumps(normalized_record(window, label_map))
            )

    for spec in sources:
        for row_index, record in _iter_rows(spec, limit_source_rows):
            placed = assignment.get((spec.name, row_index))
            if placed is None or placed[0] == "reserved":
                continue
            split, group = placed
            example = _example(spec, record, row_index, config, ontology)
            windows = align_example(example.text, example.entities, max_words, max_width)
            for window_index, window in enumerate(windows):
                key = f"{spec.name}:{row_index}:{window_index}"
                emit(
                    split,
                    gliner_record(
                        window,
                        example.prompt_labels,
                        seed,
                        key,
                        spec.name,
                        "natural",
                        group,
                    ),
                    window,
                    key,
                )
                for copy in range(copies.get(key, 0) if split == "train" else 0):
                    copy_key = f"{key}:copy{copy}"
                    emit(
                        split,
                        gliner_record(
                            window,
                            example.prompt_labels,
                            seed,
                            copy_key,
                            spec.name,
                            "oversampled",
                            group,
                        ),
                        window,
                        copy_key,
                    )

    families: dict[str, Counter[str]] = {"train": Counter(), "dev": Counter()}
    card_luhn: Counter[str] = Counter()
    for split, examples in (("train", synthetic_train), ("dev", synthetic_dev)):
        for index, synthetic in enumerate(examples):
            windows = align_example(synthetic.text, synthetic.entities, max_words, max_width)
            if len(windows) != 1 or len(windows[0].ner) != len(synthetic.entities):
                raise NumericExperimentError("synthetic example did not align exactly")
            key = f"synthetic:{split}:{index}"
            emit(
                split,
                gliner_record(
                    windows[0],
                    config.training_labels,
                    seed,
                    key,
                    "synthetic",
                    "synthetic",
                    key,
                ),
                windows[0],
                key,
            )
            families[split][synthetic.family] += 1
            if synthetic.card_shaped_luhn is not None:
                card_luhn[
                    f"{split}:{'luhn_valid' if synthetic.card_shaped_luhn else 'luhn_invalid'}"
                ] += 1

    train_path, dev_path, normalized_path = (output_dir / name for name in FILES)
    train_records = train_writer.finish([train_path])
    dev_records = dev_writer.finish([dev_path, normalized_path])

    train_mentions = sum(counts["train"].values())
    focus_mentions = sum(counts["train"][label] for label in focus_or_other)
    focus_share = focus_mentions / train_mentions if train_mentions else 0.0
    if abs(focus_share - plan.share) > 1e-9:
        raise NumericExperimentError("written focus share differs from the balancing plan")
    focus_example_total = len(focus_windows) + extra + len(synthetic_train)
    warnings = []
    if focus_share < float(balancing["min_focus_share"]):
        warnings.append("focus_share_below_minimum")
    if focus_share > float(balancing["max_focus_share"]) + 1e-9:
        warnings.append("focus_share_above_maximum")
    if limit_source_rows is not None:
        warnings.append("limited_source_rows_smoke_build")
    if ood_enabled:
        warnings.append("ood_train_sources_enabled_matching_benchmarks_not_fully_ood")

    manifest: dict[str, Any] = {
        "experiment": EXPERIMENT_ID,
        "privacy": "aggregate counts and hashes only; no source text, values, or ids",
        "seed": seed,
        "config_sha256": config.sha256,
        "data_config_sha256": config.section_sha256("data"),
        "labels_config_sha256": config.section_sha256("labels"),
        "label_vocabulary": {
            "native_count": len(config.native_labels),
            "native_sha256": canonical_sha256(list(config.native_labels)),
            "training_only": [OTHER_NUMERIC],
            "training_sha256": canonical_sha256(list(config.training_labels)),
            "label_map_sha256_lf": lf_sha256(REPOSITORY_ROOT / "configs" / "gliner_label_map.yaml"),
            "ontology_sha256_lf": lf_sha256(REPOSITORY_ROOT / "configs" / "pii_ontology.yaml"),
            "other_numeric_prompted_in": "synthetic examples only",
        },
        "format": {
            "gliner_version": REQUIRED_GLINER_VERSION,
            "record": "tokenized_text + ner[[first_word, last_word_inclusive, label]] + ner_labels",
            "word_pattern": WHITESPACE_WORD_PATTERN.pattern,
            "max_words": max_words,
            "max_width": max_width,
        },
        "sources": source_stats,
        "forbidden_training_splits": {
            name: sorted(value) if value is not None else "all"
            for name, value in FORBIDDEN_TRAINING_SPLITS.items()
        },
        "benchmark_guard": {
            "applied": bool(benchmark_paths),
            "benchmark_sha256": benchmark_digests,
        },
        "split": {
            "dev_fraction": float(data["dev_fraction"]),
            "grouping": "union of stable source id and near-duplicate text key",
            "threshold_selection_subset": {
                "reserved": bool(data["reserve_threshold_selection_subset"]),
                "seed": DEV_SELECTION_SEED,
                "size": THRESHOLD_SELECTION_SIZE,
            },
            "examples": dict(Counter(splits)),
            "groups": dict(
                Counter(split for split, _ in {(s, g) for s, g in zip(splits, groups, strict=True)})
            ),
            "natural_windows": {
                split: sum(
                    scan.windows
                    for scan, chosen in zip(scans, splits, strict=True)
                    if chosen == split
                )
                for split in ("train", "dev", "reserved")
            },
        },
        "balancing": {
            "share_definition": "focus-label and other_numeric mentions / all train mentions",
            "natural_train_windows": natural_train,
            "focus_natural_train_windows": len(focus_windows),
            "oversampled_copies": extra,
            "synthetic_train_requested": train_count,
            "synthetic_train_kept": len(synthetic_train),
            "focus_share": focus_share,
            "focus_mentions": focus_mentions,
            "train_mentions": train_mentions,
            "focus_example_share": (focus_example_total / train_records if train_records else 0.0),
            "focus_oversample_factor": int(balancing["focus_oversample_factor"]),
            "share_bounds": [
                float(balancing["min_focus_share"]),
                float(balancing["max_focus_share"]),
            ],
        },
        "hard_negatives": {
            "train_families": dict(sorted(families["train"].items())),
            "dev_families": dict(sorted(families["dev"].items())),
            "dev_dropped_as_train_duplicates": len(synthetic_dev_all) - len(synthetic_dev),
            "card_shaped_luhn": dict(sorted(card_luhn.items())),
            "labeling": "context only; Luhn never decides a label",
        },
        "records": {
            "train": train_records,
            "dev": dev_records,
            "by_kind": dict(sorted(kinds.items())),
        },
        "label_counts": {name: dict(sorted(counter.items())) for name, counter in counts.items()},
        "examples_with_label": {
            name: dict(sorted(counter.items())) for name, counter in examples_with.items()
        },
        "limit_source_rows": limit_source_rows,
        "warnings": warnings,
        "files": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in (train_path, dev_path, normalized_path)
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _constant_rows(rows: Iterable[Mapping[str, Any]]) -> SourceRows:
    return lambda: rows


def huggingface_sources(config: NumericConfig) -> list[SourceSpec]:
    """Enabled sources as lazily re-iterable pinned Hugging Face splits."""
    from datasets import load_dataset

    specs = []
    for name, dataset in SOURCE_DATASETS.items():
        settings = config.data["sources"][name]
        if not settings["enabled"]:
            continue
        split = str(settings["split"])
        if is_forbidden_split(dataset, split):
            raise NumericExperimentError("refusing to read a frozen benchmark split")
        if dataset == "nemotron":
            repository, revision = str(settings["repository"]), str(settings["revision"])
        else:
            repository, revision = DATASETS[dataset].repository, DATASETS[dataset].revision
        rows = load_dataset(repository, revision=revision, split=split)
        specs.append(
            SourceSpec(
                name=name,
                dataset=dataset,
                split=split,
                revision=revision,
                rows=_constant_rows(rows),
                total_rows=len(rows),
            )
        )
    return specs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the GLiNER numeric fine-tuning dataset.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--benchmark-normalized",
        type=Path,
        action="append",
        default=[],
        help="Pinned OOD benchmark JSONL; matching texts are excluded. Repeatable.",
    )
    parser.add_argument(
        "--limit-source-rows", type=int, help="Smoke builds: first N rows per source."
    )
    parser.add_argument("--synthetic-train", type=int)
    parser.add_argument("--synthetic-dev", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        if args.limit_source_rows is not None and args.limit_source_rows < 1:
            raise NumericExperimentError("--limit-source-rows must be positive")
        hard = config.data["hard_negatives"]
        counts = (
            int(hard["train_count"]) if args.synthetic_train is None else args.synthetic_train,
            int(hard["dev_count"]) if args.synthetic_dev is None else args.synthetic_dev,
        )
        if min(counts) < 0:
            raise NumericExperimentError("synthetic counts must be non-negative")
        check_output_path(args.output_dir)
        manifest = build_dataset(
            config,
            huggingface_sources(config),
            args.output_dir,
            benchmark_paths=args.benchmark_normalized,
            limit_source_rows=args.limit_source_rows,
            synthetic_counts=counts,
        )
    except (NumericExperimentError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    records = manifest["records"]
    print(
        f"built train={records['train']} dev={records['dev']} "
        f"focus_share={manifest['balancing']['focus_share']:.3f} "
        f"warnings={','.join(manifest['warnings']) or 'none'} output={args.output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
