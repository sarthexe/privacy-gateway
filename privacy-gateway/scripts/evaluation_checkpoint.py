"""Atomic, privacy-safe checkpoints for resumable offline PII evaluations.

A checkpoint holds only run identity, progress counters, and aggregate metrics.
Source text and entity values are never stored; entity labels are restricted to
upper-case identifiers and error samples to offset-only records.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CHECKPOINT_FORMAT = "privacy-gateway/pii-evaluation-checkpoint"
CHECKPOINT_VERSION = 1
ERROR_SAMPLE_LIMIT = 25
HASH_CHUNK_BYTES = 1 << 20

_LABEL_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_SAMPLE_KEYS = {"example_index", "entity_type", "start", "end", "failure_group"}
_FAILURE_GROUP_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class CheckpointError(ValueError):
    """Base class for checkpoints that must not be used to continue a run."""


class CheckpointCorruptError(CheckpointError):
    """The checkpoint is unreadable, fails its integrity seal, or is inconsistent."""


class CheckpointIncompatibleError(CheckpointError):
    """The checkpoint belongs to a different dataset, ontology, or detector."""


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks so large datasets are never fully loaded."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def count_lines_before(path: Path, byte_offset: int) -> int:
    """Count newline-terminated records in ``[0, byte_offset)`` in bounded chunks."""
    remaining, newlines = byte_offset, 0
    with path.open("rb") as handle:
        while remaining > 0:
            chunk = handle.read(min(HASH_CHUNK_BYTES, remaining))
            if not chunk:
                break
            newlines += chunk.count(b"\n")
            remaining -= len(chunk)
    return newlines


def canonical_sha256(value: object) -> str:
    """Hash a JSON-compatible value independent of key order and whitespace."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _require_label(label: object) -> str:
    if not isinstance(label, str) or not _LABEL_PATTERN.fullmatch(label):
        raise CheckpointError("entity labels must be upper-case identifiers")
    return label


def _require_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CheckpointCorruptError("checkpoint counters must be non-negative integers")
    return value


def _require_sample(sample: object) -> dict[str, int | str]:
    if not isinstance(sample, Mapping) or set(sample) != _SAMPLE_KEYS:
        raise CheckpointError("error samples must contain offset-only fields")
    group = sample["failure_group"]
    if not isinstance(group, str) or not _FAILURE_GROUP_PATTERN.fullmatch(group):
        raise CheckpointError("error sample failure groups must be identifiers")
    return {
        "example_index": _require_count(sample["example_index"]),
        "entity_type": _require_label(sample["entity_type"]),
        "start": _require_count(sample["start"]),
        "end": _require_count(sample["end"]),
        "failure_group": group,
    }


def _group_counter(raw: object) -> Counter[str]:
    if not isinstance(raw, Mapping):
        raise CheckpointCorruptError("failure groups must be an object")
    counter: Counter[str] = Counter()
    for group, count in raw.items():
        if not isinstance(group, str) or not _FAILURE_GROUP_PATTERN.fullmatch(group):
            raise CheckpointCorruptError("failure group names must be identifiers")
        counter[group] = _require_count(count)
    return counter


@dataclass
class EvaluationState:
    """Mergeable aggregate state of a strict exact-span evaluation."""

    completed_examples: int = 0
    byte_offset: int = 0
    dataset_exhausted: bool = False
    examples_with_pii: int = 0
    harness_exact_matches: int = 0
    # nervaluate strict in detector order; see score_batch for why it can be lower.
    harness_greedy_exact_matches: int = 0
    true_positives: Counter[str] = field(default_factory=Counter)
    false_positives: Counter[str] = field(default_factory=Counter)
    false_negatives: Counter[str] = field(default_factory=Counter)
    false_negative_groups: Counter[str] = field(default_factory=Counter)
    false_positive_groups: Counter[str] = field(default_factory=Counter)
    false_negative_samples: list[dict[str, int | str]] = field(default_factory=list)
    false_positive_samples: list[dict[str, int | str]] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    sessions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def exact_matches(self) -> int:
        return sum(self.true_positives.values())

    def aggregate(self) -> dict[str, int]:
        return {
            "tp": sum(self.true_positives.values()),
            "fp": sum(self.false_positives.values()),
            "fn": sum(self.false_negatives.values()),
        }

    def entity_types(self) -> list[str]:
        return sorted({*self.true_positives, *self.false_positives, *self.false_negatives})

    def add_error_records(
        self,
        false_negatives: Iterable[Mapping[str, int | str]],
        false_positives: Iterable[Mapping[str, int | str]],
    ) -> None:
        """Count failure groups and retain the first bounded offset-only samples."""
        for records, groups, samples in (
            (false_negatives, self.false_negative_groups, self.false_negative_samples),
            (false_positives, self.false_positive_groups, self.false_positive_samples),
        ):
            for record in records:
                groups[str(record["failure_group"])] += 1
                if len(samples) < ERROR_SAMPLE_LIMIT:
                    samples.append(dict(record))

    def to_payload(self) -> dict[str, Any]:
        per_entity = {
            _require_label(entity_type): {
                "tp": self.true_positives.get(entity_type, 0),
                "fp": self.false_positives.get(entity_type, 0),
                "fn": self.false_negatives.get(entity_type, 0),
            }
            for entity_type in self.entity_types()
        }
        return {
            "progress": {
                "completed_examples": self.completed_examples,
                "byte_offset": self.byte_offset,
                "dataset_exhausted": self.dataset_exhausted,
                "elapsed_seconds": round(self.elapsed_seconds, 3),
                "sessions": self.sessions,
            },
            "metrics": {
                "aggregate": self.aggregate(),
                "exact_matches": self.exact_matches,
                "harness_exact_matches": self.harness_exact_matches,
                "harness_greedy_exact_matches": self.harness_greedy_exact_matches,
                "examples_with_pii": self.examples_with_pii,
                "per_entity": per_entity,
                "false_negative_groups": dict(sorted(self.false_negative_groups.items())),
                "false_positive_groups": dict(sorted(self.false_positive_groups.items())),
                "false_negative_samples": [
                    _require_sample(sample) for sample in self.false_negative_samples
                ],
                "false_positive_samples": [
                    _require_sample(sample) for sample in self.false_positive_samples
                ],
            },
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> EvaluationState:
        """Rebuild state, rejecting any structural or arithmetic inconsistency."""
        try:
            progress, metrics = payload["progress"], payload["metrics"]
            state = cls(
                completed_examples=_require_count(progress["completed_examples"]),
                byte_offset=_require_count(progress["byte_offset"]),
                dataset_exhausted=progress["dataset_exhausted"],
                examples_with_pii=_require_count(metrics["examples_with_pii"]),
                harness_exact_matches=_require_count(metrics["harness_exact_matches"]),
                harness_greedy_exact_matches=_require_count(
                    metrics["harness_greedy_exact_matches"]
                ),
                false_negative_groups=_group_counter(metrics["false_negative_groups"]),
                false_positive_groups=_group_counter(metrics["false_positive_groups"]),
                false_negative_samples=[
                    _require_sample(sample) for sample in metrics["false_negative_samples"]
                ],
                false_positive_samples=[
                    _require_sample(sample) for sample in metrics["false_positive_samples"]
                ],
                elapsed_seconds=float(progress["elapsed_seconds"]),
                sessions=list(progress["sessions"]),
            )
            for entity_type, counts in metrics["per_entity"].items():
                _require_label(entity_type)
                state.true_positives[entity_type] = _require_count(counts["tp"])
                state.false_positives[entity_type] = _require_count(counts["fp"])
                state.false_negatives[entity_type] = _require_count(counts["fn"])
            aggregate = {
                key: _require_count(metrics["aggregate"][key]) for key in ("tp", "fp", "fn")
            }
            exact_matches = _require_count(metrics["exact_matches"])
        except CheckpointError as error:
            raise CheckpointCorruptError(str(error)) from error
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise CheckpointCorruptError("checkpoint structure is invalid") from error
        if not isinstance(state.dataset_exhausted, bool) or state.elapsed_seconds < 0:
            raise CheckpointCorruptError("checkpoint progress fields are invalid")
        if not all(
            isinstance(session, dict)
            and all(
                isinstance(key, str) and isinstance(value, int | float)
                for key, value in session.items()
            )
            for session in state.sessions
        ):
            raise CheckpointCorruptError("checkpoint sessions must contain numeric fields only")
        if aggregate != state.aggregate():
            raise CheckpointCorruptError("aggregate counts do not equal per-entity totals")
        if not exact_matches == state.exact_matches == state.harness_exact_matches:
            raise CheckpointCorruptError("exact-match counts are inconsistent")
        if state.harness_greedy_exact_matches > state.exact_matches:
            raise CheckpointCorruptError("greedy harness count exceeds exact matches")
        if state.examples_with_pii > state.completed_examples:
            raise CheckpointCorruptError("examples with PII exceed completed examples")
        if (state.completed_examples == 0) != (state.byte_offset == 0):
            raise CheckpointCorruptError("completed examples and byte offset disagree")
        return state


def build_document(identity: Mapping[str, Any], state: EvaluationState) -> dict[str, Any]:
    """Assemble a sealed checkpoint document."""
    document: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "identity": dict(identity),
        **state.to_payload(),
        "updated_at_unix": round(time.time(), 3),
    }
    document["sha256"] = canonical_sha256(document)
    return document


def write_checkpoint(path: Path, identity: Mapping[str, Any], state: EvaluationState) -> None:
    """Atomically replace ``path`` so readers see either the old or new checkpoint."""
    document = build_document(identity, state)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _replace_with_retry(source: Path, target: Path, attempts: int = 10) -> None:
    # Windows refuses to replace a file another process (indexer, antivirus, a
    # concurrent reader) briefly holds open; retry before surfacing the error.
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.05 * (attempt + 1))


def load_checkpoint(path: Path) -> tuple[dict[str, Any], EvaluationState]:
    """Read and verify a checkpoint's seal and structure; return identity and state."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointCorruptError("checkpoint is not readable JSON") from error
    if not isinstance(document, dict):
        raise CheckpointCorruptError("checkpoint must be a JSON object")
    if document.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointIncompatibleError("checkpoint format is not recognized")
    if document.get("version") != CHECKPOINT_VERSION:
        raise CheckpointIncompatibleError("checkpoint version is not supported")
    seal = document.pop("sha256", None)
    if not isinstance(seal, str) or seal != canonical_sha256(document):
        raise CheckpointCorruptError("checkpoint integrity hash does not match its contents")
    identity = document.get("identity")
    if not isinstance(identity, dict):
        raise CheckpointCorruptError("checkpoint identity is missing")
    return identity, EvaluationState.from_payload(document)


def identity_differences(
    expected: Mapping[str, Any], actual: Mapping[str, Any], prefix: str = ""
) -> list[str]:
    """Return dotted key paths whose values differ, without echoing the values."""
    differences: list[str] = []
    for key in sorted(set(expected) | set(actual)):
        path = f"{prefix}{key}"
        left, right = expected.get(key), actual.get(key)
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            differences.extend(identity_differences(left, right, f"{path}."))
        elif left != right:
            differences.append(path)
    return differences


def require_compatible(
    expected: Mapping[str, Any], actual: Mapping[str, Any], sections: Iterable[str]
) -> None:
    """Reject a checkpoint whose identity differs in any of ``sections``."""
    differences = identity_differences(
        {section: expected.get(section) for section in sections},
        {section: actual.get(section) for section in sections},
    )
    if differences:
        raise CheckpointIncompatibleError(
            "checkpoint does not match this run: " + ", ".join(differences)
        )
