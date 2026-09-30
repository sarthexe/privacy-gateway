"""External, offset-only store for GLiNER predictions (resumable, append-only JSONL).

Line 1 is a header with the run identity; every following line holds one example:
``{"i": example_index, "c": chunks, "d": duplicates_removed, "p": [[label, start,
end, score], ...]}``. No source text or entity value is ever written. Files must
live outside the repository. Inference and scoring are separate steps so that the
exact-match scorer is independent of the model runtime.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.evaluation_checkpoint import canonical_sha256
from scripts.gliner_evaluation import RawPrediction
from scripts.nemotron_pii import is_within_repository

PREDICTIONS_FORMAT = "privacy-gateway/gliner-predictions"
PREDICTIONS_VERSION = 1
DEV_SELECTION_SEED = "privacy-gateway/gliner-dev/v1"
_LABEL_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class PredictionsError(ValueError):
    """Unusable predictions file; messages never contain source text or values."""


def select_dev_indices(total: int, size: int, seed: str = DEV_SELECTION_SEED) -> list[int]:
    """Deterministic pseudo-random subset of ``range(total)``, returned in file order.

    Ranking indices by a keyed SHA-256 spreads the subset across the whole file
    (unlike a prefix) and is reproducible without any RNG state.
    """
    if not 0 < size <= total:
        raise ValueError("dev subset size must be within the split size")
    ranked = sorted(
        range(total), key=lambda index: hashlib.sha256(f"{seed}:{index}".encode()).digest()
    )
    return sorted(ranked[:size])


@dataclass(frozen=True)
class PredictionRecord:
    example_index: int
    chunks: int
    duplicates_removed: int
    predictions: tuple[RawPrediction, ...]


def encode_record(record: PredictionRecord) -> str:
    return json.dumps(
        {
            "i": record.example_index,
            "c": record.chunks,
            "d": record.duplicates_removed,
            "p": [[p.label, p.start, p.end, p.score] for p in record.predictions],
        },
        separators=(",", ":"),
    )


def _count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PredictionsError("prediction record counters must be non-negative integers")
    return value


def decode_record(line: str) -> PredictionRecord:
    """Parse and validate one record (labels are restricted to identifiers)."""
    try:
        raw = json.loads(line)
        items = raw["p"]
        predictions = []
        for label, start, end, score in items:
            if not isinstance(label, str) or not _LABEL_PATTERN.fullmatch(label):
                raise PredictionsError("prediction labels must be lower-case identifiers")
            begin, finish = _count(start), _count(end)
            if finish <= begin or isinstance(score, bool) or not isinstance(score, int | float):
                raise PredictionsError("prediction offsets or score are invalid")
            predictions.append(RawPrediction(label, begin, finish, float(score)))
        return PredictionRecord(
            _count(raw["i"]), _count(raw["c"]), _count(raw["d"]), tuple(predictions)
        )
    except PredictionsError:
        raise
    except (ValueError, KeyError, TypeError) as error:
        raise PredictionsError("prediction record structure is invalid") from error


def build_header(identity: Mapping[str, Any]) -> str:
    header = {
        "format": PREDICTIONS_FORMAT,
        "version": PREDICTIONS_VERSION,
        "identity": dict(identity),
        "identity_sha256": canonical_sha256(dict(identity)),
    }
    return json.dumps(header, sort_keys=True, separators=(",", ":"))


def read_header(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline()
    try:
        header = json.loads(first)
    except ValueError as error:
        raise PredictionsError("predictions header is not valid JSON") from error
    if not isinstance(header, dict) or header.get("format") != PREDICTIONS_FORMAT:
        raise PredictionsError("predictions file format is not recognized")
    if header.get("version") != PREDICTIONS_VERSION:
        raise PredictionsError("predictions file version is not supported")
    identity = header.get("identity")
    if not isinstance(identity, dict) or header.get("identity_sha256") != canonical_sha256(
        identity
    ):
        raise PredictionsError("predictions header identity is missing or altered")
    return identity


def iter_records(path: Path) -> Iterator[PredictionRecord]:
    """Yield complete records after the header, in file order."""
    with path.open("r", encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            if not line.endswith("\n"):
                raise PredictionsError("predictions file ends with an incomplete record")
            yield decode_record(line)


class PredictionWriter:
    """Append-only writer; each flushed batch is durable before the next begins."""

    def __init__(
        self,
        path: Path,
        identity: Mapping[str, Any],
        expected_indices: Sequence[int],
        resume: bool,
        restart: bool,
    ) -> None:
        if is_within_repository(path):
            raise ValueError("predictions path must be outside the repository")
        if resume and restart:
            raise ValueError("resume and restart are mutually exclusive")
        if path.exists() and not (resume or restart):
            raise PredictionsError("predictions file already exists; pass --resume or --restart")
        if resume and not path.is_file():
            raise PredictionsError("resume requested but no predictions file exists")
        self.path = path
        self.completed = 0
        if resume:
            saved = read_header(path)
            if canonical_sha256(saved) != canonical_sha256(dict(identity)):
                differing = sorted(
                    key for key in set(saved) | set(identity) if saved.get(key) != identity.get(key)
                )
                raise PredictionsError(
                    "predictions file does not match this run: " + ", ".join(differing)
                )
            self._truncate_incomplete_tail()
            for record in iter_records(path):
                if self.completed >= len(expected_indices) or (
                    record.example_index != expected_indices[self.completed]
                ):
                    raise PredictionsError("predictions file records are out of order")
                self.completed += 1
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(build_header(identity) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def _truncate_incomplete_tail(self) -> None:
        # A crash can leave a torn last line; drop it so the batch is simply redone.
        with self.path.open("rb+") as handle:
            data = handle.read()
            keep = data.rfind(b"\n") + 1
            if keep != len(data):
                handle.truncate(keep)

    def append(self, records: Sequence[PredictionRecord]) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(encode_record(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.completed += len(records)
