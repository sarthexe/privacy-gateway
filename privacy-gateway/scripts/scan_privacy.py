"""Check evaluation artifacts for leaked Nemotron-PII entity values or source text.

Indexes every ground-truth entity value (>= ``MIN_LENGTH`` characters) and a
prefix of every source text from the given normalized splits, then scans each
artifact for whole-token occurrences. Synthetic values include ordinary words
(an occupation "Worker", a label-like password "first_name"), so a match whose
text is also template vocabulary — a word in this repository's source/configs or
in a recorded hardware descriptor — is counted but not failed, and numeric-only
matches (e.g. a count equal to a synthetic PIN) are listed for review. Anything
else fails. Predictions files are validated structurally instead: every record
must decode to identifier labels and integer offsets. Findings are reported as
file, offset, and length only; matched values are never printed.

    python scripts/scan_privacy.py --data-dir D --splits test train \
        --paths reports/pii D/gliner --predictions D/gliner/test.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.gliner_predictions import PredictionsError, iter_records, read_header
from scripts.nemotron_pii import REPOSITORY_ROOT

MIN_LENGTH = 6
TEXT_PREFIX = 40
SCANNED_SUFFIXES = {".json", ".md", ".txt", ".log", ".yaml", ".csv"}
VOCABULARY_SOURCES = ("scripts/*.py", "configs/*.yaml", "app/**/*.py")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def build_index(paths: Iterable[Path]) -> tuple[dict[str, set[str]], int]:
    """Map the first ``MIN_LENGTH`` characters of each sensitive string to the strings."""
    index: dict[str, set[str]] = defaultdict(set)
    count = 0
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                text = record["text"]
                values = [text[e["start"] : e["end"]] for e in record["entities"]]
                values.append(text[:TEXT_PREFIX])
                for value in values:
                    if len(value) >= MIN_LENGTH:
                        index[value[:MIN_LENGTH]].add(value)
                        count += 1
    return index, count


def template_vocabulary(extra: Iterable[str] = ()) -> set[str]:
    """Case-folded words the report generators and configs can legitimately emit."""
    words: set[str] = set()
    for pattern in VOCABULARY_SOURCES:
        for path in REPOSITORY_ROOT.glob(pattern):
            words.update(w.casefold() for w in _WORD.findall(path.read_text(encoding="utf-8")))
    for text in extra:
        words.update(w.casefold() for w in _WORD.findall(text))
    return words


def hardware_strings(value: Any) -> list[str]:
    """Device descriptors (platform, CPU, GPU names) recorded under ``hardware`` keys."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "hardware" and isinstance(item, dict):
                found.extend(str(v) for v in item.values() if isinstance(v, str))
            else:
                found.extend(hardware_strings(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(hardware_strings(item))
    return found


@dataclass(frozen=True)
class Hit:
    offset: int
    length: int
    kind: str  # "leak", "template_vocabulary", or "numeric"


def _is_boundary(content: str, position: int) -> bool:
    character = content[position] if 0 <= position < len(content) else " "
    return not (character.isalnum() or character == "_")


def scan_text(content: str, index: dict[str, set[str]], vocabulary: set[str]) -> list[Hit]:
    """Return whole-token occurrences of sensitive strings, classified."""
    hits: list[Hit] = []
    for position in range(len(content) - MIN_LENGTH + 1):
        candidates = index.get(content[position : position + MIN_LENGTH])
        if not candidates or not _is_boundary(content, position - 1):
            continue
        for value in candidates:
            end = position + len(value)
            if not content.startswith(value, position) or not _is_boundary(content, end):
                continue
            if not any(ch.isalpha() for ch in value):
                kind = "numeric"
            # Words or space-separated phrases of template vocabulary only; any other
            # character (e.g. "@", ".", "-") keeps the match a leak.
            elif all(w.casefold() in vocabulary for w in _WORD.findall(value)) and not re.search(
                r"[^A-Za-z0-9_ ]", value
            ):
                kind = "template_vocabulary"
            else:
                kind = "leak"
            hits.append(Hit(position, len(value), kind))
    return hits


def artifact_files(paths: Sequence[Path], skip: set[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        candidates = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        files.extend(
            p for p in candidates if p.suffix in SCANNED_SUFFIXES and p.resolve() not in skip
        )
    return files


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan artifacts for leaked PII values.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["test", "train"])
    parser.add_argument("--paths", nargs="+", type=Path, default=[])
    parser.add_argument("--predictions", nargs="*", type=Path, default=[])
    args = parser.parse_args(argv)

    failures = 0
    for predictions in args.predictions:
        try:
            header = read_header(predictions)
            records = sum(1 for _ in iter_records(predictions))
        except (PredictionsError, OSError) as error:
            print(f"FAIL predictions {predictions.name}: {error}")
            failures += 1
            continue
        print(
            f"ok   predictions {predictions.name}: {records} records decode to labels/offsets "
            f"only; header sections {sorted(header)}"
        )

    sources = [args.data_dir / f"{split}.jsonl" for split in args.splits]
    index, count = build_index(sources)
    print(f"indexed {count} sensitive strings (>= {MIN_LENGTH} chars) from {len(sources)} splits")
    skip = {p.resolve() for p in (*sources, *args.predictions)}
    files = artifact_files(args.paths, skip)
    descriptors: list[str] = []
    for path in files:
        if path.suffix == ".json":
            try:
                descriptors.extend(hardware_strings(json.loads(path.read_text(encoding="utf-8"))))
            except ValueError:
                pass
    vocabulary = template_vocabulary(descriptors)
    for path in files:
        hits = scan_text(path.read_text(encoding="utf-8", errors="replace"), index, vocabulary)
        leaks = [hit for hit in hits if hit.kind == "leak"]
        numeric = sum(hit.kind == "numeric" for hit in hits)
        template = sum(hit.kind == "template_vocabulary" for hit in hits)
        failures += bool(leaks)
        print(
            f"{'FAIL' if leaks else 'ok  '} {path}: leaks={len(leaks)} "
            f"template_vocabulary={template} numeric_coincidences_to_review={numeric}"
        )
        for hit in leaks[:10]:
            print(f"       match at offset {hit.offset}, length {hit.length}")
    print("privacy scan passed" if not failures else f"privacy scan FAILED ({failures})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
