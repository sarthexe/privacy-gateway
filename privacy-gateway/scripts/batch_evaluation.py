"""Streaming, checkpointed exact-span evaluation engine for the Presidio baseline.

The normalized split is read in bounded batches. After every batch an atomic,
PII-free checkpoint is written so an interrupted run can resume. Worker
processes use ``spawn`` so behavior is identical on Windows, macOS, and Linux;
everything a worker needs lives in this importable module, not in ``__main__``.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import io
import json
import multiprocessing
import signal
import time
from collections import Counter, deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scripts.evaluation_checkpoint import (
    CheckpointError,
    EvaluationState,
    canonical_sha256,
    count_lines_before,
    load_checkpoint,
    require_compatible,
    sha256_file,
    write_checkpoint,
)
from scripts.nemotron_pii import DATASET_ID, is_within_repository, load_ontology
from scripts.presidio_evaluation import (
    EVALUATION_TYPE_MAP,
    ExactMatch,
    PresidioEvaluationAdapter,
    build_error_summary,
    harness_strict_metrics,
    validate_normalized_example,
)

if TYPE_CHECKING:
    # Type-only so importing the streaming/scoring engine does not load Presidio.
    from app.detection import PIIDetector

SPLIT = "test"
DEFAULT_BATCH_SIZE = 1_000
NORMALIZED_RECORD_SCHEMA = "jsonl:text+entities[type,start,end];half-open;v1"

DetectorFactory = Callable[[], "PIIDetector"]
EntityTuple = tuple[str, int, int]
Prediction = tuple[tuple[EntityTuple, ...], tuple[EntityTuple, ...]]


class EvaluationError(RuntimeError):
    """Data or harness failure; messages never contain source text or values."""


def create_presidio_detector() -> PIIDetector:
    """Build the unmodified Phase 1 runtime detector."""
    from app.detection import PresidioPIIDetector

    return PresidioPIIDetector()


# --- Run identity -------------------------------------------------------------------------


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_source_sha256(target: type) -> str | None:
    module = inspect.getmodule(target)
    try:
        source = inspect.getsource(module) if module else None
    except (OSError, TypeError):
        source = None
    if source is None:
        return None
    return hashlib.sha256(source.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def describe_detector(detector: PIIDetector) -> dict[str, Any]:
    """Describe the loaded detector; any change here invalidates a checkpoint."""
    detector_type = type(detector)
    description: dict[str, Any] = {
        "class": f"{detector_type.__module__}.{detector_type.__qualname__}",
        "source_sha256": _module_source_sha256(detector_type),
    }
    analyzer = getattr(detector, "_analyzer", None)
    if analyzer is None:
        return description
    recognizers = getattr(getattr(analyzer, "registry", None), "recognizers", [])
    models = getattr(getattr(analyzer, "nlp_engine", None), "nlp", None) or {}
    description.update(
        {
            "presidio_analyzer": _package_version("presidio-analyzer"),
            "spacy": _package_version("spacy"),
            "nlp_models": {
                language: f"{model.meta.get('lang')}_{model.meta.get('name')}"
                f"=={model.meta.get('version')}"
                for language, model in sorted(models.items())
            },
            "recognizers": sorted(
                f"{recognizer.name}:{'|'.join(sorted(recognizer.supported_entities))}"
                for recognizer in recognizers
            ),
            "supported_entities": sorted(getattr(detector, "_SUPPORTED", [])),
        }
    )
    return description


def dataset_identity(data_path: Path) -> dict[str, Any]:
    return {
        "dataset_id": DATASET_ID,
        "split": SPLIT,
        "file_name": data_path.name,
        "size_bytes": data_path.stat().st_size,
        "sha256": sha256_file(data_path),
    }


def normalization_identity() -> dict[str, Any]:
    return {
        "ontology_sha256": canonical_sha256(load_ontology()),
        "record_schema": NORMALIZED_RECORD_SCHEMA,
        "evaluation_type_map": dict(sorted(EVALUATION_TYPE_MAP.items())),
    }


def matching_identity() -> dict[str, Any]:
    return {
        "strategy": "strict",
        "span": "exact half-open [start, end)",
        "entity_type": "exact",
        "harness": "nervaluate",
        "harness_version": _package_version("nervaluate"),
    }


# --- Detection backends -------------------------------------------------------------------

_WORKER_ADAPTER: PresidioEvaluationAdapter | None = None
_WORKER_DETECTOR: PIIDetector | None = None


def _as_tuples(entities: Sequence[ExactMatch]) -> tuple[EntityTuple, ...]:
    return tuple((entity.entity_type, entity.start, entity.end) for entity in entities)


def predict_line(adapter: PresidioEvaluationAdapter, example_index: int, line: bytes) -> Prediction:
    """Parse, validate, and detect one record, returning offsets and types only."""
    try:
        record = json.loads(line)
    except ValueError:  # JSONDecodeError and UnicodeDecodeError
        raise EvaluationError(
            f"invalid normalized record at index {example_index}: not valid UTF-8 JSON"
        ) from None
    try:
        text, truth = validate_normalized_example(record)
    except ValueError as error:
        # The validator only raises fixed messages that never include record content.
        message = f"invalid normalized record at index {example_index}: {error}"
        raise EvaluationError(message) from None
    return _as_tuples(truth), _as_tuples(adapter.predict(text))


def _initialize_worker(factory: DetectorFactory) -> None:
    # The parent owns Ctrl+C so it can report the last checkpoint and stop the pool.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global _WORKER_ADAPTER, _WORKER_DETECTOR
    _WORKER_DETECTOR = factory()
    _WORKER_ADAPTER = PresidioEvaluationAdapter(_WORKER_DETECTOR)


def _worker_describe() -> dict[str, Any]:
    assert _WORKER_DETECTOR is not None
    return describe_detector(_WORKER_DETECTOR)


def _worker_predict(item: tuple[int, bytes]) -> Prediction:
    assert _WORKER_ADAPTER is not None
    return predict_line(_WORKER_ADAPTER, *item)


class InProcessBackend:
    """Sequential detection in the parent process (``--workers 1``)."""

    def __init__(self, factory: DetectorFactory) -> None:
        self._detector = factory()
        self._adapter = PresidioEvaluationAdapter(self._detector)

    def describe(self) -> dict[str, Any]:
        return describe_detector(self._detector)

    def submit(self, items: list[tuple[int, bytes]]) -> list[tuple[int, bytes]]:
        return items

    def result(self, handle: list[tuple[int, bytes]]) -> list[Prediction]:
        return [predict_line(self._adapter, index, line) for index, line in handle]

    def close(self) -> None:
        return None


class ProcessPoolBackend:
    """Parallel detection with one detector per ``spawn`` worker process."""

    def __init__(self, factory: DetectorFactory, workers: int) -> None:
        self._workers = workers
        context = multiprocessing.get_context("spawn")
        self._pool = context.Pool(workers, initializer=_initialize_worker, initargs=(factory,))

    def describe(self) -> dict[str, Any]:
        return self._pool.apply(_worker_describe)

    def submit(self, items: list[tuple[int, bytes]]) -> Any:
        chunksize = max(1, len(items) // (self._workers * 8))
        return self._pool.map_async(_worker_predict, items, chunksize=chunksize)

    def result(self, handle: Any) -> list[Prediction]:
        return handle.get()

    def close(self) -> None:
        self._pool.terminate()
        self._pool.join()


# --- Streaming and scoring ----------------------------------------------------------------


@dataclass
class Batch:
    items: list[tuple[int, bytes]]
    end_offset: int
    reached_eof: bool


def read_batches(
    handle: io.BufferedReader,
    start_index: int,
    start_offset: int,
    batch_size: int,
    limit: int | None,
) -> Iterator[Batch]:
    """Yield bounded batches of raw lines and the byte offset after each batch."""
    index, offset = start_index, start_offset
    while limit is None or index < limit:
        size = batch_size if limit is None else min(batch_size, limit - index)
        items: list[tuple[int, bytes]] = []
        while len(items) < size and (line := handle.readline()):
            if not line.endswith(b"\n"):
                raise EvaluationError(f"record at index {index} is not newline-terminated")
            offset += len(line)
            items.append((index, line))
            index += 1
        reached_eof = not handle.peek(1)
        if items:
            yield Batch(items, offset, reached_eof)
        if reached_eof:
            return


def score_batch(state: EvaluationState, batch: Batch, predictions: list[Prediction]) -> None:
    """Fold one batch into the aggregate state and cross-check it with nervaluate."""
    batch_truth: list[list[ExactMatch]] = []
    batch_predicted: list[list[ExactMatch]] = []
    batch_exact_first: list[list[ExactMatch]] = []
    batch_matches = 0
    for (example_index, _), (truth_tuples, predicted_tuples) in zip(
        batch.items, predictions, strict=True
    ):
        truth = [ExactMatch(*entity) for entity in truth_tuples]
        predicted = [ExactMatch(*entity) for entity in predicted_tuples]
        truth_counts, predicted_counts = Counter(truth), Counter(predicted)
        matches = truth_counts & predicted_counts
        for entity, count in matches.items():
            state.true_positives[entity.entity_type] += count
            batch_matches += count
        for entity, count in (predicted_counts - matches).items():
            state.false_positives[entity.entity_type] += count
        for entity, count in (truth_counts - matches).items():
            state.false_negatives[entity.entity_type] += count
        state.examples_with_pii += bool(truth)
        state.add_error_records(*build_error_summary(example_index, truth, predicted))
        batch_truth.append(truth)
        batch_predicted.append(predicted)
        # Stable sort: exact matches first, detector order otherwise preserved.
        batch_exact_first.append(sorted(predicted, key=lambda entity: entity not in truth_counts))
    tags = sorted({entity.entity_type for row in batch_truth + batch_predicted for entity in row})

    def harness_correct(predicted_rows: list[list[ExactMatch]]) -> int:
        metrics = harness_strict_metrics(batch_truth, predicted_rows, tags)
        return sum(int(row["correct"]) for row in metrics.values())

    # nervaluate's strict matcher is greedy in prediction order: an overlapping
    # prediction listed before an exact one consumes the true entity as
    # "incorrect", so the exact prediction is then scored spurious. Presidio emits
    # such nested spans (notably DATE_TIME). Exact matching is order-independent,
    # so the cross-check presents exact matches first and must agree exactly; the
    # detector-order count is kept as a diagnostic of the greedy undercount.
    harness_matches = harness_correct(batch_exact_first)
    if harness_matches != batch_matches:
        raise EvaluationError("strict harness result did not match adapter exact-match count")
    state.harness_exact_matches += harness_matches
    state.harness_greedy_exact_matches += harness_correct(batch_predicted)
    state.completed_examples += len(batch.items)
    state.byte_offset = batch.end_offset
    state.dataset_exhausted = batch.reached_eof


# --- Orchestration ------------------------------------------------------------------------


@dataclass
class RunOptions:
    data_path: Path
    checkpoint_path: Path
    resume: bool = False
    restart: bool = False
    limit: int | None = None
    batch_size: int = DEFAULT_BATCH_SIZE
    workers: int = 1


@dataclass
class RunResult:
    identity: dict[str, Any]
    state: EvaluationState
    limit: int | None
    startup_seconds: float
    session_examples: int
    session_seconds: float


def _print(message: str) -> None:
    print(message, flush=True)


def _verify_resume_position(data_path: Path, state: EvaluationState) -> None:
    if state.byte_offset > data_path.stat().st_size:
        raise CheckpointError("checkpoint offset is beyond the end of the dataset")
    if state.byte_offset:
        with data_path.open("rb") as handle:
            handle.seek(state.byte_offset - 1)
            if handle.read(1) != b"\n":
                raise CheckpointError("checkpoint offset is not on a record boundary")
    if count_lines_before(data_path, state.byte_offset) != state.completed_examples:
        raise CheckpointError("checkpoint offset does not align with completed examples")


def run_evaluation(
    options: RunOptions,
    detector_factory: DetectorFactory = create_presidio_detector,
    log: Callable[[str], None] = _print,
) -> RunResult:
    """Evaluate (or continue evaluating) the split, checkpointing after every batch."""
    if options.batch_size < 1 or options.workers < 1:
        raise ValueError("batch size and worker count must be positive")
    if options.limit is not None and options.limit < 1:
        raise ValueError("limit must be positive")
    if options.resume and options.restart:
        raise ValueError("resume and restart are mutually exclusive")
    if is_within_repository(options.checkpoint_path):
        raise ValueError("checkpoint path must be outside the repository")
    if options.checkpoint_path.exists() and not (options.resume or options.restart):
        raise CheckpointError("checkpoint already exists; pass --resume or --restart")
    if options.resume and not options.checkpoint_path.is_file():
        raise CheckpointError("resume requested but no checkpoint exists")

    identity: dict[str, Any] = {
        "dataset": dataset_identity(options.data_path),
        "normalization": normalization_identity(),
        "matching": matching_identity(),
    }
    state, saved_identity = EvaluationState(), None
    if options.resume:
        saved_identity, state = load_checkpoint(options.checkpoint_path)
        require_compatible(identity, saved_identity, ("dataset", "normalization", "matching"))
        _verify_resume_position(options.data_path, state)
        if options.limit is not None and state.completed_examples > options.limit:
            raise CheckpointError("checkpoint already covers more examples than the limit")
        log(f"resuming from checkpoint: completed_examples={state.completed_examples}")

    started = time.perf_counter()
    backend: InProcessBackend | ProcessPoolBackend = (
        InProcessBackend(detector_factory)
        if options.workers == 1
        else ProcessPoolBackend(detector_factory, options.workers)
    )
    try:
        identity["detector"] = backend.describe()
        if saved_identity is not None:
            require_compatible(identity, saved_identity, ("detector",))
        startup_seconds = time.perf_counter() - started
        log(f"detector ready: workers={options.workers} startup_seconds={startup_seconds:.1f}")

        first_index, first_offset = state.completed_examples, state.byte_offset
        dataset_size = identity["dataset"]["size_bytes"]
        session: dict[str, Any] = {
            "resumed_from": first_index,
            "workers": options.workers,
            "batch_size": options.batch_size,
            "examples": 0,
            "elapsed_seconds": 0.0,
        }
        state.sessions.append(session)
        base_elapsed, session_start = state.elapsed_seconds, time.perf_counter()

        def checkpoint() -> None:
            now = time.perf_counter()
            session["examples"] = state.completed_examples - first_index
            session["elapsed_seconds"] = round(now - session_start, 3)
            state.elapsed_seconds = base_elapsed + (now - session_start)
            write_checkpoint(options.checkpoint_path, identity, state)

        def complete(batch: Batch, handle: Any) -> None:
            score_batch(state, batch, backend.result(handle))
            checkpoint()
            elapsed = max(session["elapsed_seconds"], 1e-9)
            if options.limit is not None:
                remaining = (options.limit - state.completed_examples) * elapsed
                remaining /= max(session["examples"], 1)
            else:
                remaining = (dataset_size - state.byte_offset) * elapsed
                remaining /= max(state.byte_offset - first_offset, 1)
            log(
                f"checkpoint: completed_examples={state.completed_examples} "
                f"examples_per_second={session['examples'] / elapsed:.1f} "
                f"eta_seconds={remaining:.0f}"
            )

        if not (state.dataset_exhausted or state.completed_examples == options.limit):
            with options.data_path.open("rb") as data_file:
                data_file.seek(state.byte_offset)
                # Keep one batch queued behind the one being scored so workers do not
                # idle while the parent aggregates and checkpoints. Memory stays bounded
                # to two batches of raw lines.
                pending: deque[tuple[Batch, Any]] = deque()
                for batch in read_batches(
                    data_file, first_index, state.byte_offset, options.batch_size, options.limit
                ):
                    pending.append((batch, backend.submit(batch.items)))
                    if len(pending) > 1:
                        complete(*pending.popleft())
                while pending:
                    complete(*pending.popleft())
        # Also records an empty session when resuming an already finished checkpoint.
        checkpoint()
    finally:
        backend.close()
    return RunResult(
        identity=identity,
        state=state,
        limit=options.limit,
        startup_seconds=startup_seconds,
        session_examples=int(session["examples"]),
        session_seconds=float(session["elapsed_seconds"]),
    )
