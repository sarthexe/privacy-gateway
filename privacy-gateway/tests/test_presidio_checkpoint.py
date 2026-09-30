"""Tests for resumable, checkpointed Presidio evaluation with a synthetic detector."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from app.models.privacy import EntitySpan, EntityType
from scripts import batch_evaluation, evaluate_presidio
from scripts.batch_evaluation import EvaluationError, RunOptions, run_evaluation
from scripts.evaluation_checkpoint import (
    CheckpointCorruptError,
    CheckpointError,
    CheckpointIncompatibleError,
    EvaluationState,
    canonical_sha256,
    load_checkpoint,
    write_checkpoint,
)

# Synthetic records; the comments give the expected strict outcome per record.
RECORDS = [
    # PERSON tp, EMAIL tp
    {"text": "alice mail a@b.io", "entities": [["PERSON", 0, 5], ["EMAIL_ADDRESS", 11, 17]]},
    # PERSON tp, PERSON fn
    {"text": "bob and carol", "entities": [["PERSON", 0, 3], ["PERSON", 8, 13]]},
    # nothing
    {"text": "no entities here", "entities": []},
    # EMAIL tp, PERSON fp, DATE (-> DATE_TIME) fn
    {"text": "x@y.io alice", "entities": [["EMAIL_ADDRESS", 0, 6], ["DATE", 7, 12]]},
    # PERSON tp
    {"text": "hello bob", "entities": [["PERSON", 6, 9]]},
    # UNMAPPED fn
    {"text": "carol", "entities": [["UNMAPPED", 0, 5]]},
]
EXPECTED_PER_ENTITY = {
    "DATE_TIME": {"tp": 0, "fp": 0, "fn": 1},
    "EMAIL_ADDRESS": {"tp": 2, "fp": 0, "fn": 0},
    "PERSON": {"tp": 3, "fp": 1, "fn": 1},
    "UNMAPPED": {"tp": 0, "fp": 0, "fn": 1},
}
SENSITIVE_STRINGS = ["alice", "bob", "carol", "a@b.io", "x@y.io", "hello"]
INTERRUPT_ON_TEXT: str | None = None


class KeywordDetector:
    """Deterministic detector fake: fixed names are PERSON, ``@`` tokens are EMAIL."""

    def detect(self, text: str) -> list[EntitySpan]:
        if text == INTERRUPT_ON_TEXT:
            raise KeyboardInterrupt
        spans = [
            EntitySpan(entity_type=EntityType.PERSON, start=m.start(), end=m.end(), value=m[0])
            for m in re.finditer(r"\b(?:alice|bob)\b", text)
        ] + [
            EntitySpan(
                entity_type=EntityType.EMAIL_ADDRESS, start=m.start(), end=m.end(), value=m[0]
            )
            for m in re.finditer(r"\S+@\S+", text)
        ]
        return sorted(spans, key=lambda span: (span.start, span.end))


class OtherDetector(KeywordDetector):
    """A differently configured detector whose checkpoints must not be mixed."""


def keyword_detector() -> KeywordDetector:
    return KeywordDetector()


def other_detector() -> OtherDetector:
    return OtherDetector()


def write_dataset(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            entities = [
                {"type": entity_type, "start": start, "end": end}
                for entity_type, start, end in record["entities"]  # type: ignore[attr-defined]
            ]
            handle.write(json.dumps({"text": record["text"], "entities": entities}) + "\n")


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "data" / "test.jsonl"
    path.parent.mkdir()
    write_dataset(path, RECORDS)
    return path


def options(dataset: Path, **overrides: object) -> RunOptions:
    values: dict[str, object] = {
        "data_path": dataset,
        "checkpoint_path": dataset.parent / "checkpoints" / "presidio_test.json",
        "batch_size": 2,
    }
    values.update(overrides)
    return RunOptions(**values)  # type: ignore[arg-type]


def run(dataset: Path, factory=keyword_detector, **overrides: object):  # type: ignore[no-untyped-def]
    return run_evaluation(options(dataset, **overrides), factory, log=lambda _: None)


def per_entity(state) -> dict[str, dict[str, int]]:  # type: ignore[no-untyped-def]
    return state.to_payload()["metrics"]["per_entity"]


def metrics_only(state) -> dict[str, object]:  # type: ignore[no-untyped-def]
    payload = state.to_payload()
    return {**payload["metrics"], "completed": state.completed_examples}


def reseal(path: Path, mutate) -> None:  # type: ignore[no-untyped-def]
    document = json.loads(path.read_text(encoding="utf-8"))
    document.pop("sha256")
    mutate(document)
    document["sha256"] = canonical_sha256(document)
    path.write_text(json.dumps(document), encoding="utf-8")


def test_full_run_matches_hand_computed_strict_metrics(dataset: Path) -> None:
    state = run(dataset).state
    assert per_entity(state) == EXPECTED_PER_ENTITY
    assert state.aggregate() == {"tp": 5, "fp": 1, "fn": 3}
    assert state.exact_matches == state.harness_exact_matches == 5
    assert state.completed_examples == 6 and state.examples_with_pii == 5
    assert state.dataset_exhausted
    assert state.byte_offset == dataset.stat().st_size


def test_checkpoint_is_written_after_every_batch_and_holds_no_text(dataset: Path) -> None:
    written: list[int] = []
    original = batch_evaluation.write_checkpoint

    def spy(path, identity, state):  # type: ignore[no-untyped-def]
        written.append(state.completed_examples)
        original(path, identity, state)

    batch_evaluation.write_checkpoint = spy
    try:
        result = run(dataset)
    finally:
        batch_evaluation.write_checkpoint = original
    assert written[:3] == [2, 4, 6]

    checkpoint = options(dataset).checkpoint_path
    raw = checkpoint.read_text(encoding="utf-8")
    assert not [value for value in SENSITIVE_STRINGS if value in raw]
    assert not [record["text"] for record in RECORDS if record["text"] in raw]
    assert list(checkpoint.parent.iterdir()) == [checkpoint], "temporary files were left"

    identity, state = load_checkpoint(checkpoint)
    assert identity["dataset"]["split"] == "test"
    assert identity["dataset"]["sha256"] == result.identity["dataset"]["sha256"]
    assert {"ontology_sha256", "evaluation_type_map"} <= set(identity["normalization"])
    assert identity["detector"]["class"].endswith("KeywordDetector")
    assert identity["matching"]["strategy"] == "strict"
    metrics = json.loads(raw)["metrics"]
    assert metrics["aggregate"] == {"tp": 5, "fp": 1, "fn": 3}
    assert metrics["exact_matches"] == 5
    assert json.loads(raw)["progress"]["elapsed_seconds"] >= 0
    assert per_entity(state) == EXPECTED_PER_ENTITY


def test_interrupted_run_resumes_to_identical_metrics(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = run(dataset, checkpoint_path=dataset.parent / "baseline.json").state

    # Interrupt while detecting the 4th record (second batch): only batch one is durable.
    monkeypatch.setattr(sys.modules[__name__], "INTERRUPT_ON_TEXT", RECORDS[3]["text"])
    with pytest.raises(KeyboardInterrupt):
        run(dataset)
    _, interrupted = load_checkpoint(options(dataset).checkpoint_path)
    assert interrupted.completed_examples == 2
    monkeypatch.setattr(sys.modules[__name__], "INTERRUPT_ON_TEXT", None)

    resumed = run(dataset, resume=True).state
    assert metrics_only(resumed) == metrics_only(baseline)
    assert [session["resumed_from"] for session in resumed.sessions] == [0, 2]


def test_limit_evaluates_prefix_and_resume_extends_it(dataset: Path) -> None:
    pilot = run(dataset, limit=3).state
    assert pilot.completed_examples == 3 and not pilot.dataset_exhausted
    assert pilot.aggregate() == {"tp": 3, "fp": 0, "fn": 1}

    extended = run(dataset, resume=True).state
    assert extended.completed_examples == 6 and extended.dataset_exhausted
    assert per_entity(extended) == EXPECTED_PER_ENTITY

    # Resuming a finished run scores nothing new and changes no metrics.
    again = run(dataset, resume=True).state
    assert metrics_only(again) == metrics_only(extended)
    assert again.sessions[-1]["examples"] == 0


def test_limit_equal_to_dataset_size_marks_split_complete(dataset: Path) -> None:
    assert run(dataset, limit=len(RECORDS)).state.dataset_exhausted


def test_limit_below_checkpoint_progress_is_rejected(dataset: Path) -> None:
    run(dataset, limit=4)
    with pytest.raises(CheckpointError, match="more examples than the limit"):
        run(dataset, resume=True, limit=2)


def test_existing_checkpoint_requires_explicit_resume_or_restart(dataset: Path) -> None:
    run(dataset, limit=2)
    with pytest.raises(CheckpointError, match="--resume or --restart"):
        run(dataset)
    assert run(dataset, restart=True).state.sessions[0]["resumed_from"] == 0


def test_resume_without_checkpoint_is_rejected(dataset: Path) -> None:
    with pytest.raises(CheckpointError, match="no checkpoint exists"):
        run(dataset, resume=True)


def test_checkpoint_inside_repository_is_rejected(dataset: Path) -> None:
    inside = Path(evaluate_presidio.__file__).resolve().parents[1] / "reports" / "c.json"
    with pytest.raises(ValueError, match="outside the repository"):
        run(dataset, checkpoint_path=inside)


def _replace_text(old: str, new: str):  # type: ignore[no-untyped-def]
    def corrupt(path: Path) -> None:
        path.write_text(path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")

    return corrupt


@pytest.mark.parametrize(
    ("corrupt", "reason"),
    [
        pytest.param(_replace_text("", "{"), "not readable JSON", id="garbled"),
        pytest.param(
            _replace_text('"completed_examples": 4', '"completed_examples": 5'),
            "integrity hash",
            id="tampered-without-reseal",
        ),
        pytest.param(
            lambda path: reseal(
                path, lambda doc: doc["metrics"]["aggregate"].__setitem__("tp", 99)
            ),
            "aggregate counts",
            id="inconsistent-aggregate",
        ),
        pytest.param(
            lambda path: reseal(
                path, lambda doc: doc["metrics"]["per_entity"]["PERSON"].__setitem__("tp", -1)
            ),
            "non-negative",
            id="negative-count",
        ),
        pytest.param(
            lambda path: reseal(path, lambda doc: doc["progress"].__setitem__("byte_offset", 7)),
            "record boundary",
            id="offset-not-on-record-boundary",
        ),
    ],
)
def test_corrupted_checkpoint_is_rejected(dataset: Path, corrupt, reason: str) -> None:  # type: ignore[no-untyped-def]
    run(dataset, limit=4)
    corrupt(options(dataset).checkpoint_path)
    with pytest.raises(CheckpointError, match=reason):
        run(dataset, resume=True)


def test_corrupted_checkpoint_raises_corruption_error(dataset: Path) -> None:
    run(dataset, limit=2)
    checkpoint = options(dataset).checkpoint_path
    checkpoint.write_bytes(checkpoint.read_bytes()[:50])
    with pytest.raises(CheckpointCorruptError):
        load_checkpoint(checkpoint)


def test_changed_dataset_is_incompatible(dataset: Path) -> None:
    run(dataset, limit=2)
    write_dataset(dataset, [*RECORDS, {"text": "bob", "entities": [["PERSON", 0, 3]]}])
    with pytest.raises(CheckpointIncompatibleError, match="dataset.sha256"):
        run(dataset, resume=True)


def test_changed_detector_is_incompatible(dataset: Path) -> None:
    run(dataset, limit=2)
    with pytest.raises(CheckpointIncompatibleError, match="detector.class"):
        run(dataset, factory=other_detector, resume=True)


def test_changed_normalization_is_incompatible(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run(dataset, limit=2)
    changed = {**batch_evaluation.normalization_identity(), "ontology_sha256": "0" * 64}
    monkeypatch.setattr(batch_evaluation, "normalization_identity", lambda: changed)
    with pytest.raises(CheckpointIncompatibleError, match="normalization.ontology_sha256"):
        run(dataset, resume=True)


def test_unsupported_checkpoint_version_is_incompatible(dataset: Path) -> None:
    run(dataset, limit=2)
    reseal(options(dataset).checkpoint_path, lambda doc: doc.__setitem__("version", 999))
    with pytest.raises(CheckpointIncompatibleError, match="version"):
        run(dataset, resume=True)


def test_failed_atomic_write_keeps_previous_checkpoint(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = run(dataset, limit=2)
    checkpoint = options(dataset).checkpoint_path
    before = checkpoint.read_bytes()

    def fail(*_: object) -> None:
        raise OSError("simulated failure")

    monkeypatch.setattr("scripts.evaluation_checkpoint.os.replace", fail)
    with pytest.raises(OSError):
        write_checkpoint(checkpoint, result.identity, result.state)
    assert checkpoint.read_bytes() == before
    assert list(checkpoint.parent.iterdir()) == [checkpoint]


def test_free_text_entity_label_is_never_checkpointed(dataset: Path) -> None:
    write_dataset(dataset, [{"text": "alice", "entities": [["alice smith", 0, 5]]}])
    with pytest.raises(CheckpointError, match="upper-case identifiers"):
        run(dataset)
    assert not options(dataset).checkpoint_path.exists()


def test_invalid_record_error_does_not_echo_content(dataset: Path) -> None:
    dataset.write_text('{"text": "alice", "entities": [{"type": "PERSON"}]}\n', encoding="utf-8")
    with pytest.raises(EvaluationError) as error:
        run(dataset)
    assert "index 0" in str(error.value) and "alice" not in str(error.value)


def test_report_is_aggregate_only_and_marks_partial_runs(dataset: Path) -> None:
    report = evaluate_presidio.build_report(run(dataset, limit=3))
    assert report["complete_split"] is False and report["examples_evaluated"] == 3
    rendered = json.dumps(report) + evaluate_presidio.render_markdown(report, "cmd")
    assert not [value for value in SENSITIVE_STRINGS if value in rendered]


def test_cli_refuses_to_overwrite_or_reuse_corrupt_checkpoint(dataset: Path) -> None:
    checkpoint = dataset.parent / "cli.json"
    checkpoint.write_text("not json", encoding="utf-8")
    base = ["--data-dir", str(dataset.parent), "--checkpoint", str(checkpoint)]
    assert evaluate_presidio.main(base) == 2
    assert evaluate_presidio.main([*base, "--resume"]) == 2
    assert checkpoint.read_text(encoding="utf-8") == "not json"


def test_worker_pool_matches_in_process_metrics(dataset: Path) -> None:
    sequential = run(dataset, checkpoint_path=dataset.parent / "w1.json").state
    parallel = run(dataset, checkpoint_path=dataset.parent / "w2.json", workers=2).state
    assert metrics_only(parallel) == metrics_only(sequential)
    assert parallel.sessions[0]["workers"] == 2


def test_nested_same_type_predictions_do_not_hide_exact_matches() -> None:
    # Presidio pattern seen on Nemotron: a shorter nested DATE_TIME listed first.
    batch = batch_evaluation.Batch(items=[(0, b"{}\n")], end_offset=3, reached_eof=True)
    truth = (("DATE_TIME", 235, 255),)
    predicted = (("DATE_TIME", 235, 242), ("DATE_TIME", 235, 255))
    state = EvaluationState()
    batch_evaluation.score_batch(state, batch, [(truth, predicted)])
    assert state.aggregate() == {"tp": 1, "fp": 1, "fn": 0}
    assert state.harness_exact_matches == 1
    assert state.harness_greedy_exact_matches == 0  # nervaluate's order-dependent undercount
