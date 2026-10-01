from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import prepare_ood_pii as prepare
from scripts.ood_pii import OodNormalizationError, normalize
from tests.test_ood_pii import MODEL


SENSITIVE_TEXT = "private-value-DO-NOT-AUDIT"


def source_record(spans: list[dict[str, Any]]) -> dict[str, Any]:
    return {"source_text": SENSITIVE_TEXT, "language": "English", "privacy_mask": spans}


def valid_record() -> dict[str, Any]:
    # Inconsistent value metadata is accepted; no source value enters the audit.
    return source_record([{"start": 0, "end": 7, "label": "EMAIL", "value": "different"}])


def invalid_record() -> dict[str, Any]:
    return source_record(
        [
            {"start": 0, "end": 7, "label": "EMAIL", "value": SENSITIVE_TEXT[:7]},
            {"start": 8, "end": len(SENSITIVE_TEXT) + 1, "label": "TIME"},
        ]
    )


@pytest.fixture
def stub_sources(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    rows = [valid_record(), invalid_record(), valid_record()]
    monkeypatch.setattr(prepare, "load_dataset", lambda *args, **kwargs: rows)
    monkeypatch.setattr(prepare, "load_label_map", lambda: MODEL)
    monkeypatch.setattr(prepare, "load_ontology", lambda: {"email": "EMAIL_ADDRESS"})
    return rows


def arguments(output_dir: Path, *, skip_invalid: bool = False) -> list[str]:
    result = [
        "--dataset", "ai4privacy", "--split", "validation", "--output-dir", str(output_dir)
    ]
    if skip_invalid:
        result.append("--skip-invalid")
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def assert_manifest(
    directory: Path, source_count: int, normalized_count: int, skipped_count: int,
    *, skip_invalid: bool,
) -> None:
    output = directory / "ai4privacy_validation.jsonl"
    audit = directory / "ai4privacy_validation.audit.jsonl"
    metadata = json.loads(
        (directory / "ai4privacy_validation.metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["source_row_count"] == source_count
    assert metadata["normalized_row_count"] == metadata["row_count"] == normalized_count
    assert metadata["skipped_row_count"] == skipped_count
    assert source_count == normalized_count + skipped_count
    assert metadata["normalized_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert metadata["audit_sha256"] == (
        hashlib.sha256(audit.read_bytes()).hexdigest() if skip_invalid else None
    )
    assert metadata["normalization_schema"] == "privacy-gateway/ood-pii-normalized/v1"


def test_default_mode_remains_fail_closed_without_an_audit(
    tmp_path: Path, stub_sources: list[dict[str, Any]], capsys: pytest.CaptureFixture[str],
) -> None:
    assert prepare.main(arguments(tmp_path)) == 2
    assert read_jsonl(tmp_path / "ai4privacy_validation.jsonl") == [
        normalize("ai4privacy", stub_sources[0], 0, "validation", MODEL)
    ]
    assert not (tmp_path / "ai4privacy_validation.audit.jsonl").exists()
    assert not (tmp_path / "ai4privacy_validation.metadata.json").exists()
    error = capsys.readouterr().err
    assert error == (
        "error: normalization failed dataset=ai4privacy split=validation "
        "example_index=1: source annotation offsets are outside text\n"
    )
    assert SENSITIVE_TEXT not in error


def test_skip_invalid_discards_whole_examples_and_keeps_source_indices(
    tmp_path: Path, stub_sources: list[dict[str, Any]],
) -> None:
    assert prepare.main(arguments(tmp_path, skip_invalid=True)) == 0
    records = read_jsonl(tmp_path / "ai4privacy_validation.jsonl")
    assert records == [
        normalize("ai4privacy", stub_sources[index], index, "validation", MODEL)
        for index in (0, 2)
    ]
    audit_path = tmp_path / "ai4privacy_validation.audit.jsonl"
    assert read_jsonl(audit_path) == [
        {"example_index": 1, "error": "source annotation offsets are outside text"}
    ]
    assert SENSITIVE_TEXT not in audit_path.read_text(encoding="utf-8")
    assert_manifest(tmp_path, 3, 2, 1, skip_invalid=True)


@pytest.mark.parametrize(
    ("bad_span", "reason"),
    [
        ({"start": -1, "end": 2, "label": "EMAIL"}, "source annotation offsets are outside text"),
        ({"start": 1, "end": 1, "label": "EMAIL"}, "source annotation offsets are outside text"),
        ({"start": 2, "end": 1, "label": "EMAIL"}, "source annotation offsets are outside text"),
        ({"start": 0, "end": 999, "label": "EMAIL"}, "source annotation offsets are outside text"),
        ({"start": True, "end": 2, "label": "EMAIL"}, "source annotation offsets are not integers"),
        ({"start": 0, "end": False, "label": "EMAIL"}, "source annotation offsets are not integers"),
        ({"start": 0.0, "end": 2, "label": "EMAIL"}, "source annotation offsets are not integers"),
        ({"start": "0", "end": 2, "label": "EMAIL"}, "source annotation offsets are not integers"),
        ({"start": 0, "label": "EMAIL"}, "source annotation offsets are not integers"),
        ({"start": 0, "end": 2}, "source annotation label is missing"),
        ({"start": 0, "end": 2, "label": ""}, "source annotation label is missing"),
        ({"start": 0, "end": 2, "label": 42}, "source annotation label is missing"),
    ],
)
def test_any_invalid_annotation_discards_even_preceding_valid_spans(
    tmp_path: Path, stub_sources: list[dict[str, Any]], bad_span: dict[str, Any], reason: str,
) -> None:
    stub_sources[:] = [
        source_record([{"start": 0, "end": 1, "label": "EMAIL"}, bad_span]),
        valid_record(),
    ]
    assert prepare.main(arguments(tmp_path, skip_invalid=True)) == 0
    records = read_jsonl(tmp_path / "ai4privacy_validation.jsonl")
    assert records == [normalize("ai4privacy", stub_sources[1], 1, "validation", MODEL)]
    assert read_jsonl(tmp_path / "ai4privacy_validation.audit.jsonl") == [
        {"example_index": 0, "error": reason}
    ]
    assert_manifest(tmp_path, 2, 1, 1, skip_invalid=True)


def test_multiple_invalid_examples_have_one_audit_entry_each(
    tmp_path: Path, stub_sources: list[dict[str, Any]],
) -> None:
    stub_sources[:] = [invalid_record(), invalid_record()]
    assert prepare.main(arguments(tmp_path, skip_invalid=True)) == 0
    assert (tmp_path / "ai4privacy_validation.jsonl").read_bytes() == b""
    assert read_jsonl(tmp_path / "ai4privacy_validation.audit.jsonl") == [
        {"example_index": index, "error": "source annotation offsets are outside text"}
        for index in (0, 1)
    ]
    assert_manifest(tmp_path, 2, 0, 2, skip_invalid=True)


@pytest.mark.parametrize("skip_invalid", [False, True])
def test_all_valid_input_preserves_the_record_schema(
    tmp_path: Path, stub_sources: list[dict[str, Any]], skip_invalid: bool,
) -> None:
    stub_sources[:] = [valid_record(), valid_record()]
    assert prepare.main(arguments(tmp_path, skip_invalid=skip_invalid)) == 0
    assert read_jsonl(tmp_path / "ai4privacy_validation.jsonl") == [
        normalize("ai4privacy", row, index, "validation", MODEL)
        for index, row in enumerate(stub_sources)
    ]
    audit_path = tmp_path / "ai4privacy_validation.audit.jsonl"
    if skip_invalid:
        assert audit_path.read_bytes() == b""
    else:
        assert not audit_path.exists()
    assert_manifest(tmp_path, 2, 2, 0, skip_invalid=skip_invalid)


def test_audit_redacts_unrecognized_error_details(
    tmp_path: Path, stub_sources: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_sources[:] = [valid_record()]

    def unsafe_error(*args: Any, **kwargs: Any) -> None:
        raise OodNormalizationError(f"annotation contains {SENSITIVE_TEXT}")

    monkeypatch.setattr(prepare, "normalize", unsafe_error)
    assert prepare.main(arguments(tmp_path, skip_invalid=True)) == 0
    audit_path = tmp_path / "ai4privacy_validation.audit.jsonl"
    assert read_jsonl(audit_path) == [
        {"example_index": 0, "error": "normalization error (details redacted)"}
    ]
    assert SENSITIVE_TEXT not in audit_path.read_text(encoding="utf-8")
    assert_manifest(tmp_path, 1, 0, 1, skip_invalid=True)


def test_skip_invalid_does_not_swallow_unexpected_errors(
    tmp_path: Path, stub_sources: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_error(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(prepare, "normalize", unexpected_error)
    with pytest.raises(RuntimeError, match="unexpected failure"):
        prepare.main(arguments(tmp_path, skip_invalid=True))
    assert not (tmp_path / "ai4privacy_validation.metadata.json").exists()


@pytest.mark.parametrize("suffix", [".jsonl", ".metadata.json", ".audit.jsonl"])
def test_skip_invalid_never_overwrites_existing_files(
    tmp_path: Path, stub_sources: list[dict[str, Any]], suffix: str,
) -> None:
    existing = tmp_path / f"ai4privacy_validation{suffix}"
    existing.write_text("existing file\n", encoding="utf-8")
    assert prepare.main(arguments(tmp_path, skip_invalid=True)) == 2
    assert existing.read_text(encoding="utf-8") == "existing file\n"
    assert list(tmp_path.iterdir()) == [existing]