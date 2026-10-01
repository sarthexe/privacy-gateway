from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import gliner_ood_evaluate as evaluate


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\u0085"])
def test_unicode_separator_inside_json_string_is_not_a_record_boundary(
    tmp_path: Path, separator: str,
) -> None:
    record = {
        "example_index": 0,
        "text": f"first{separator}second",
        "entities": [{"type": "PERSON", "start": 6, "end": 12}],
    }
    path = tmp_path / "normalized.jsonl"
    payload = json.dumps(record, ensure_ascii=False) + "\n"
    path.write_text(payload, encoding="utf-8")

    # The separator is literal UTF-8 in the file, not a JSON escape.
    assert separator.encode("utf-8") in path.read_bytes()
    assert payload.count("\n") == 1
    assert len(payload.splitlines()) == 2
    rows = evaluate._load_normalized_records(path)
    assert rows == [record]
    text, entities = evaluate.validate_normalized_example(rows[0])
    assert text == record["text"]
    assert len(entities) == 1
    assert text[entities[0].start:entities[0].end] == "second"


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_physical_records_blank_lines_and_unterminated_final_record(
    tmp_path: Path, newline: str,
) -> None:
    records = [{"text": "first\u2028record"}, {"text": "second\u2029record"}]
    payload = newline.join(
        ["", json.dumps(records[0], ensure_ascii=False), " \t", json.dumps(records[1], ensure_ascii=False)]
    )
    path = tmp_path / "normalized.jsonl"
    path.write_bytes(payload.encode("utf-8"))
    assert evaluate._load_normalized_records(path) == records


@pytest.mark.parametrize("payload", ['not-json\n', '{"text":\n'])
def test_invalid_json_is_still_rejected(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "normalized.jsonl"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        evaluate._load_normalized_records(path)


def test_runner_parses_unicode_record_before_model_loading_without_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "normalized.jsonl"
    path.write_text(
        json.dumps({"text": "first\u2028second", "entities": []}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        evaluate, "load_label_map",
        lambda: SimpleNamespace(model_revision=evaluate.MODEL_REVISION),
    )

    class StopBeforeInference(Exception):
        pass

    def stop_model_loading(*args: Any, **kwargs: Any) -> None:
        raise StopBeforeInference

    monkeypatch.setattr(evaluate, "load_model", stop_model_loading)
    with pytest.raises(StopBeforeInference):
        evaluate.run(path, tmp_path / "reports", "cuda")
    assert not (tmp_path / "reports").exists()