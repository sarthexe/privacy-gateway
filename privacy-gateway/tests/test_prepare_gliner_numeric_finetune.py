from __future__ import annotations

import hashlib

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest

from scripts import prepare_gliner_numeric_finetune as prepare
from scripts.evaluation_checkpoint import canonical_sha256, sha256_file
from scripts.gliner_credit_card_experiment import luhn
from scripts.gliner_evaluation import load_label_map, whitespace_words
from scripts.gliner_numeric_common import (
    OTHER_NUMERIC,
    GlinerInputSizer,
    NumericConfig,
    NumericExperimentError,
    is_forbidden_split,
    load_config,
    validate_config,
)
from scripts.nemotron_pii import REPOSITORY_ROOT, load_ontology

CONFIG = load_config()
SECRET_MARKERS = ("Zephyrine", "Quillon", "4111 1111 1111 1111", "zq@example.test")


class FakeTokenizer:
    model_max_length = 10**30

    def __init__(self, pieces: dict[str, int] | None = None) -> None:
        self.pieces = pieces or {}

    def __call__(self, words: list[str], **kwargs: Any) -> dict[str, list[int]]:
        assert kwargs == {"is_split_into_words": True, "truncation": False}
        count = 2 + sum(self.pieces.get(word, 1) for word in words)
        return {"input_ids": list(range(count))}


class FakeProcessor:
    def __init__(self, pieces: dict[str, int] | None = None) -> None:
        self.transformer_tokenizer = FakeTokenizer(pieces)

    @staticmethod
    def prepare_inputs(
        texts: list[list[str]], labels: list[list[str]]
    ) -> tuple[list[list[str]], list[int]]:
        prompt = [part for label in labels[0] for part in ("<<ENT>>", label)]
        prompt.append("<<SEP>>")
        return [prompt + texts[0]], [len(prompt)]


def input_sizer(limit: int = 10_000, pieces: dict[str, int] | None = None) -> GlinerInputSizer:
    return GlinerInputSizer(
        FakeProcessor(pieces),
        limit,
        tokenizer_model_max_length=10**30,
        encoder_max_position_embeddings=limit,
    )


def config_with(**changes: Any) -> NumericConfig:
    raw = copy.deepcopy(dict(CONFIG.raw))
    for dotted, value in changes.items():
        target = raw
        *parents, leaf = dotted.split("__")
        for key in parents:
            target = target[key]
        target[leaf] = value
    validate_config(raw, CONFIG.label_map, load_ontology())
    return NumericConfig(raw=raw, sha256=canonical_sha256(raw), label_map=CONFIG.label_map)


def span(text: str, value: str, label: str) -> dict[str, Any]:
    start = text.index(value)
    return {"start": start, "end": start + len(value), "label": label}


def nemotron_rows(count: int) -> list[dict[str, Any]]:
    rows = []
    for index in range(count):
        if index % 3 == 0:
            text = f"Zephyrine Quillon paid with card 4111 1111 1111 1111 on order {index}."
            spans = [
                span(text, "Zephyrine", "first_name"),
                span(text, "Quillon", "last_name"),
                span(text, "4111 1111 1111 1111", "credit_debit_card"),
            ]
        elif index % 3 == 1:
            text = f"Write to zq@example.test about account 00{index}77123 today."
            spans = [
                span(text, "zq@example.test", "email"),
                span(text, f"00{index}77123", "account_number"),
            ]
        else:
            text = f"The meeting moved to the {index}th floor."
            spans = []
        rows.append({"uid": f"uid-{index}", "text": text, "spans": str(spans)})
    return rows


def source(rows: list[dict[str, Any]], total: int = 5_000) -> prepare.SourceSpec:
    return prepare.SourceSpec(
        name="nemotron_train",
        dataset="nemotron",
        split="train",
        revision="pinned",
        rows=lambda: rows,
        total_rows=total,
    )


def build(tmp_path: Path, name: str, rows: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    return prepare.build_dataset(
        kwargs.pop("config", CONFIG),
        [source(rows)],
        tmp_path / name,
        synthetic_counts=kwargs.pop("synthetic_counts", (40, 10)),
        input_sizer=kwargs.pop("input_sizer", input_sizer()),
        **kwargs,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="\n") as handle:
        return [json.loads(line) for line in handle]


# --- Alignment and format ----------------------------------------------------------------


def test_alignment_keeps_exact_offsets_and_gliner_word_spans() -> None:
    text = "Card:  4111-1111-1111-1111, holder Zephyrine."
    entities = [
        ("credit_debit_card", text.index("4111"), text.index(",")),
        ("first_name", text.index("Zephyrine"), text.index("Zephyrine") + 9),
    ]
    [window] = prepare.align_example(text, entities, 384, 24)
    assert window.text == text
    assert window.char_entities == tuple(sorted(entities, key=lambda item: item[1]))
    words = whitespace_words(text)
    assert window.tokens == tuple(text[start:end] for start, end in words)
    for (first, last, label), (char_label, start, end) in zip(
        window.ner, window.char_entities, strict=True
    ):
        assert label == char_label
        assert words[first][0] == start and words[last][1] == end


@pytest.mark.parametrize(
    ("entities", "reason"),
    [
        ([("email", 1, 5)], "misaligned_word_boundary"),
        ([("account_number", 0, -1)], "wider_than_max_width"),
        ([("account_number", 0, 3), ("ssn", 0, 3)], "conflicting_labels_same_span"),
    ],
)
def test_unrepresentable_spans_are_rejected_not_moved(
    entities: list[tuple[str, int, int]], reason: str
) -> None:
    text = "abc " + " ".join(["x"] * 60)
    entities = [(label, start, len(text) if end == -1 else end) for label, start, end in entities]
    with pytest.raises(prepare.Rejection) as caught:
        prepare.align_example(text, entities, 384, 24)
    assert caught.value.reason == reason


def test_windows_never_cut_an_entity_and_offsets_survive() -> None:
    words = [f"w{index}" for index in range(30)]
    text = " ".join(words)
    start = text.index("w8 ")
    end = text.index("w11") + 3
    [first, second, *rest] = prepare.align_example(text, [("unique_id", start, end)], 10, 5)
    assert len(first.tokens) == 8 and first.ner == ()
    assert second.ner == ((0, 3, "unique_id"),)
    _, local_start, local_end = second.char_entities[0]
    assert second.text[local_start:local_end] == text[start:end]
    assert all(len(window.tokens) <= 10 for window in (first, second, *rest))


def test_token_window_accepts_an_example_within_512_subwords() -> None:
    text = " ".join(f"word{index}" for index in range(300))
    windows = prepare.align_example(
        text,
        (),
        384,
        24,
        prompt_labels=["email"],
        input_sizer=input_sizer(512),
    )
    assert len(windows) == 1
    assert input_sizer(512).fits(windows[0].tokens, ["email"])


def test_token_window_divides_a_long_example() -> None:
    text = " ".join(f"word{index}" for index in range(600))
    sizer = input_sizer(512)
    windows = prepare.align_example(
        text,
        (),
        1_000,
        24,
        prompt_labels=["email"],
        input_sizer=sizer,
    )
    assert len(windows) == 2
    assert [len(window.tokens) for window in windows] == [507, 93]
    assert all(sizer.fits(window.tokens, ["email"]) for window in windows)


def test_token_window_accounts_for_words_with_multiple_subwords() -> None:
    text = " ".join(["multisub"] * 200)
    sizer = input_sizer(512, {"multisub": 3})
    windows = prepare.align_example(
        text,
        (),
        1_000,
        24,
        prompt_labels=["email"],
        input_sizer=sizer,
    )
    assert [len(window.tokens) for window in windows] == [169, 31]
    assert all(sizer.fits(window.tokens, ["email"]) for window in windows)


def test_token_boundary_moves_before_an_entity_instead_of_splitting_it() -> None:
    words = [f"w{index}" for index in range(12)]
    text = " ".join(words)
    start = text.index("w4")
    end = text.index("w6") + len("w6")
    windows = prepare.align_example(
        text,
        [("account_number", start, end)],
        100,
        24,
        prompt_labels=["account_number"],
        input_sizer=input_sizer(10),
    )
    assert (windows[0].word_start, windows[0].word_end) == (0, 4)
    assert windows[0].ner == ()
    assert windows[1].ner == ((0, 2, "account_number"),)


def test_token_windows_preserve_every_entity_label_and_exact_offset() -> None:
    words = [f"token{index}" for index in range(40)]
    text = " ".join(words)
    entities = [
        ("first_name", text.index("token3"), text.index("token3") + len("token3")),
        (
            "account_number",
            text.index("token18"),
            text.index("token20") + len("token20"),
        ),
        ("ipv4", text.index("token35"), text.index("token35") + len("token35")),
    ]
    windows = prepare.align_example(
        text,
        entities,
        100,
        24,
        prompt_labels=["first_name", "account_number", "ipv4"],
        input_sizer=input_sizer(24),
    )
    recovered = []
    words_with_offsets = whitespace_words(text)
    for window in windows:
        original_offset = words_with_offsets[window.word_start][0]
        recovered.extend(
            (label, original_offset + start, original_offset + end)
            for label, start, end in window.char_entities
        )
    assert sorted(recovered) == sorted(entities)


def test_synthetic_other_numeric_offset_survives_token_windowing() -> None:
    words = [f"context{index}" for index in range(80)]
    words[43] = "INV-000123"
    text = " ".join(words)
    start = text.index("INV-000123")
    entity = (OTHER_NUMERIC, start, start + len("INV-000123"))
    windows = prepare.align_example(
        text,
        [entity],
        384,
        24,
        prompt_labels=CONFIG.training_labels,
        input_sizer=input_sizer(130),
    )
    matches = [
        (window, item)
        for window in windows
        for item in window.char_entities
        if item[0] == OTHER_NUMERIC
    ]
    assert len(windows) > 1
    [(window, (label, local_start, local_end))] = matches
    assert label == OTHER_NUMERIC
    assert window.text[local_start:local_end] == "INV-000123"


def test_token_windowing_is_deterministic() -> None:
    text = " ".join(["multisub"] * 300)
    kwargs = {
        "prompt_labels": ["email", "account_number"],
        "input_sizer": input_sizer(64, {"multisub": 2}),
    }
    assert prepare.align_example(text, (), 384, 24, **kwargs) == prepare.align_example(
        text, (), 384, 24, **kwargs
    )


def test_unfit_entity_is_rejected_with_an_aggregate_reason(tmp_path: Path) -> None:
    row = {
        "uid": "unfit",
        "text": "huge",
        "spans": str([{"start": 0, "end": 4, "label": "account_number"}]),
    }
    manifest = build(
        tmp_path,
        "out",
        [row],
        synthetic_counts=(0, 0),
        input_sizer=input_sizer(150, {"huge": 100}),
    )
    assert manifest["sources"]["nemotron_train"]["rejected"] == {"entity_exceeds_subword_budget": 1}
    assert manifest["records"] == {"train": 0, "dev": 0, "by_kind": {}}


def test_gliner_record_shape_and_prompts() -> None:
    [window] = prepare.align_example("Order 12345 shipped", [(OTHER_NUMERIC, 6, 11)], 384, 24)
    record = prepare.gliner_record(
        window, CONFIG.training_labels, 1, "k", "synthetic", "synthetic", "g"
    )
    assert record["tokenized_text"] == ["Order", "12345", "shipped"]
    assert record["ner"] == [[1, 1, OTHER_NUMERIC]]
    assert sorted(record["ner_labels"]) == sorted(CONFIG.training_labels)
    assert record == prepare.gliner_record(
        window, CONFIG.training_labels, 1, "k", "synthetic", "synthetic", "g"
    )
    assert len(CONFIG.native_labels) == 55
    assert CONFIG.native_labels == load_label_map().labels


# --- Synthetic hard negatives ------------------------------------------------------------


def test_hard_negatives_are_deterministic_and_seed_dependent() -> None:
    settings = CONFIG.data["hard_negatives"]
    first = prepare.generate_hard_negatives(300, 7, "train", settings)
    assert first == prepare.generate_hard_negatives(300, 7, "train", settings)
    assert first != prepare.generate_hard_negatives(300, 8, "train", settings)
    assert first != prepare.generate_hard_negatives(300, 7, "dev", settings)


def test_hard_negatives_cover_required_shapes_and_never_label_cards() -> None:
    settings = CONFIG.data["hard_negatives"]
    examples = prepare.generate_hard_negatives(2_000, CONFIG.seed, "train", settings)
    families = {example.family for example in examples}
    assert {
        "card_shaped",
        "multi_span",
        "unlabeled_quantity",
        "contrast_reference",
        "contrast_native",
        "ip_lookalike",
        "date_lookalike",
        "postcode_lookalike",
        "reference_leading_zero",
        "reference_long",
        "reference_grouped",
        "reference_alnum",
    } <= families
    labels = {label for example in examples for label, _, _ in example.entities}
    assert labels == {OTHER_NUMERIC, "employee_id", "customer_id"}
    assert "credit_debit_card" not in labels
    for example in examples:
        windows = prepare.align_example(example.text, example.entities, 384, 24)
        assert len(windows) == 1 and len(windows[0].ner) == len(example.entities)
        if example.family == "multi_span":
            assert len(example.entities) >= 2
        if example.family == "unlabeled_quantity":
            assert example.entities == ()
    lowered = " ".join(example.text.lower() for example in examples)
    for context in settings["other_numeric_contexts"]:
        assert context in lowered


def test_luhn_never_decides_the_label() -> None:
    settings = CONFIG.data["hard_negatives"]
    cards = [
        example
        for example in prepare.generate_hard_negatives(2_000, 3, "train", settings)
        if example.family == "card_shaped"
    ]
    valid = [example for example in cards if example.card_shaped_luhn]
    invalid = [example for example in cards if example.card_shaped_luhn is False]
    assert valid and invalid
    for example in cards:
        [(label, start, end)] = example.entities
        digits = re.sub(r"\D", "", example.text[start:end])
        assert luhn(digits) is bool(example.card_shaped_luhn)
        assert label == OTHER_NUMERIC


def test_contrast_pairs_share_a_value_under_different_contexts() -> None:
    settings = CONFIG.data["hard_negatives"]
    examples = prepare.generate_hard_negatives(1_000, 5, "train", settings)
    pairs = [
        (left, right)
        for left, right in zip(examples, examples[1:], strict=False)
        if left.family == "contrast_reference" and right.family == "contrast_native"
    ]
    assert pairs
    for left, right in pairs:
        (_, a, b), (label, c, d) = left.entities[0], right.entities[0]
        assert left.text[a:b] == right.text[c:d]
        assert label in {"employee_id", "customer_id"}


def test_luhn_helpers_match_existing_checksum() -> None:
    for payload in ("411111111111111", "3714496353984", "123456789"):
        digits = payload + prepare.luhn_check_digit(payload)
        assert prepare.luhn_valid(digits) and luhn(digits)


# --- Splits ------------------------------------------------------------------------------


def test_grouped_split_is_deterministic_and_order_independent() -> None:
    keys = [[f"text:{i % 50}", f"id:{i % 7}"] for i in range(200)] + [["text:solo"]]
    reserved = [False] * len(keys)
    splits, groups = prepare.assign_splits(keys, reserved, 11, 0.3)
    order = list(reversed(range(len(keys))))
    rev_splits, rev_groups = prepare.assign_splits(
        [keys[i] for i in order], [reserved[i] for i in order], 11, 0.3
    )
    assert [splits[i] for i in order] == rev_splits
    assert [groups[i] for i in order] == rev_groups
    by_group: dict[str, set[str]] = {}
    for split, group in zip(splits, groups, strict=True):
        by_group.setdefault(group, set()).add(split)
    assert all(len(values) == 1 for values in by_group.values())


def test_shared_id_or_text_forces_same_split_and_reserved_propagates() -> None:
    keys = [["text:a", "id:1"], ["text:b", "id:1"], ["text:b"], ["text:c"]]
    splits, groups = prepare.assign_splits(keys, [False, False, True, False], 0, 0.5)
    assert len(set(groups[:3])) == 1
    assert splits[:3] == ["reserved"] * 3
    assert splits[3] in {"train", "dev"}


def test_text_key_folds_case_and_whitespace() -> None:
    assert prepare.text_key("Hello   World\n") == prepare.text_key("hello world")
    assert prepare.text_key("hello world") != prepare.text_key("hello worlds")


# --- Leakage guards ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dataset", "split"),
    [("ai4privacy", "validation"), ("gretel", "test"), ("argilla", "train"), ("nemotron", "test")],
)
def test_frozen_benchmark_splits_are_forbidden(dataset: str, split: str) -> None:
    assert is_forbidden_split(dataset, split)
    spec = prepare.SourceSpec("x", dataset, split, "r", lambda: [], 1)
    with pytest.raises(NumericExperimentError, match="frozen benchmark"):
        prepare.build_dataset(CONFIG, [spec], Path("/nonexistent-out"))


def test_config_cannot_point_a_source_at_a_benchmark_split() -> None:
    with pytest.raises(NumericExperimentError, match="frozen benchmark"):
        config_with(data__sources__ai4privacy_train={"enabled": False, "split": "validation"})
    with pytest.raises(NumericExperimentError, match="frozen benchmark"):
        config_with(data__sources__gretel_train={"enabled": False, "split": "test"})


def test_ood_adapter_refuses_benchmark_split() -> None:
    with pytest.raises(NumericExperimentError):
        prepare.ood_example("ai4privacy", "validation", {}, 0, CONFIG.label_map)


def test_ood_rows_with_unmapped_labels_are_dropped() -> None:
    record = {
        "source_text": "Ref 12345 for Zephyrine",
        "privacy_mask": [
            {"start": 4, "end": 9, "label": "NOT_A_KNOWN_LABEL"},
            {"start": 14, "end": 23, "label": "GIVENNAME"},
        ],
    }
    with pytest.raises(prepare.Rejection, match="unmapped"):
        prepare.ood_example("ai4privacy", "train", record, 0, CONFIG.label_map)


def test_ood_sources_require_benchmark_guard(tmp_path: Path) -> None:
    spec = prepare.SourceSpec("ai4privacy_train", "ai4privacy", "train", "r", lambda: [], 1)
    with pytest.raises(NumericExperimentError, match="benchmark-normalized"):
        prepare.build_dataset(CONFIG, [spec], tmp_path / "out")


def test_only_pinned_benchmark_files_are_accepted(tmp_path: Path) -> None:
    path = tmp_path / "bench.jsonl"
    path.write_text(json.dumps({"text": "x", "entities": []}) + "\n", encoding="utf-8")
    with pytest.raises(NumericExperimentError, match="pinned"):
        prepare.load_benchmark_text_keys([path])


def test_benchmark_texts_are_excluded_from_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = nemotron_rows(12)
    bench = tmp_path / "bench.jsonl"
    bench.write_text(
        json.dumps({"text": rows[1]["text"].upper(), "entities": []}) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(prepare, "OOD_BENCHMARK_SHA256", {"gretel": sha256_file(bench)})
    manifest = build(tmp_path, "out", rows, benchmark_paths=[bench])
    assert manifest["sources"]["nemotron_train"]["benchmark_overlap_excluded"] == 1
    texts = {
        " ".join(row["text"].split())
        for row in read_jsonl(tmp_path / "out" / "dev_normalized.jsonl")
    }
    assert " ".join(rows[1]["text"].split()) not in texts


def test_malformed_annotations_abort_the_build(tmp_path: Path) -> None:
    rows = nemotron_rows(3)
    rows[1]["spans"] = str([{"start": 5, "end": 2, "label": "email"}])
    with pytest.raises(NumericExperimentError, match="row 1"):
        build(tmp_path, "out", rows)


def test_output_inside_repository_is_refused() -> None:
    with pytest.raises(NumericExperimentError, match="outside the repository"):
        prepare.build_dataset(CONFIG, [source([])], REPOSITORY_ROOT / "tmp-dataset")


# --- End-to-end smoke build --------------------------------------------------------------


def test_build_is_deterministic_and_well_formed(tmp_path: Path) -> None:
    rows = nemotron_rows(60)
    first = build(tmp_path, "a", rows)
    second = build(tmp_path, "b", rows)
    assert first["files"] == second["files"]
    train = read_jsonl(tmp_path / "a" / "train.jsonl")
    dev = read_jsonl(tmp_path / "a" / "dev.jsonl")
    normalized = read_jsonl(tmp_path / "a" / "dev_normalized.jsonl")
    assert len(train) == first["records"]["train"] and len(dev) == len(normalized)
    for record in train + dev:
        assert set(record) == {"tokenized_text", "ner", "ner_labels", "source", "kind", "group"}
        for first_word, last_word, label in record["ner"]:
            assert 0 <= first_word <= last_word < len(record["tokenized_text"])
            assert label in record["ner_labels"]
        if record["kind"] == "synthetic":
            assert OTHER_NUMERIC in record["ner_labels"]
        else:
            assert sorted(record["ner_labels"]) == sorted(CONFIG.native_labels)
    for record, view in zip(dev, normalized, strict=True):
        assert [label for _, _, label in record["ner"]] == [
            item["native_label"] for item in view["native_entities"]
        ]
        for item in view["native_entities"]:
            assert (
                view["text"][item["start"] : item["end"]].strip()
                == view["text"][item["start"] : item["end"]]
            )


def neutral_rows(count: int, offset: int) -> list[dict[str, Any]]:
    return [
        {"uid": f"uid-n{index}", "text": f"Status note {index + offset} was filed.", "spans": "[]"}
        for index in range(count)
    ]


def test_focus_oversampling_keeps_every_natural_example(tmp_path: Path) -> None:
    rows = nemotron_rows(30) + neutral_rows(300, 1_000)
    manifest = build(tmp_path, "out", rows, synthetic_counts=(20, 5))
    kinds = manifest["records"]["by_kind"]
    balance = manifest["balancing"]
    assert kinds["train:natural"] == balance["natural_train_windows"]
    assert kinds.get("train:oversampled", 0) == balance["oversampled_copies"] > 0
    assert balance["oversampled_copies"] <= balance["focus_natural_train_windows"] * 2
    assert balance["focus_share"] <= CONFIG.data["balancing"]["max_focus_share"] + 1e-9
    assert "focus_share_above_maximum" not in manifest["warnings"]


def test_focus_heavy_sources_are_never_trimmed_only_flagged(tmp_path: Path) -> None:
    rows = [
        {
            "uid": f"c{index}",
            "text": f"Card 4111 1111 1111 {index:04d} on file.",
            "spans": str([{"start": 5, "end": 24, "label": "credit_debit_card"}]),
        }
        for index in range(40)
    ]
    manifest = build(tmp_path, "out", rows)
    balance = manifest["balancing"]
    assert balance["oversampled_copies"] == 0 and balance["synthetic_train_kept"] == 0
    assert manifest["records"]["by_kind"]["train:natural"] == balance["natural_train_windows"]
    assert "focus_share_above_maximum" in manifest["warnings"]


def test_balance_plan_respects_the_cap_and_is_deterministic() -> None:
    windows = [(f"w{index}", 1, 5) for index in range(100)]
    plan = prepare.balance_plan(100, 1_000, windows, [(1, 1)] * 50, 3, 0.4, seed=1)
    assert plan.synthetic_kept == 50
    assert sum(plan.copies.values()) == 200 and max(plan.copies.values()) == 2
    assert plan == prepare.balance_plan(100, 1_000, windows, [(1, 1)] * 50, 3, 0.4, seed=1)
    capped = prepare.balance_plan(100, 1_000, windows, [(1, 1)] * 50, 3, 0.15, seed=1)
    assert capped.share <= 0.15 and sum(capped.copies.values()) < 200
    trimmed = prepare.balance_plan(380, 1_000, windows, [(1, 1)] * 50, 3, 0.4, seed=1)
    assert trimmed.synthetic_kept < 50 and trimmed.share <= 0.4
    assert trimmed.focus_mentions >= 380


def test_manifest_and_logs_hold_no_text_or_values(tmp_path: Path) -> None:
    manifest = build(tmp_path, "out", nemotron_rows(30))
    serialized = json.dumps(manifest)
    for marker in SECRET_MARKERS:
        assert marker not in serialized
    synthetic = prepare.generate_hard_negatives(
        40, CONFIG.seed, "train", CONFIG.data["hard_negatives"]
    )
    for example in synthetic:
        for _, start, end in example.entities:
            assert example.text[start:end] not in serialized
    assert "uid-" not in serialized
    assert set(manifest["label_counts"]["train"]) <= set(CONFIG.training_labels)


def test_build_refuses_non_empty_output(tmp_path: Path) -> None:
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "stale.txt").write_text("x", encoding="utf-8")
    with pytest.raises(NumericExperimentError, match="not empty"):
        build(tmp_path, "out", nemotron_rows(3))


def test_threshold_selection_rows_are_reserved(tmp_path: Path) -> None:
    rows = nemotron_rows(60)
    reserved = config_with(data__reserve_threshold_selection_subset=True)
    manifest = build(tmp_path, "out", rows, config=reserved)
    assert manifest["split"]["examples"].get("reserved", 0) > 0
    unreserved = config_with(data__reserve_threshold_selection_subset=False)
    other = build(tmp_path, "other", rows, config=unreserved)
    assert "reserved" not in other["split"]["examples"]



def test_balance_plan_skips_oversized_copy_and_continues_balancing() -> None:
    seed = 1
    keys = [f"small-{index}" for index in range(50)] + ["large-window"]
    first_key = min(
        keys,
        key=lambda key: __import__("hashlib").sha256(
            f"{seed}|copy|{key}".encode()
        ).digest(),
    )
    small = [(key, 1, 5) for key in keys if key != first_key]
    large = (first_key, 100, 100)
    windows = [large, *small]

    plan = prepare.balance_plan(
        natural_focus=10,
        natural_total=100,
        focus_windows=windows,
        synthetic=[],
        factor=3,
        max_share=0.20,
        seed=seed,
    )

    assert first_key not in plan.copies
    assert sum(plan.copies.values()) == 100
    assert plan.share >= 0.15
    assert plan.share <= 0.20
