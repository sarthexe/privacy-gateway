"""Regression tests for the GLiNER-PII benchmark adapter (mocked model output only)."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from scripts.batch_evaluation import Batch, score_batch
from scripts.evaluate_gliner import (
    AlignedExample,
    classify_false_negative,
    classify_false_positive,
    coverage_section,
    iter_aligned,
    score_examples,
)
from scripts.evaluation_checkpoint import EvaluationState
from scripts.gliner_evaluation import (
    UNMAPPED,
    GlinerLabelMap,
    GlinerOutputError,
    GlinerPredictor,
    InferenceConfig,
    RawPrediction,
    apply_threshold,
    canonical_order,
    deduplicate,
    load_label_map,
    merge_chunk_predictions,
    overlapping_pairs,
    parse_model_output,
    plan_chunks,
    to_gateway,
    unreachable_reasons,
    whitespace_words,
)
from scripts.gliner_predictions import (
    PredictionRecord,
    PredictionsError,
    PredictionWriter,
    decode_record,
    encode_record,
    iter_records,
    read_header,
    select_dev_indices,
)
from scripts.nemotron_pii import REPOSITORY_ROOT, load_ontology
from scripts.presidio_evaluation import ExactMatch, harness_strict_metrics
from scripts.select_gliner_threshold import select

ONTOLOGY = {
    "first_name": "PERSON",
    "last_name": "PERSON",
    "email": "EMAIL_ADDRESS",
    "date": "DATE",
    "time": "TIME",
    "company_name": "ORGANIZATION",
    "age": UNMAPPED,
    "occupation": UNMAPPED,
}
LABEL_MAP = GlinerLabelMap(
    model_id="nvidia/gliner-PII",
    model_revision="test",
    labels=tuple(ONTOLOGY),
    mapping=dict(ONTOLOGY),
    deviations={},
)
TEXT = "Ann Lee mailed ann@x.io at noon"
SENSITIVE = ["Ann", "Lee", "ann@x.io", "noon"]


def entity(label: str, start: int, end: int, score: float = 0.9) -> dict[str, Any]:
    # "text" mimics GLiNER's own field; the adapter must never propagate it.
    return {"label": label, "start": start, "end": end, "score": score, "text": TEXT[start:end]}


class FakeModel:
    """Returns canned per-text outputs keyed by the exact input string."""

    def __init__(self, outputs: dict[str, list[dict[str, Any]]]) -> None:
        self.outputs = outputs
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def inference(
        self,
        texts: list[str],
        labels: list[str],
        flat_ner: bool = True,
        threshold: float = 0.5,
        multi_label: bool = False,
        batch_size: int = 8,
    ) -> list[list[dict[str, Any]]]:
        kwargs = {
            "flat_ner": flat_ner,
            "threshold": threshold,
            "multi_label": multi_label,
            "batch_size": batch_size,
        }
        self.calls.append((list(texts), kwargs))
        return [list(self.outputs.get(text, [])) for text in texts]


def splitter(text: str) -> list[tuple[str, int, int]]:
    return [(text[s:e], s, e) for s, e in whitespace_words(text)]


# --- Span conversion and type mapping -----------------------------------------------------


def test_span_conversion_keeps_offsets_and_drops_entity_text() -> None:
    predictions = parse_model_output([entity("first_name", 0, 3)], len(TEXT), ONTOLOGY)
    assert predictions == [RawPrediction("first_name", 0, 3, 0.9)]
    assert not any(value in repr(predictions) for value in SENSITIVE)


def test_entity_type_mapping_matches_ground_truth_normalization() -> None:
    predictions = [
        RawPrediction("first_name", 0, 3, 0.9),
        RawPrediction("email", 15, 23, 0.9),
        RawPrediction("time", 27, 31, 0.9),
        RawPrediction("company_name", 4, 7, 0.9),
    ]
    gateway = to_gateway(predictions, LABEL_MAP)
    # DATE/TIME use the same evaluation_type mapping as the ground truth.
    assert gateway.kept == [
        ExactMatch("PERSON", 0, 3),
        ExactMatch("ORGANIZATION", 4, 7),
        ExactMatch("EMAIL_ADDRESS", 15, 23),
        ExactMatch("DATE_TIME", 27, 31),
    ]
    assert gateway.removed_by_label == Counter()


def test_unmapped_labels_are_counted_not_scored_or_silently_dropped() -> None:
    gateway = to_gateway(
        [RawPrediction("age", 0, 3, 0.9), RawPrediction("occupation", 4, 7, 0.8)], LABEL_MAP
    )
    assert gateway.kept == []
    assert gateway.removed_by_label == Counter({"age": 1, "occupation": 1})


def test_label_outside_prompt_vocabulary_is_rejected() -> None:
    with pytest.raises(GlinerOutputError, match="outside the prompt vocabulary"):
        parse_model_output([entity("crypto_wallet", 0, 3)], len(TEXT), ONTOLOGY)


def test_repository_label_map_agrees_with_ground_truth_ontology() -> None:
    label_map = load_label_map()
    ontology = load_ontology()
    assert set(label_map.labels) == set(ontology)
    assert all(label_map.gateway_type(label) == ontology[label] for label in ontology)
    assert label_map.deviations == {}


def test_label_map_rejects_undocumented_divergence(tmp_path: Path) -> None:
    config = tmp_path / "map.yaml"
    config.write_text(
        "model_id: m\nmodel_revision: r\ndeviations: {}\nlabels:\n  age: PERSON\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="document it under deviations"):
        load_label_map(config, ontology={"age": UNMAPPED, "first_name": "PERSON"})
    config.write_text(
        "model_id: m\nmodel_revision: r\ndeviations:\n  age: reviewed reason\n"
        "labels:\n  age: PERSON\n",
        encoding="utf-8",
    )
    loaded = load_label_map(config, ontology={"age": UNMAPPED, "first_name": "PERSON"})
    assert loaded.gateway_type("age") == "PERSON"


def test_label_map_rejects_unknown_gateway_type(tmp_path: Path) -> None:
    config = tmp_path / "map.yaml"
    config.write_text(
        "model_id: m\nmodel_revision: r\nlabels:\n  age: NOT_A_TYPE\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unknown gateway type"):
        load_label_map(config, ontology={"age": UNMAPPED})


# --- Malformed output ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "output",
    [
        "not a list",
        ["not an object"],
        [{"label": "email", "start": 0, "score": 0.5}],
        [{"label": "email", "start": True, "end": 3, "score": 0.5}],
        [{"label": "email", "start": 0, "end": 3.0, "score": 0.5}],
        [{"label": "email", "start": 3, "end": 3, "score": 0.5}],
        [{"label": "email", "start": 0, "end": 999, "score": 0.5}],
        [{"label": "email", "start": -1, "end": 3, "score": 0.5}],
        [{"label": "email", "start": 0, "end": 3, "score": "high"}],
        [{"label": "email", "start": 0, "end": 3, "score": float("nan")}],
        [{"label": "email", "start": 0, "end": 3, "score": 1.5}],
        [{"label": 7, "start": 0, "end": 3, "score": 0.5}],
    ],
)
def test_malformed_model_output_raises_without_echoing_values(output: object) -> None:
    with pytest.raises(GlinerOutputError) as raised:
        parse_model_output(output, len(TEXT), ONTOLOGY)
    assert not any(value in str(raised.value) for value in SENSITIVE)


def test_predictor_rejects_output_count_mismatch() -> None:
    class ShortModel(FakeModel):
        def inference(
            self,
            texts: list[str],
            labels: list[str],
            flat_ner: bool = True,
            threshold: float = 0.5,
            multi_label: bool = False,
            batch_size: int = 8,
        ) -> list[list[dict[str, Any]]]:
            return []

    predictor = GlinerPredictor(ShortModel({}), LABEL_MAP, splitter, InferenceConfig(0.3, 2, 50, 5))
    with pytest.raises(GlinerOutputError, match="number of outputs"):
        predictor.predict([TEXT])


# --- Duplicates, overlaps, determinism ----------------------------------------------------


def test_duplicate_spans_collapse_to_best_score_and_are_counted() -> None:
    unique, removed = deduplicate(
        [
            RawPrediction("email", 15, 23, 0.4),
            RawPrediction("email", 15, 23, 0.8),
            RawPrediction("first_name", 0, 3, 0.9),
        ]
    )
    assert unique == [RawPrediction("first_name", 0, 3, 0.9), RawPrediction("email", 15, 23, 0.8)]
    assert removed == 1


def test_same_span_with_different_labels_is_not_a_duplicate() -> None:
    unique, removed = deduplicate(
        [RawPrediction("first_name", 0, 3, 0.9), RawPrediction("company_name", 0, 3, 0.5)]
    )
    assert len(unique) == 2 and removed == 0


def test_overlapping_and_nested_spans_are_preserved_and_counted() -> None:
    predictions = [
        RawPrediction("first_name", 0, 7, 0.9),
        RawPrediction("last_name", 4, 7, 0.8),  # nested
        RawPrediction("company_name", 6, 10, 0.7),  # crossing
        RawPrediction("email", 15, 23, 0.9),  # disjoint
        RawPrediction("date", 23, 26, 0.9),  # adjacent (half-open: no overlap)
    ]
    assert overlapping_pairs(predictions) == 3
    gateway = to_gateway(predictions, LABEL_MAP)
    assert len(gateway.kept) == 5  # the adapter never resolves overlaps itself


def test_normalization_is_deterministic_regardless_of_emission_order() -> None:
    predictions = [
        RawPrediction("email", 15, 23, 0.9),
        RawPrediction("first_name", 0, 3, 0.9),
        RawPrediction("last_name", 4, 7, 0.8),
    ]
    assert canonical_order(predictions) == canonical_order(reversed(predictions))
    first, _ = deduplicate(predictions)
    second, _ = deduplicate(list(reversed(predictions)))
    assert first == second
    assert to_gateway(first, LABEL_MAP) == to_gateway(second, LABEL_MAP)


def test_threshold_filter_uses_the_decoders_strict_comparison() -> None:
    predictions = [RawPrediction("email", 0, 3, 0.5), RawPrediction("email", 4, 7, 0.51)]
    assert apply_threshold(predictions, 0.5) == [predictions[1]]


def test_predictor_restores_input_order_after_length_sorted_batching() -> None:
    short, long = "Ann", "Lee mailed ann@x.io"
    model = FakeModel(
        {
            short: [{"label": "first_name", "start": 0, "end": 3, "score": 0.9}],
            long: [{"label": "email", "start": 11, "end": 19, "score": 0.9}],
        }
    )
    predictor = GlinerPredictor(model, LABEL_MAP, splitter, InferenceConfig(0.3, 2, 50, 5))
    results = predictor.predict([short, long])
    assert model.calls[0][0] == [long, short]  # longest first to minimise padding
    assert model.calls[0][1] == {
        "flat_ner": True,
        "threshold": 0.3,
        "multi_label": False,
        "batch_size": 2,
    }
    assert [r.predictions for r in results] == [
        [RawPrediction("first_name", 0, 3, 0.9)],
        [RawPrediction("email", 11, 19, 0.9)],
    ]


# --- Long-text chunking -------------------------------------------------------------------


def test_short_text_is_a_single_unchanged_chunk() -> None:
    words = whitespace_words(TEXT)
    [chunk] = plan_chunks(words, len(TEXT), max_words=50, margin=5)
    assert (chunk.char_start, chunk.char_end, chunk.own_start, chunk.own_end) == (
        0,
        len(TEXT),
        0,
        len(TEXT),
    )


@pytest.mark.parametrize("word_count", [21, 29, 30, 47, 100])
def test_chunk_owned_regions_partition_text_and_contain_every_short_entity(
    word_count: int,
) -> None:
    text = " ".join(f"w{i}" for i in range(word_count))
    words = whitespace_words(text)
    max_words, margin = 20, 4
    chunks = plan_chunks(words, len(text), max_words, margin)
    assert all(chunk.word_count <= max_words for chunk in chunks)
    assert chunks[0].own_start == 0 and chunks[-1].own_end == len(text)
    assert all(a.own_end == b.own_start for a, b in zip(chunks, chunks[1:], strict=False))
    for first in range(word_count):
        for width in range(1, margin + 1):
            last = first + width - 1
            if last >= word_count:
                continue
            start, end = words[first][0], words[last][1]
            owners = [c for c in chunks if c.own_start <= start < c.own_end]
            assert len(owners) == 1
            assert owners[0].char_start <= start and end <= owners[0].char_end


def test_chunked_predictions_are_shifted_and_owned_once() -> None:
    text = " ".join(f"w{i}" for i in range(30))
    words = whitespace_words(text)
    chunks = plan_chunks(words, len(text), max_words=20, margin=4)
    assert len(chunks) == 2
    target = words[15]  # inside the overlap: both windows see it
    per_chunk = [
        [RawPrediction("age", target[0] - c.char_start, target[1] - c.char_start, 0.9)]
        for c in chunks
    ]
    merged = merge_chunk_predictions(chunks, per_chunk)
    assert merged == [RawPrediction("age", target[0], target[1], 0.9)]


def test_predictor_chunks_long_texts_through_the_model() -> None:
    text = " ".join(f"w{i}" for i in range(30))
    words = whitespace_words(text)
    config = InferenceConfig(0.3, 4, max_words=20, margin=4)
    chunks = plan_chunks(words, len(text), 20, 4)
    outputs: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        chunk_text = text[chunk.char_start : chunk.char_end]
        # Every window reports the last word it sees as an "age".
        last = whitespace_words(chunk_text)[-1]
        outputs[chunk_text] = [{"label": "age", "start": last[0], "end": last[1], "score": 0.9}]
    result = GlinerPredictor(FakeModel(outputs), LABEL_MAP, splitter, config).predict([text])[0]
    assert result.chunks == 2
    # The first window's tail is in the second window's owned region, so it is dropped.
    assert result.predictions == [RawPrediction("age", words[-1][0], words[-1][1], 0.9)]


def test_chunk_prediction_beyond_its_chunk_is_rejected() -> None:
    text = " ".join(f"w{i}" for i in range(30))
    chunks = plan_chunks(whitespace_words(text), len(text), 20, 4)
    bad = [[RawPrediction("age", 0, 10_000, 0.9)], []]
    with pytest.raises(GlinerOutputError):
        merge_chunk_predictions(chunks, bad)


def test_structurally_unreachable_ground_truth() -> None:
    words = whitespace_words("a1 b2 c3 d4")
    assert unreachable_reasons(words, [(0, 2), (1, 2), (0, 11)], max_width=3) == [
        None,
        "misaligned_word_boundary",
        "wider_than_max_width",
    ]


def test_scorer_word_splitter_matches_gliner() -> None:
    gliner_tokenizer = pytest.importorskip("gliner.data_processing.tokenizer")
    sample = "Dr. O'Neil-Smith (id_42) e-mail: a.b@c.io; tel +1-555-0100\tok…"
    library = [(s, e) for _, s, e in gliner_tokenizer.WhitespaceTokenSplitter()(sample)]
    assert library == whitespace_words(sample)


# --- Exact evaluator compatibility --------------------------------------------------------


def test_adapter_output_scores_through_existing_strict_harness() -> None:
    truth = [
        ExactMatch("PERSON", 0, 3),
        ExactMatch("PERSON", 4, 7),
        ExactMatch("DATE_TIME", 27, 31),
    ]
    raw = [
        RawPrediction("first_name", 0, 3, 0.9),  # exact
        RawPrediction("last_name", 4, 6, 0.9),  # boundary error
        RawPrediction("time", 27, 31, 0.9),  # exact after TIME -> DATE_TIME
        RawPrediction("age", 8, 14, 0.9),  # filtered: never scored
    ]
    predicted = to_gateway(raw, LABEL_MAP).kept
    state = EvaluationState()
    batch = Batch([(0, b"")], 0, True)
    score_batch(
        state,
        batch,
        [
            (
                tuple((t.entity_type, t.start, t.end) for t in truth),
                tuple((p.entity_type, p.start, p.end) for p in predicted),
            )
        ],
    )
    assert state.aggregate() == {"tp": 2, "fp": 1, "fn": 1}
    assert state.harness_exact_matches == 2  # nervaluate strict agrees
    metrics = harness_strict_metrics([truth], [predicted], ["DATE_TIME", "PERSON"])
    assert metrics["PERSON"]["correct"] == 1


def test_error_taxonomy_categories() -> None:
    truth = [ExactMatch("PERSON", 0, 3)]
    assert classify_false_negative(ExactMatch(UNMAPPED, 0, 3), [], []) == (
        "unmapped_ground_truth_label"
    )
    assert classify_false_negative(truth[0], [ExactMatch("ORGANIZATION", 0, 3)], []) == (
        "wrong_entity_type"
    )
    assert classify_false_negative(truth[0], [ExactMatch("PERSON", 0, 2)], []) == (
        "wrong_boundaries"
    )
    assert classify_false_negative(truth[0], [ExactMatch("ORGANIZATION", 1, 5)], []) == (
        "overlapping_prediction"
    )
    removed = [RawPrediction("age", 0, 3, 0.9)]
    assert classify_false_negative(truth[0], [], removed) == "unmapped_prediction_label"
    assert classify_false_negative(truth[0], [], []) == "missed_entity"
    assert classify_false_positive(ExactMatch("ORGANIZATION", 0, 3), truth) == (
        "wrong_entity_type",
        True,
    )
    assert classify_false_positive(ExactMatch("PERSON", 0, 5), truth) == ("wrong_boundaries", True)
    assert classify_false_positive(ExactMatch("ORGANIZATION", 2, 6), truth) == (
        "overlapping_prediction",
        False,
    )
    assert classify_false_positive(ExactMatch("PERSON", 10, 12), truth) == (
        "false_positive",
        False,
    )


def _example(record: PredictionRecord) -> AlignedExample:
    return AlignedExample(
        example_index=record.example_index,
        uid="u0",
        document_format="unstructured",
        text=TEXT,
        truth=[
            ExactMatch("PERSON", 0, 3),
            ExactMatch("PERSON", 4, 7),
            ExactMatch("EMAIL_ADDRESS", 15, 23),
            ExactMatch(UNMAPPED, 27, 31),
        ],
        native_truth=[
            ("first_name", 0, 3),
            ("last_name", 4, 7),
            ("email", 15, 23),
            ("age", 27, 31),
        ],
        record=record,
    )


def test_score_examples_separates_gateway_and_native_views() -> None:
    record = PredictionRecord(
        0,
        1,
        0,
        (
            RawPrediction("first_name", 0, 3, 0.95),
            RawPrediction("last_name", 4, 7, 0.25),  # below the scoring threshold
            RawPrediction("email", 15, 23, 0.9),
            RawPrediction("age", 27, 31, 0.9),  # correct natively, filtered in gateway view
        ),
    )
    score = score_examples(iter([_example(record)]), LABEL_MAP, threshold=0.3, max_width=12)
    assert score.state.aggregate() == {"tp": 2, "fp": 0, "fn": 2}
    assert score.fn_categories == Counter({"missed_entity": 1, "unmapped_ground_truth_label": 1})
    coverage = coverage_section(score, LABEL_MAP)
    assert coverage["predictions_removed_by_ontology_filtering"] == 1
    assert coverage["removed_predictions_that_exactly_match_native_ground_truth"] == 1
    assert coverage["unmapped_ground_truth"] == {"total": 1, "exact_native_match": 1}
    assert coverage["native_label_overall"]["tp"] == 3
    assert coverage["native_label_overall"]["fn"] == 1
    # Samples carry offsets and identifiers only.
    assert not any(value in json.dumps(score.samples) for value in SENSITIVE)


def test_aligned_reader_rejects_mismatched_raw_records(tmp_path: Path) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    normalized = tmp_path / "test.jsonl"
    normalized.write_text(
        json.dumps({"text": TEXT, "entities": [{"type": "PERSON", "start": 0, "end": 3}]}) + "\n",
        encoding="utf-8",
    )
    raw = tmp_path / "raw.parquet"

    def write_raw(label: str) -> None:
        spans = str([{"start": 0, "end": 3, "label": label}])
        table = pa.table(
            {"uid": ["u0"], "text": [TEXT], "spans": [spans], "document_format": ["unstructured"]}
        )
        pq.write_table(table, raw)

    record = PredictionRecord(0, 1, 0, ())
    write_raw("first_name")
    [example] = list(iter_aligned(normalized, raw, iter([record]), ONTOLOGY))
    assert example.native_truth == [("first_name", 0, 3)] and example.uid == "u0"
    write_raw("email")  # same offsets, but maps to a different gateway type
    with pytest.raises(RuntimeError, match="not aligned"):
        list(iter_aligned(normalized, raw, iter([record]), ONTOLOGY))


# --- Predictions store and threshold selection --------------------------------------------


def test_prediction_records_round_trip_without_text() -> None:
    record = PredictionRecord(3, 2, 0, (RawPrediction("email", 15, 23, 0.875),))
    encoded = encode_record(record)
    assert decode_record(encoded + "\n") == record
    assert not any(value in encoded for value in SENSITIVE)
    with pytest.raises(PredictionsError):
        decode_record('{"i": 0, "c": 1, "d": 0, "p": [["Ann Lee", 0, 3, 0.5]]}')


def test_prediction_writer_resumes_and_drops_a_torn_tail(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    identity = {"source": {"sha256": "abc"}, "inference": {"threshold": 0.3}}
    writer = PredictionWriter(path, identity, [0, 2, 5], resume=False, restart=False)
    writer.append([PredictionRecord(0, 1, 0, ())])
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"i": 2, "c"')  # interrupted mid-write
    resumed = PredictionWriter(path, identity, [0, 2, 5], resume=True, restart=False)
    assert resumed.completed == 1
    resumed.append([PredictionRecord(2, 1, 0, ()), PredictionRecord(5, 1, 0, ())])
    assert [r.example_index for r in iter_records(path)] == [0, 2, 5]
    assert read_header(path) == identity
    with pytest.raises(PredictionsError, match="does not match this run: inference"):
        PredictionWriter(
            path, {**identity, "inference": {"threshold": 0.5}}, [0], resume=True, restart=False
        )
    with pytest.raises(PredictionsError, match="already exists"):
        PredictionWriter(path, identity, [0], resume=False, restart=False)


def test_prediction_writer_refuses_repository_paths() -> None:
    with pytest.raises(ValueError, match="outside the repository"):
        PredictionWriter(REPOSITORY_ROOT / "p.jsonl", {}, [], resume=False, restart=False)


def test_dev_subset_is_deterministic_spread_and_sized() -> None:
    first = select_dev_indices(10_000, 200)
    assert first == select_dev_indices(10_000, 200)
    assert first == sorted(first) and len(set(first)) == 200
    assert first[-1] > 9_000  # not a prefix
    assert select_dev_indices(10_000, 200, seed="other") != first


def test_threshold_selection_rule_breaks_ties_toward_recall() -> None:
    results = {
        "0.3": {"gateway": {"f1": 0.80000}},
        "0.4": {"gateway": {"f1": 0.80004}},
        "0.5": {"gateway": {"f1": 0.79}},
    }
    assert select(results) == 0.3
    results["0.4"]["gateway"]["f1"] = 0.81
    assert select(results) == 0.4


# --- Privacy scanner ----------------------------------------------------------------------


def test_privacy_scanner_flags_planted_values_but_not_template_words(tmp_path: Path) -> None:
    from scripts.scan_privacy import build_index, scan_text, template_vocabulary

    split = tmp_path / "test.jsonl"
    text = "Jennifer Okafor works as a Worker; contact jen.okafor@example.org or 4411872."
    entities = [
        {"type": kind, "start": text.index(value), "end": text.index(value) + len(value)}
        for kind, value in [
            ("PERSON", "Jennifer Okafor"),
            ("UNMAPPED", "Worker"),
            ("EMAIL_ADDRESS", "jen.okafor@example.org"),
            ("UNMAPPED", "4411872"),
        ]
    ]
    split.write_text(json.dumps({"text": text, "entities": entities}) + "\n", encoding="utf-8")
    index, _ = build_index([split])
    vocabulary = template_vocabulary()

    def kinds(content: str) -> list[str]:
        return [hit.kind for hit in scan_text(content, index, vocabulary)]

    assert kinds('{"note": "Jennifer Okafor"}') == ["leak"]
    assert kinds("mail jen.okafor@example.org here") == ["leak"]
    # The text prefix itself, the name inside it, and a template word inside it.
    assert sorted(kinds(text[:40])) == ["leak", "leak", "template_vocabulary"]
    assert kinds('"workers": 8, "Worker"') == ["template_vocabulary"]
    assert kinds("counted 4411872 items") == ["numeric"]
    assert kinds("id 944118725 and XJennifer Okafory") == []  # not whole tokens


def test_privacy_scanner_template_phrases_pass_but_real_names_fail(tmp_path: Path) -> None:
    from scripts.scan_privacy import build_index, scan_text

    split = tmp_path / "test.jsonl"
    text = "an email from Okafor Nwosu"
    entities = [
        {"type": "UNMAPPED", "start": 0, "end": 8},  # synthetic value that is template text
        {"type": "PERSON", "start": 14, "end": 26},
    ]
    split.write_text(json.dumps({"text": text, "entities": entities}) + "\n", encoding="utf-8")
    index, _ = build_index([split])
    vocabulary = {"an", "email"}
    assert [h.kind for h in scan_text("e.g. an email address", index, vocabulary)] == [
        "template_vocabulary"
    ]
    assert [h.kind for h in scan_text("seen: Okafor Nwosu.", index, vocabulary)] == ["leak"]


# --- max_width CLI/config propagation -------------------------------------------------------


class _FakeConfig:
    def __init__(self, max_width: int) -> None:
        self.max_width = max_width
        self.max_len = 384
        self.model_name = "microsoft/deberta-v3-large"
        self.span_mode = "markerV0"
        self.words_splitter_type = "whitespace"


class _FakeGliner:
    """Stands in for ``gliner.GLiNER``; records ``from_pretrained`` keyword arguments."""

    calls: list[dict[str, Any]] = []
    ignore_override = False

    def __init__(self, max_width: int) -> None:
        self.config = _FakeConfig(max_width)

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> _FakeGliner:
        cls.calls.append({"model_id": model_id, **kwargs})
        return cls(12 if cls.ignore_override else kwargs.get("max_width") or 12)

    def eval(self) -> None:
        return None

    def parameters(self) -> list[Any]:
        return []


@pytest.fixture
def fake_gliner(monkeypatch: pytest.MonkeyPatch) -> type[_FakeGliner]:
    import sys
    import types

    module = types.ModuleType("gliner")
    module.GLiNER = _FakeGliner  # type: ignore[attr-defined]
    _FakeGliner.calls = []
    _FakeGliner.ignore_override = False
    monkeypatch.setitem(sys.modules, "gliner", module)
    return _FakeGliner


def test_max_width_cli_defaults_to_checkpoint_value() -> None:
    from scripts.gliner_predict import DEFAULT_MAX_WIDTH, parse_args

    base = ["--data-dir", "d", "--source", "test", "--output", "o", "--threshold", "0.7"]
    assert DEFAULT_MAX_WIDTH == 12
    assert parse_args(base).max_width == 12
    assert parse_args([*base, "--max-width", "24"]).max_width == 24


@pytest.mark.parametrize("width", [12, 16, 24, 32])
def test_max_width_is_passed_to_from_pretrained_with_pinned_revision(
    fake_gliner: type[_FakeGliner], width: int
) -> None:
    from scripts.gliner_predict import describe_model, load_model

    label_map = load_label_map()
    model, _ = load_model(label_map, "cpu", "fp32", width)
    [call] = fake_gliner.calls
    assert call["model_id"] == "nvidia/gliner-PII"
    assert call["revision"] == "bd23e8ef4425fd04e34c5204ab49ffaa706eae79"
    assert call["max_width"] == width
    assert describe_model(model, label_map, "fp32")["max_width"] == width


def test_load_model_default_width_is_twelve(fake_gliner: type[_FakeGliner]) -> None:
    from scripts.gliner_predict import load_model

    load_model(load_label_map(), "cpu", "fp32")
    assert fake_gliner.calls[0]["max_width"] == 12


def test_load_model_rejects_an_override_that_did_not_take_effect(
    fake_gliner: type[_FakeGliner],
) -> None:
    from scripts.gliner_predict import load_model

    fake_gliner.ignore_override = True
    with pytest.raises(RuntimeError, match="requested max_width=24"):
        load_model(load_label_map(), "cpu", "fp32", 24)


def test_max_width_sets_chunk_margin_and_is_recorded_in_run_identity(tmp_path: Path) -> None:
    from scripts.gliner_predict import build_identity, inference_config

    label_map = load_label_map()
    identities = {}
    for width in (12, 24):
        model_identity = {"max_len": 384, "max_width": width, "model_id": label_map.model_id}
        config = inference_config(model_identity, threshold=0.7, batch_size=1)
        assert config.margin == width and config.max_words == 384
        identities[width] = build_identity(
            {"sha256": "abc"}, label_map, model_identity, config, "cuda:0"
        )
        assert identities[width]["model"]["max_width"] == width
        assert identities[width]["inference"]["chunk_margin_words"] == width
    # Threshold, batch size and device are unaffected by the width.
    assert identities[12]["inference"]["threshold"] == identities[24]["inference"]["threshold"]
    # A file written at one width can never be resumed at another.
    path = tmp_path / "predictions.jsonl"
    PredictionWriter(path, identities[24], [0], resume=False, restart=False)
    assert read_header(path)["model"]["max_width"] == 24
    with pytest.raises(PredictionsError, match="does not match this run: .*model"):
        PredictionWriter(path, identities[12], [0], resume=True, restart=False)


def test_default_width_identity_matches_the_completed_baseline_header() -> None:
    """The refactor must not change the identity of default (width 12) runs."""
    from scripts.gliner_predict import build_identity, inference_config

    header_path = (
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "Temp"
        / "nemotron-pii-normalized-evaluation"
        / "gliner"
        / "test.jsonl"
    )
    if not header_path.is_file():
        pytest.skip("baseline predictions file is not available on this machine")
    saved = read_header(header_path)
    config = inference_config(saved["model"], threshold=0.7, batch_size=1)
    rebuilt = build_identity(saved["source"], load_label_map(), saved["model"], config, "cuda")
    assert rebuilt == saved
