"""Score stored GLiNER-PII predictions with the existing strict exact-span harness.

Streams, in lockstep and with per-record alignment checks, the normalized split
(ground truth used by the Presidio baseline), the raw Nemotron-PII parquet from
the Hugging Face cache (native labels, uid, document format), and the offset-only
predictions file. Gateway-ontology metrics go through ``score_batch`` /
``EvaluationState`` unchanged; native-label coverage and the error taxonomy are
additional, separately reported views. Nothing here imports torch or gliner.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.batch_evaluation import (
    Batch,
    RunResult,
    matching_identity,
    normalization_identity,
    score_batch,
)
from scripts.evaluate_presidio import build_report
from scripts.evaluation_checkpoint import EvaluationState, canonical_sha256, sha256_file
from scripts.gliner_evaluation import (
    DEFAULT_LABEL_MAP,
    UNMAPPED,
    GlinerLabelMap,
    RawPrediction,
    apply_threshold,
    load_label_map,
    overlapping_pairs,
    to_gateway,
    unreachable_reasons,
    whitespace_words,
)
from scripts.gliner_predictions import (
    PredictionRecord,
    iter_records,
    read_header,
    select_dev_indices,
)
from scripts.nemotron_pii import (
    DATASET_ID,
    is_within_repository,
    load_ontology,
    parse_spans,
)
from scripts.presidio_evaluation import (
    ExactMatch,
    evaluation_type,
    metric_rows,
    validate_normalized_example,
)

REPORT_DIRECTORY = Path(__file__).resolve().parents[1] / "reports" / "pii"
SCORE_BATCH = 1_000
SAMPLES_PER_CATEGORY = 5
RAW_COLUMNS = ["uid", "text", "spans", "document_format"]

FN_CATEGORIES = (
    "unmapped_ground_truth_label",
    "wrong_entity_type",
    "wrong_boundaries",
    "overlapping_prediction",
    "unmapped_prediction_label",
    "missed_entity",
)
FP_CATEGORIES = (
    "wrong_entity_type",
    "wrong_boundaries",
    "overlapping_prediction",
    "false_positive",
)


class ScoringError(RuntimeError):
    """Misaligned or inconsistent inputs; messages never contain text or values."""


# --- Inputs -------------------------------------------------------------------------------


def raw_parquet_path(split_file: str, revision: str | None = None) -> Path:
    """Locate the cached raw parquet for a split without any network access."""
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            DATASET_ID,
            filename=f"data/{split_file}-00000-of-00001.parquet",
            repo_type="dataset",
            revision=revision,
            local_files_only=True,
        )
    )


def iter_raw_rows(path: Path) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(path).iter_batches(batch_size=2_000, columns=RAW_COLUMNS):
        yield from batch.to_pylist()


@dataclass(frozen=True)
class AlignedExample:
    example_index: int
    uid: str
    document_format: str
    text: str
    truth: list[ExactMatch]
    native_truth: list[tuple[str, int, int]]
    record: PredictionRecord


def iter_aligned(
    normalized_path: Path,
    raw_path: Path,
    records: Iterator[PredictionRecord],
    ontology: Mapping[str, str],
) -> Iterator[AlignedExample]:
    """Join the three inputs by example index, verifying they describe the same record."""
    raw_rows = iter_raw_rows(raw_path)
    with normalized_path.open("r", encoding="utf-8") as normalized:
        position = -1
        for record in records:
            line: str | None = None
            raw: dict[str, Any] | None = None
            while position < record.example_index:
                line, raw = normalized.readline(), next(raw_rows, None)
                position += 1
                if not line or raw is None:
                    raise ScoringError("prediction index is beyond the end of the dataset")
            if position != record.example_index or line is None or raw is None:
                raise ScoringError("prediction records are not in increasing index order")
            try:
                text, truth = validate_normalized_example(json.loads(line))
            except ValueError:
                raise ScoringError(f"normalized record {position} is invalid") from None
            native = sorted(
                (str(span["label"]), int(span["start"]), int(span["end"]))
                for span in parse_spans(raw["spans"])
            )
            if any(label not in ontology for label, _, _ in native):
                raise ScoringError(f"record {position} has a label missing from the ontology")
            expected = sorted((evaluation_type(ontology[label]), s, e) for label, s, e in native)
            if raw["text"] != text or expected != sorted(
                (item.entity_type, item.start, item.end) for item in truth
            ):
                raise ScoringError(f"raw and normalized record {position} are not aligned")
            yield AlignedExample(
                example_index=position,
                uid=str(raw["uid"]),
                document_format=str(raw["document_format"]),
                text=text,
                truth=truth,
                native_truth=native,
                record=record,
            )


# --- Scoring ------------------------------------------------------------------------------


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def _nested(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return (a_start <= b_start and b_end <= a_end) or (b_start <= a_start and a_end <= b_end)


def classify_false_negative(
    entity: ExactMatch, predicted: Sequence[ExactMatch], removed: Sequence[RawPrediction]
) -> str:
    """Assign one taxonomy category to an unmatched ground-truth entity (priority order)."""
    if entity.entity_type == UNMAPPED:
        return "unmapped_ground_truth_label"
    overlapping = [p for p in predicted if _overlaps(p.start, p.end, entity.start, entity.end)]
    if any(p.start == entity.start and p.end == entity.end for p in overlapping):
        return "wrong_entity_type"
    if any(p.entity_type == entity.entity_type for p in overlapping):
        return "wrong_boundaries"
    if overlapping:
        return "overlapping_prediction"
    if any(_overlaps(p.start, p.end, entity.start, entity.end) for p in removed):
        return "unmapped_prediction_label"
    return "missed_entity"


def classify_false_positive(
    prediction: ExactMatch, truth: Sequence[ExactMatch]
) -> tuple[str, bool]:
    """Category for an unmatched scored prediction, and whether any overlap is nested."""
    overlapping = [t for t in truth if _overlaps(t.start, t.end, prediction.start, prediction.end)]
    nested = any(_nested(t.start, t.end, prediction.start, prediction.end) for t in overlapping)
    if any(t.start == prediction.start and t.end == prediction.end for t in overlapping):
        return "wrong_entity_type", nested
    if any(t.entity_type == prediction.entity_type for t in overlapping):
        return "wrong_boundaries", nested
    if overlapping:
        return "overlapping_prediction", nested
    return "false_positive", False


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


@dataclass
class GlinerScore:
    """Aggregate, offset-free results of one scoring pass."""

    threshold: float
    state: EvaluationState = field(default_factory=EvaluationState)
    native_truth: Counter[str] = field(default_factory=Counter)
    native_predicted: Counter[str] = field(default_factory=Counter)
    native_matches: Counter[str] = field(default_factory=Counter)
    removed_by_label: Counter[str] = field(default_factory=Counter)
    removed_exact_native_matches: Counter[str] = field(default_factory=Counter)
    unmapped_truth: Counter[str] = field(default_factory=Counter)
    fn_categories: Counter[str] = field(default_factory=Counter)
    fp_categories: Counter[str] = field(default_factory=Counter)
    fn_categories_by_type: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    fp_categories_by_type: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    fp_nested_overlaps: int = 0
    unreachable_truth: Counter[str] = field(default_factory=Counter)
    unreachable_false_negatives: Counter[str] = field(default_factory=Counter)
    reachability_by_native_label: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    fn_overlapping_native_labels: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    by_format: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    by_chunking: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    chunks: int = 0
    chunked_examples: int = 0
    duplicates_removed: int = 0
    prediction_overlap_pairs: int = 0
    samples: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))

    def sample(self, category: str, example: AlignedExample, entity: ExactMatch) -> None:
        bucket = self.samples[category]
        if len(bucket) < SAMPLES_PER_CATEGORY:
            bucket.append(
                {
                    "uid": example.uid,
                    "example_index": example.example_index,
                    "entity_type": entity.entity_type,
                    "start": entity.start,
                    "end": entity.end,
                }
            )


def score_examples(
    examples: Iterator[AlignedExample],
    label_map: GlinerLabelMap,
    threshold: float,
    max_width: int,
) -> GlinerScore:
    """Score aligned examples at ``threshold`` (``score > threshold``)."""
    result = GlinerScore(threshold=threshold)
    pending: list[tuple[AlignedExample, list[ExactMatch]]] = []

    def flush() -> None:
        if not pending:
            return
        batch = Batch([(ex.example_index, b"") for ex, _ in pending], 0, False)
        score_batch(
            result.state,
            batch,
            [
                (
                    tuple((t.entity_type, t.start, t.end) for t in ex.truth),
                    tuple((p.entity_type, p.start, p.end) for p in predicted),
                )
                for ex, predicted in pending
            ],
        )
        pending.clear()

    for example in examples:
        native_predictions = apply_threshold(example.record.predictions, threshold)
        gateway = to_gateway(native_predictions, label_map)
        predicted = gateway.kept
        removed = [p for p in native_predictions if label_map.gateway_type(p.label) == UNMAPPED]
        pending.append((example, predicted))
        if len(pending) >= SCORE_BATCH:
            flush()

        # Native-label (pre-ontology) exact matching.
        native_truth = Counter(example.native_truth)
        native_pred = Counter((p.label, p.start, p.end) for p in native_predictions)
        native_hits = native_truth & native_pred
        for (label, _, _), count in native_truth.items():
            result.native_truth[label] += count
        for (label, _, _), count in native_pred.items():
            result.native_predicted[label] += count
        for (label, _, _), count in native_hits.items():
            result.native_matches[label] += count
        for label, count in gateway.removed_by_label.items():
            result.removed_by_label[label] += count
        for (label, _, _), count in native_hits.items():
            if label_map.gateway_type(label) == UNMAPPED:
                result.removed_exact_native_matches[label] += count

        # Ground truth the gateway ontology cannot express: did the model see it?
        predicted_spans = {(p.start, p.end) for p in native_predictions}
        for label, start, end in example.native_truth:
            if label_map.gateway_type(label) != UNMAPPED:
                continue
            result.unmapped_truth["total"] += 1
            if (label, start, end) in native_pred:
                result.unmapped_truth["exact_native_match"] += 1
            elif (start, end) in predicted_spans:
                result.unmapped_truth["exact_span_other_native_label"] += 1
            elif any(_overlaps(p.start, p.end, start, end) for p in native_predictions):
                result.unmapped_truth["overlapping_prediction"] += 1
            else:
                result.unmapped_truth["no_prediction"] += 1

        # Error taxonomy on the gateway view (multiset semantics as in score_batch).
        truth_counts, predicted_counts = Counter(example.truth), Counter(predicted)
        matched = truth_counts & predicted_counts
        words = whitespace_words(example.text)
        reasons = unreachable_reasons(words, [(t.start, t.end) for t in example.truth], max_width)
        reason_by_entity = dict(zip(example.truth, reasons, strict=True))
        for reason in reasons:
            if reason:
                result.unreachable_truth[reason] += 1
        for entity, count in (truth_counts - matched).items():
            category = classify_false_negative(entity, predicted, removed)
            result.fn_categories[category] += count
            result.fn_categories_by_type[entity.entity_type][category] += count
            if reason_by_entity.get(entity):
                result.unreachable_false_negatives[str(reason_by_entity[entity])] += count
            result.sample(f"fn_{category}", example, entity)
            if category != "unmapped_ground_truth_label":
                # Which native labels took the span instead (pre-ontology, labels only).
                for p in native_predictions:
                    if _overlaps(p.start, p.end, entity.start, entity.end):
                        result.fn_overlapping_native_labels[entity.entity_type][p.label] += count
        native_hits_set = set(native_hits)
        native_reasons = unreachable_reasons(
            words, [(s, e) for _, s, e in example.native_truth], max_width
        )
        for (label, start, end), reason in zip(example.native_truth, native_reasons, strict=True):
            bucket = result.reachability_by_native_label[label]
            bucket[reason or "reachable"] += 1
            if (label, start, end) not in native_hits_set:
                bucket[f"{reason or 'reachable'}_missed"] += 1
        for prediction, count in (predicted_counts - matched).items():
            category, nested = classify_false_positive(prediction, example.truth)
            result.fp_categories[category] += count
            result.fp_categories_by_type[prediction.entity_type][category] += count
            result.fp_nested_overlaps += count * nested
            result.sample(f"fp_{category}", example, prediction)

        tp = sum(matched.values())
        fp = sum(predicted_counts.values()) - tp
        fn = sum(truth_counts.values()) - tp
        chunked = "chunked" if example.record.chunks > 1 else "single_window"
        for bucket in (result.by_format[example.document_format], result.by_chunking[chunked]):
            bucket["examples"] += 1
            bucket["tp"] += tp
            bucket["fp"] += fp
            bucket["fn"] += fn
        result.chunks += example.record.chunks
        result.chunked_examples += example.record.chunks > 1
        result.duplicates_removed += example.record.duplicates_removed
        result.prediction_overlap_pairs += overlapping_pairs(native_predictions)
    flush()
    return result


# --- Reporting ----------------------------------------------------------------------------


def coverage_section(score: GlinerScore, label_map: GlinerLabelMap) -> dict[str, Any]:
    detected = dict(sorted(score.native_predicted.items()))
    mapped = {k: v for k, v in detected.items() if label_map.gateway_type(k) != UNMAPPED}
    unmapped = {k: v for k, v in detected.items() if label_map.gateway_type(k) == UNMAPPED}
    native_rows: dict[str, dict[str, Any]] = {
        label: {**row, "gateway_type": label_map.gateway_type(label)}
        for label, row in metric_rows(
            score.native_truth, score.native_predicted, score.native_matches
        ).items()
    }
    native_tp = sum(score.native_matches.values())
    native_fp = sum(score.native_predicted.values()) - native_tp
    native_fn = sum(score.native_truth.values()) - native_tp
    prompt_labels = set(label_map.labels)
    return {
        "prompt_labels": len(label_map.labels),
        "model_labels_detected": len(detected),
        "prompt_labels_never_predicted": sorted(prompt_labels - set(detected)),
        "predictions_by_model_label": detected,
        "gateway_mapped_labels": mapped,
        "unmapped_labels": unmapped,
        "predictions_before_ontology_filtering": sum(detected.values()),
        "predictions_removed_by_ontology_filtering": sum(score.removed_by_label.values()),
        "removed_predictions_that_exactly_match_native_ground_truth": sum(
            score.removed_exact_native_matches.values()
        ),
        "removed_exact_matches_by_label": dict(sorted(score.removed_exact_native_matches.items())),
        "unmapped_ground_truth": dict(sorted(score.unmapped_truth.items())),
        "native_label_overall": _prf(native_tp, native_fp, native_fn),
        "native_label_per_entity": native_rows,
    }


def build_gliner_report(
    score: GlinerScore,
    identity: dict[str, Any],
    label_map: GlinerLabelMap,
    runtime: Mapping[str, Any] | None,
    selection: Mapping[str, Any] | None,
    complete: bool,
) -> dict[str, Any]:
    state = score.state
    state.dataset_exhausted = complete
    # Same aggregate structure as the Presidio baseline; ``run`` is replaced below.
    report = build_report(RunResult(identity, state, None, 0.0, 0, 0.0))
    overall = report["overall"]
    overall["true_positives"] = overall["exact_matches"]
    report["overall"] = overall
    for row in report["per_entity"].values():
        row["true_positives"] = row["exact_matches"]
    # build_report hard-codes the test split; record what was actually scored.
    report["split"] = identity["dataset"]["split"]
    report["dataset_file"] = identity["dataset"]["file_name"]
    report["threshold"] = score.threshold
    report["threshold_selection"] = dict(selection or {})
    report["label_map"] = {
        "file": "configs/gliner_label_map.yaml",
        "mapping": dict(sorted(label_map.mapping.items())),
        "deviations_from_ground_truth_ontology": dict(label_map.deviations),
    }
    report["coverage"] = coverage_section(score, label_map)
    report["error_analysis"] = {
        "false_negative_categories": {k: score.fn_categories.get(k, 0) for k in FN_CATEGORIES},
        "false_positive_categories": {k: score.fp_categories.get(k, 0) for k in FP_CATEGORIES},
        "false_positive_overlaps_that_are_nested": score.fp_nested_overlaps,
        "false_negative_categories_by_type": {
            k: dict(sorted(v.items())) for k, v in sorted(score.fn_categories_by_type.items())
        },
        "false_positive_categories_by_type": {
            k: dict(sorted(v.items())) for k, v in sorted(score.fp_categories_by_type.items())
        },
        "structurally_unreachable_ground_truth": dict(sorted(score.unreachable_truth.items())),
        "structurally_unreachable_false_negatives": dict(
            sorted(score.unreachable_false_negatives.items())
        ),
        # Native labels with any structurally unreachable ground truth.
        "reachability_by_native_label": {
            label: dict(sorted(counts.items()))
            for label, counts in sorted(score.reachability_by_native_label.items())
            if counts.get("reachable", 0)
            != sum(v for k, v in counts.items() if not k.endswith("_missed"))
        },
        "false_negative_overlapping_native_labels_by_type": {
            entity_type: dict(counts.most_common(5))
            for entity_type, counts in sorted(score.fn_overlapping_native_labels.items())
        },
        "predictions_overlapping_each_other_pairs": score.prediction_overlap_pairs,
        "duplicate_predictions_removed": score.duplicates_removed,
        "samples": {k: v for k, v in sorted(score.samples.items())},
    }
    report["structure"] = {
        "chunks": score.chunks,
        "chunked_examples": score.chunked_examples,
        "by_document_format": {
            k: {"examples": v["examples"], **_prf(v["tp"], v["fp"], v["fn"])}
            for k, v in sorted(score.by_format.items())
        },
        "by_chunking": {
            k: {"examples": v["examples"], **_prf(v["tp"], v["fp"], v["fn"])}
            for k, v in sorted(score.by_chunking.items())
        },
    }
    report["run"] = {
        "identity_sha256": canonical_sha256(identity),
        "dataset_sha256": identity["dataset"]["sha256"],
        "raw_parquet": identity["raw_parquet"],
        "model": identity["predictions"]["model"],
        "inference": identity["predictions"]["inference"],
        "runtime": summarize_runtime(runtime),
    }
    return report


def summarize_runtime(runtime: Mapping[str, Any] | None) -> dict[str, Any]:
    if not runtime:
        return {}
    sessions = list(runtime.get("sessions", []))
    examples = sum(int(s.get("examples", 0)) for s in sessions)
    inference = sum(float(s.get("inference_seconds", 0.0)) for s in sessions)
    wall = sum(float(s.get("wall_seconds", 0.0)) for s in sessions)
    load = sum(float(s.get("model_load_seconds", 0.0)) for s in sessions)
    return {
        "sessions": sessions,
        "examples": examples,
        "chunks": sum(int(s.get("chunks", 0)) for s in sessions),
        "model_load_seconds_total": round(load, 1),
        "inference_seconds": round(inference, 1),
        "total_runtime_seconds": round(wall + load, 1),
        "examples_per_second": round(examples / inference, 2) if inference else None,
    }


# --- CLI ----------------------------------------------------------------------------------


def load_run(
    data_dir: Path, predictions_path: Path, label_map: GlinerLabelMap
) -> tuple[dict[str, Any], Path, Path, list[int]]:
    """Validate the predictions header against the data and label map in use."""
    header = read_header(predictions_path)
    source = header["source"]
    normalized_path = data_dir / source["file_name"]
    if not normalized_path.is_file():
        raise ScoringError("normalized source file named by the predictions is missing")
    dataset_sha = sha256_file(normalized_path)
    if dataset_sha != source["sha256"]:
        raise ScoringError("predictions were produced for a different normalized file")
    if canonical_sha256(header["label_map"]) != canonical_sha256(label_map.as_identity()):
        raise ScoringError("predictions were produced with a different label map")
    if header["model"].get("words_splitter_type") != "whitespace":
        raise ScoringError("scorer's word splitter does not match the model's")
    split_file = "test" if source["source"] == "test" else "train"
    raw_path = raw_parquet_path(split_file)
    with normalized_path.open("rb") as handle:
        total = sum(1 for _ in handle)
    if source["selection"]["kind"] == "all":
        indices = list(range(total))
    else:
        indices = select_dev_indices(
            total, int(source["selection"]["size"]), source["selection"]["seed"]
        )
    identity = {
        "dataset": {
            "dataset_id": DATASET_ID,
            "split": source["source"],
            "file_name": normalized_path.name,
            "sha256": dataset_sha,
        },
        "raw_parquet": {"file": raw_path.name, "snapshot": raw_path.parent.parent.name},
        "normalization": normalization_identity(),
        "matching": matching_identity(),
        "predictions": header,
        "detector": {"class": "nvidia/gliner-PII", **header["model"]},
    }
    return identity, normalized_path, raw_path, indices


def run_scoring(
    data_dir: Path,
    predictions_path: Path,
    threshold: float | None,
    label_map: GlinerLabelMap,
) -> tuple[GlinerScore, dict[str, Any], bool]:
    identity, normalized_path, raw_path, indices = load_run(data_dir, predictions_path, label_map)
    run_threshold = float(identity["predictions"]["inference"]["threshold"])
    threshold = run_threshold if threshold is None else threshold
    if threshold < run_threshold:
        raise ScoringError("cannot score below the threshold the predictions were produced at")
    records = list(iter_records(predictions_path))
    if [r.example_index for r in records] != indices[: len(records)]:
        raise ScoringError("predictions do not follow the source selection order")
    complete = len(records) == len(indices)
    examples = iter_aligned(normalized_path, raw_path, iter(records), load_ontology())
    score = score_examples(
        examples, label_map, threshold, int(identity["predictions"]["model"]["max_width"])
    )
    return score, identity, complete


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score GLiNER predictions (strict exact span).")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIRECTORY)
    parser.add_argument("--report-name", default="gliner_baseline")
    parser.add_argument(
        "--selection",
        type=Path,
        help="Threshold-selection JSON (from select_gliner_threshold.py) to embed.",
    )
    parser.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP)
    args = parser.parse_args(argv)
    if is_within_repository(args.predictions):
        print("error: predictions must live outside the repository", file=sys.stderr)
        return 2
    label_map = load_label_map(args.label_map)
    try:
        score, identity, complete = run_scoring(args.data_dir, args.predictions, None, label_map)
    except (ScoringError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    runtime_file = args.predictions.with_name(args.predictions.name + ".runtime.json")
    runtime = (
        json.loads(runtime_file.read_text(encoding="utf-8")) if runtime_file.is_file() else None
    )
    selection = json.loads(args.selection.read_text(encoding="utf-8")) if args.selection else None
    report = build_gliner_report(score, identity, label_map, runtime, selection, complete)
    presidio_path = REPORT_DIRECTORY / "presidio_baseline.json"
    report["dataset_matches_presidio_baseline"] = (
        json.loads(presidio_path.read_text(encoding="utf-8"))["run"]["dataset_sha256"]
        == report["run"]["dataset_sha256"]
        if presidio_path.is_file()
        else None
    )
    from scripts.gliner_report import render_gliner_markdown

    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / f"{args.report_name}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.report_dir / f"{args.report_name}.md").write_text(
        render_gliner_markdown(report), encoding="utf-8"
    )
    overall = report["overall"]
    print(
        f"scored examples={report['examples_evaluated']} complete_split={complete} "
        f"threshold={score.threshold} precision={overall['precision']:.4f} "
        f"recall={overall['recall']:.4f} f1={overall['f1']:.4f} reports={args.report_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
