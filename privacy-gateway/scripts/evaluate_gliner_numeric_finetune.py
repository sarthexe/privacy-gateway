"""Fixed-protocol evaluation of the numeric fine-tuning experiment (offline).

``predict``   runs a baseline or fine-tuned checkpoint at t=0.3 and t=0.7 with the
              frozen OOD protocol (max_width=24, batch_size=1, fp32, whitespace
              splitter) and writes offset-only prediction files outside the repo.
``compare``   scores a baseline and a candidate prediction file against the same
              normalized JSONL with the shared strict exact-span scorer, reports
              forgetting for every native label, and applies the safety gate.
``summarize`` collects compare reports into one JSON and Markdown table.

``other_numeric`` predictions map to no gateway type: they are removed before
gateway, supported-only, and native scoring, and reported separately. Thresholds
are fixed; nothing here tunes them, and no verdict selects a production model.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from itertools import zip_longest
from pathlib import Path
from typing import IO, Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.batch_evaluation import Batch, score_batch
from scripts.evaluation_checkpoint import EvaluationState, canonical_sha256, sha256_file
from scripts.gliner_evaluation import (
    UNMAPPED,
    GlinerLabelMap,
    to_gateway,
    whitespace_words,
)
from scripts.gliner_numeric_common import (
    DEFAULT_CONFIG,
    EXPERIMENT_ID,
    OOD_BENCHMARK_SHA256,
    OTHER_NUMERIC,
    PROTOCOL_MAX_WIDTH,
    PROTOCOL_THRESHOLDS,
    NumericConfig,
    NumericExperimentError,
    check_output_path,
    extended_label_map,
    load_config,
    verify_architecture,
)
from scripts.gliner_predictions import (
    PREDICTIONS_FORMAT,
    PREDICTIONS_VERSION,
    PredictionRecord,
    PredictionsError,
    decode_record,
)
from scripts.presidio_evaluation import (
    SUPPORTED_TYPES,
    ExactMatch,
    metric_rows,
    validate_normalized_example,
)

LINEAGE_FILE = "numeric_finetune_lineage.json"
LABEL_SETS = ("native", "native+other_numeric")
Span = tuple[str, int, int]


# --- Prediction files -------------------------------------------------------------------


@contextmanager
def _open_text(path: Path) -> Iterator[IO[str]]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", newline="\n") as handle:
            yield handle
    else:
        with path.open("r", encoding="utf-8", newline="\n") as handle:
            yield handle


def _header(line: str) -> dict[str, Any]:
    try:
        header = json.loads(line)
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


def read_prediction_identity(path: Path) -> dict[str, Any]:
    with _open_text(path) as handle:
        return _header(handle.readline())


def iter_prediction_records(path: Path) -> Iterator[PredictionRecord]:
    """Plain or gzip prediction files (the frozen baseline artifacts are gzip)."""
    with _open_text(path) as handle:
        _header(handle.readline())
        for line in handle:
            if not line.endswith("\n"):
                raise PredictionsError("predictions file ends with an incomplete record")
            yield decode_record(line)


def iter_normalized(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="\n") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("normalized record must be an object")
                yield row


def label_set(identity: Mapping[str, Any]) -> str:
    """Older baseline files predate the label-set field and prompted the 55 labels."""
    protocol = identity.get("protocol") or {}
    value = protocol.get("label_set", "native")
    if value not in LABEL_SETS:
        raise NumericExperimentError("prediction label set is not recognized")
    return str(value)


def benchmark_name(digest: str) -> str:
    for name, expected in OOD_BENCHMARK_SHA256.items():
        if digest == expected:
            return name
    return "custom"


# --- predict ----------------------------------------------------------------------------


def prompt_label_map(config: NumericConfig, labels: str) -> GlinerLabelMap:
    if labels not in LABEL_SETS:
        raise NumericExperimentError("unknown label set")
    native = config.label_map
    return extended_label_map(native) if labels == "native+other_numeric" else native


def resolve_label_set(requested: str, variant: str) -> str:
    if requested == "auto":
        return "native+other_numeric" if variant == "finetuned" else "native"
    return requested


def load_lineage(model_dir: Path, config: NumericConfig) -> dict[str, Any]:
    path = model_dir / LINEAGE_FILE
    if not path.is_file():
        raise NumericExperimentError("fine-tuned model has no lineage file")
    lineage: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    base = config.raw["base_model"]
    expected = {"model_id": base["model_id"], "revision": base["revision"]}
    if lineage.get("experiment") != EXPERIMENT_ID or lineage.get("base_model") != expected:
        raise NumericExperimentError("fine-tuned model lineage differs from this experiment")
    if lineage.get("training_labels") != list(config.training_labels):
        raise NumericExperimentError("fine-tuned model was trained with another vocabulary")
    return lineage


def predict(args: argparse.Namespace) -> list[Path]:
    from scripts.gliner_evaluation import GlinerPredictor, InferenceConfig
    from scripts.gliner_predict import describe_model, load_model, require_max_width
    from scripts.gliner_predictions import PredictionWriter

    config = load_config(args.config)
    check_output_path(args.output_dir)
    variant = "baseline" if args.model == "base" else "finetuned"
    labels = resolve_label_set(args.labels, variant)
    label_map = prompt_label_map(config, labels)
    evaluation = config.evaluation
    max_width, batch_size = int(evaluation["max_width"]), int(evaluation["batch_size"])
    normalized_sha256 = sha256_file(args.normalized)
    rows = list(iter_normalized(args.normalized))

    model_extra: dict[str, Any] = {"variant": variant}
    if variant == "baseline":
        model, _ = load_model(config.label_map, args.device, "fp32", max_width=max_width)
    else:
        from gliner import GLiNER

        model_dir = Path(args.model)
        lineage = load_lineage(model_dir, config)
        model = GLiNER.from_pretrained(
            str(model_dir), map_location=args.device, max_width=max_width
        )
        require_max_width(model, max_width)
        verify_architecture(model, config.raw["base_model"])
        model.eval()
        model_extra.update(
            lineage_sha256=canonical_sha256(lineage),
            pilot=bool(lineage.get("pilot")),
            weights_sha256={
                path.name: sha256_file(path)
                for path in sorted(model_dir.iterdir())
                if path.suffix in {".safetensors", ".bin"}
            },
        )
    identity_base = {
        "normalized_sha256": normalized_sha256,
        "benchmark": benchmark_name(normalized_sha256),
        "model": {**describe_model(model, config.label_map, "fp32"), **model_extra},
        "protocol": {
            "max_width": max_width,
            "batch_size": batch_size,
            "thresholds": list(PROTOCOL_THRESHOLDS),
            "email_hybrid": "not_run",
            "label_set": labels,
            "prompt_labels_sha256": canonical_sha256(list(label_map.labels)),
        },
    }
    written = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for threshold in PROTOCOL_THRESHOLDS:
        path = args.output_dir / f"predictions_t{threshold}.jsonl"
        predictor = GlinerPredictor(
            model,
            label_map,
            lambda text: [(text[s:e], s, e) for s, e in whitespace_words(text)],
            InferenceConfig(threshold, batch_size, int(model.config.max_len), max_width),
        )
        writer = PredictionWriter(
            path, {**identity_base, "threshold": threshold}, range(len(rows)), args.resume, False
        )
        for index in range(writer.completed, len(rows)):
            text, _ = validate_normalized_example(rows[index])
            result = predictor.predict([text])[0]
            writer.append(
                [
                    PredictionRecord(
                        index, result.chunks, result.duplicates_removed, tuple(result.predictions)
                    )
                ]
            )
        written.append(path)
    return written


# --- compare ----------------------------------------------------------------------------


def _summary(state: EvaluationState) -> dict[str, Any]:
    totals = state.aggregate()
    types = state.entity_types()
    support = Counter({t: state.true_positives[t] + state.false_negatives[t] for t in types})
    predicted = Counter({t: state.true_positives[t] + state.false_positives[t] for t in types})
    overall = metric_rows(
        Counter({"overall": totals["tp"] + totals["fn"]}),
        Counter({"overall": totals["tp"] + totals["fp"]}),
        Counter({"overall": totals["tp"]}),
    )["overall"]
    per_type = {
        entity_type: {
            **row,
            "tp": row["exact_matches"],
            "fp": row["false_positives"],
            "fn": row["false_negatives"],
        }
        for entity_type, row in metric_rows(support, predicted, state.true_positives).items()
    }
    return {
        **totals,
        "precision": overall["precision"],
        "recall": overall["recall"],
        "f1": overall["f1"],
        "per_type": per_type,
    }


def _score(
    state: EvaluationState, index: int, truth: Sequence[Span], predicted: Sequence[Span]
) -> None:
    score_batch(state, Batch([(index, b"")], 0, False), [(tuple(truth), tuple(predicted))])


def _spans(items: Sequence[ExactMatch]) -> list[Span]:
    return [(item.entity_type, item.start, item.end) for item in items]


def _native_truth(row: Mapping[str, Any], text: str, labels: Sequence[str]) -> list[Span]:
    """Native GT spans; UNMAPPED and the training-only label are not scored natively."""
    native = row.get("native_entities")
    if not isinstance(native, list):
        raise ValueError("normalized record must contain native_entities")
    allowed = {*labels, UNMAPPED, OTHER_NUMERIC}
    spans = []
    for item in native:
        if not isinstance(item, dict) or item.get("native_label") not in allowed:
            raise ValueError("native ground-truth label is invalid")
        start, end = item.get("start"), item.get("end")
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text)):
            raise ValueError("native ground-truth offsets are invalid")
        spans.append((str(item["native_label"]), start, end))
    if Counter((s, e) for _, s, e in spans) != Counter(
        (item["start"], item["end"]) for item in row["entities"]
    ):
        raise ValueError("native and gateway ground-truth spans differ")
    return [span for span in spans if span[0] not in (UNMAPPED, OTHER_NUMERIC)]


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


class _Side:
    """Strict scoring state for one model's predictions on one dataset."""

    def __init__(self, label_map: GlinerLabelMap) -> None:
        self.label_map = label_map
        self.full = EvaluationState()
        self.supported = EvaluationState()
        self.native = EvaluationState()
        self.other_numeric_predictions = 0
        self.other_numeric_on_gt = 0
        self.other_numeric_exact_gt = 0
        self.other_numeric_gt = EvaluationState()

    def add(
        self,
        index: int,
        text: str,
        truth: Sequence[ExactMatch],
        native_truth: Sequence[Span],
        other_truth: Sequence[Span],
        record: PredictionRecord,
    ) -> tuple[set[Span], list[tuple[int, int]]]:
        for item in record.predictions:
            if (
                item.label not in self.label_map.mapping
                or item.end > len(text)
                or not math.isfinite(item.score)
            ):
                raise ValueError("prediction label, offsets, or score are invalid")
        kept = _spans(to_gateway(record.predictions, self.label_map).kept)
        gold = _spans(truth)
        _score(self.full, index, gold, kept)
        _score(
            self.supported,
            index,
            [span for span in gold if span[0] in SUPPORTED_TYPES],
            [span for span in kept if span[0] in SUPPORTED_TYPES],
        )
        native_predicted = [
            (p.label, p.start, p.end) for p in record.predictions if p.label != OTHER_NUMERIC
        ]
        _score(self.native, index, native_truth, native_predicted)
        other = [(p.start, p.end) for p in record.predictions if p.label == OTHER_NUMERIC]
        self.other_numeric_predictions += len(other)
        mapped_gold = [span for span in gold if span[0] != UNMAPPED]
        self.other_numeric_on_gt += sum(
            any(_overlaps((s, e), span) for span in other) for _, s, e in mapped_gold
        )
        self.other_numeric_exact_gt += sum((s, e) in other for _, s, e in mapped_gold)
        _score(
            self.other_numeric_gt,
            index,
            list(other_truth),
            [(OTHER_NUMERIC, s, e) for s, e in other],
        )
        return set(gold) & set(kept), other

    def report(self, mapped_gt: int) -> dict[str, Any]:
        return {
            "full_strict": _summary(self.full),
            "supported_only": _summary(self.supported),
            "native_label_space": _summary(self.native),
            "other_numeric": {
                "predictions": self.other_numeric_predictions,
                "mapped_gt_overlapped": self.other_numeric_on_gt,
                "mapped_gt_exact_span": self.other_numeric_exact_gt,
                "absorbed_fraction": self.other_numeric_on_gt / mapped_gt if mapped_gt else 0.0,
                "vs_other_numeric_gt": _summary(self.other_numeric_gt),
            },
        }


def _f1(report: Mapping[str, Any], space: str, entity_type: str) -> tuple[float | None, int]:
    row = report[space]["per_type"].get(entity_type)
    if row is None:
        return None, 0
    return float(row["f1"]), int(row["support"])


def forgetting_table(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any], labels: Sequence[str]
) -> dict[str, dict[str, Any]]:
    table = {}
    for label in labels:
        before, support = _f1(baseline, "native_label_space", label)
        after, _ = _f1(candidate, "native_label_space", label)
        table[label] = {
            "support": support,
            "baseline_f1": before,
            "candidate_f1": after,
            "f1_delta": None if support == 0 else (after or 0.0) - (before or 0.0),
        }
    return table


def safety_gate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    forgetting: Mapping[str, Any],
    tp_lost: int,
    gate: Mapping[str, Any],
    targets: Sequence[str],
) -> dict[str, Any]:
    """Flags only. CREDIT_CARD improving does not offset any flag."""
    flags = []
    drop = baseline["supported_only"]["f1"] - candidate["supported_only"]["f1"]
    if drop > gate["supported_only_f1_max_drop"]:
        flags.append({"flag": "supported_only_f1_drop", "drop": drop})
    for entity_type in gate["major_types"]:
        before, support = _f1(baseline, "full_strict", entity_type)
        after, _ = _f1(candidate, "full_strict", entity_type)
        if support >= gate["major_type_min_support"]:
            change = (before or 0.0) - (after or 0.0)
            if change > gate["major_type_f1_max_drop"]:
                flags.append({"flag": "major_type_f1_drop", "type": entity_type, "drop": change})
    for label, row in forgetting.items():
        if row["support"] >= gate["native_label_min_support"] and row["f1_delta"] is not None:
            if -row["f1_delta"] > gate["native_label_f1_max_drop"]:
                flags.append(
                    {"flag": "native_label_f1_drop", "label": label, "drop": -row["f1_delta"]}
                )
    absorbed = candidate["other_numeric"]["absorbed_fraction"]
    if absorbed > gate["other_numeric_absorbed_max_fraction"]:
        flags.append({"flag": "other_numeric_absorbs_supported_entities", "fraction": absorbed})
    baseline_tp = baseline["full_strict"]["tp"]
    lost_fraction = tp_lost / baseline_tp if baseline_tp else 0.0
    if lost_fraction > gate["tp_lost_to_other_numeric_max_fraction"]:
        flags.append({"flag": "true_positives_lost_to_other_numeric", "fraction": lost_fraction})
    improved = []
    for entity_type in targets:
        before, support = _f1(baseline, "full_strict", entity_type)
        after, _ = _f1(candidate, "full_strict", entity_type)
        if support and (after or 0.0) > (before or 0.0):
            improved.append(entity_type)
    if flags:
        verdict = "flagged"
    elif not improved:
        verdict = "no_target_improvement"
    else:
        verdict = "no_gate_flags_review_required"
    return {
        "flags": flags,
        "improved_target_types": improved,
        "tp_lost_to_other_numeric": tp_lost,
        "tp_lost_to_other_numeric_fraction": lost_fraction,
        "verdict": verdict,
        "note": "never selects a production model; a human review decides",
    }


def check_comparable(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    normalized_sha256: str,
    config: NumericConfig,
) -> None:
    for name, identity in (("baseline", baseline), ("candidate", candidate)):
        if identity.get("normalized_sha256") != normalized_sha256:
            raise NumericExperimentError(f"{name} predictions are for another dataset")
        protocol = identity.get("protocol") or {}
        if protocol.get("max_width") != PROTOCOL_MAX_WIDTH or protocol.get("batch_size") != 1:
            raise NumericExperimentError(f"{name} predictions used another protocol")
        model = identity.get("model") or {}
        if (model.get("model_id"), model.get("model_revision")) != (
            config.label_map.model_id,
            config.label_map.model_revision,
        ):
            raise NumericExperimentError(f"{name} predictions are not from the pinned lineage")
        if model.get("dtype", "fp32") != "fp32":
            raise NumericExperimentError(f"{name} predictions were not fp32")
    if baseline.get("threshold") != candidate.get("threshold"):
        raise NumericExperimentError("baseline and candidate thresholds differ")
    if baseline.get("threshold") not in PROTOCOL_THRESHOLDS:
        raise NumericExperimentError("threshold is outside the fixed protocol")


def compare(
    config: NumericConfig, normalized: Path, baseline_path: Path, candidate_path: Path
) -> dict[str, Any]:
    normalized_sha256 = sha256_file(normalized)
    identities = read_prediction_identity(baseline_path), read_prediction_identity(candidate_path)
    check_comparable(*identities, normalized_sha256, config)
    maps = [prompt_label_map(config, label_set(identity)) for identity in identities]
    sides = [_Side(label_map) for label_map in maps]
    native_labels = config.native_labels
    mapped_gt = tp_lost = examples = 0
    for index, (row, base_record, cand_record) in enumerate(
        zip_longest(
            iter_normalized(normalized),
            iter_prediction_records(baseline_path),
            iter_prediction_records(candidate_path),
        )
    ):
        if row is None or base_record is None or cand_record is None:
            raise NumericExperimentError("prediction record counts differ from the dataset")
        if base_record.example_index != index or cand_record.example_index != index:
            raise NumericExperimentError("prediction records are not in dataset order")
        text, truth = validate_normalized_example(row)
        native_truth = _native_truth(row, text, native_labels)
        other_truth = [
            (OTHER_NUMERIC, int(item["start"]), int(item["end"]))
            for item in row["native_entities"]
            if item["native_label"] == OTHER_NUMERIC
        ]
        base_tp, _ = sides[0].add(index, text, truth, native_truth, other_truth, base_record)
        cand_tp, cand_other = sides[1].add(
            index, text, truth, native_truth, other_truth, cand_record
        )
        tp_lost += sum(
            any(_overlaps((s, e), span) for span in cand_other) for _, s, e in base_tp - cand_tp
        )
        mapped_gt += sum(item.entity_type != UNMAPPED for item in truth)
        examples += 1
    baseline, candidate = (side.report(mapped_gt) for side in sides)
    evaluation = config.evaluation
    forgetting = forgetting_table(baseline, candidate, native_labels)
    highlighted = {
        entity_type: {
            "baseline": baseline["full_strict"]["per_type"].get(entity_type),
            "candidate": candidate["full_strict"]["per_type"].get(entity_type),
        }
        for entity_type in evaluation["highlighted_types"]
    }
    return {
        "experiment": EXPERIMENT_ID,
        "privacy": "aggregate-only; no source text, entity values, or offsets",
        "dataset": {
            "normalized_sha256": normalized_sha256,
            "benchmark": benchmark_name(normalized_sha256),
            "examples": examples,
            "mapped_gt_entities": mapped_gt,
        },
        "threshold": identities[0]["threshold"],
        "baseline_identity_sha256": canonical_sha256(identities[0]),
        "candidate_identity_sha256": canonical_sha256(identities[1]),
        "label_sets": {"baseline": label_set(identities[0]), "candidate": label_set(identities[1])},
        "candidate_pilot": bool((identities[1].get("model") or {}).get("pilot")),
        "baseline": baseline,
        "candidate": candidate,
        "deltas": {
            space: candidate[space]["f1"] - baseline[space]["f1"]
            for space in ("full_strict", "supported_only", "native_label_space")
        },
        "highlighted_types": highlighted,
        "forgetting": forgetting,
        "safety_gate": safety_gate(
            baseline,
            candidate,
            forgetting,
            tp_lost,
            evaluation["safety_gate"],
            evaluation["target_types"],
        ),
    }


# --- summarize --------------------------------------------------------------------------


def summarize(reports: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], str]:
    rows = []
    for report in reports:
        gate = report["safety_gate"]
        targets = {}
        for entity_type, sides in report["highlighted_types"].items():
            targets[entity_type] = {
                side: (sides[side] or {}).get("f1") for side in ("baseline", "candidate")
            }
        rows.append(
            {
                "benchmark": report["dataset"]["benchmark"],
                "threshold": report["threshold"],
                "full_strict_f1": [
                    report["baseline"]["full_strict"]["f1"],
                    report["candidate"]["full_strict"]["f1"],
                ],
                "supported_only_f1": [
                    report["baseline"]["supported_only"]["f1"],
                    report["candidate"]["supported_only"]["f1"],
                ],
                "native_f1": [
                    report["baseline"]["native_label_space"]["f1"],
                    report["candidate"]["native_label_space"]["f1"],
                ],
                "other_numeric_predictions": report["candidate"]["other_numeric"]["predictions"],
                "highlighted_f1": targets,
                "flags": [flag["flag"] for flag in gate["flags"]],
                "verdict": gate["verdict"],
            }
        )
    rows.sort(key=lambda row: (row["benchmark"], row["threshold"]))
    lines = [
        "| Benchmark | t | Supported F1 (base → cand) | Native F1 | CREDIT_CARD F1 | "
        "BANK_ACCOUNT F1 | ID_CARD F1 | other_numeric preds | Verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    def pair(values: Sequence[float | None]) -> str:
        return " → ".join("n/a" if value is None else f"{value:.4f}" for value in values)

    for row in rows:
        highlighted = row["highlighted_f1"]
        cells = [
            pair([highlighted.get(t, {}).get("baseline"), highlighted.get(t, {}).get("candidate")])
            for t in ("CREDIT_CARD", "BANK_ACCOUNT", "ID_CARD")
        ]
        lines.append(
            f"| {row['benchmark']} | {row['threshold']} | {pair(row['supported_only_f1'])} | "
            f"{pair(row['native_f1'])} | {' | '.join(cells)} | "
            f"{row['other_numeric_predictions']} | {row['verdict']} |"
        )
    summary = {
        "experiment": EXPERIMENT_ID,
        "rows": rows,
        "any_flagged": any(row["flags"] for row in rows),
        "note": "aggregate-only; no production model is selected",
    }
    return summary, "\n".join(lines) + "\n"


# --- CLI --------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the GLiNER numeric fine-tune.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("predict", help="Fixed-protocol predictions at t=0.3 and t=0.7.")
    run.add_argument("--normalized", type=Path, required=True)
    run.add_argument("--model", required=True, help="'base' or a fine-tuned final_model dir.")
    run.add_argument("--labels", choices=("auto", *LABEL_SETS), default="auto")
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--device", default="cuda")
    run.add_argument("--resume", action="store_true")
    pair = commands.add_parser("compare", help="Strict before/after comparison and gate.")
    pair.add_argument("--normalized", type=Path, required=True)
    pair.add_argument("--baseline", type=Path, required=True)
    pair.add_argument("--candidate", type=Path, required=True)
    pair.add_argument("--output", type=Path, required=True)
    collect = commands.add_parser("summarize", help="Combine compare reports.")
    collect.add_argument("reports", type=Path, nargs="+")
    collect.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "predict":
            for path in predict(args):
                print(f"wrote {path}", flush=True)
        elif args.command == "compare":
            check_output_path(args.output)
            report = compare(
                load_config(args.config), args.normalized, args.baseline, args.candidate
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            gate = report["safety_gate"]
            print(
                f"{report['dataset']['benchmark']} t={report['threshold']} "
                f"supported_f1_delta={report['deltas']['supported_only']:+.4f} "
                f"verdict={gate['verdict']} flags={len(gate['flags'])}",
                flush=True,
            )
        else:
            check_output_path(args.output)
            reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
            summary, table = summarize(reports)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            args.output.with_suffix(".md").write_text(table, encoding="utf-8")
            print(table, end="", flush=True)
    except (NumericExperimentError, PredictionsError, OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
