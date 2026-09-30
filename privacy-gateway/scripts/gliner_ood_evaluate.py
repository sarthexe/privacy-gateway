"""GPU-only, fixed-condition GLiNER OOD runner; do not use for threshold tuning."""
from __future__ import annotations
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.batch_evaluation import Batch, score_batch
from scripts.evaluation_checkpoint import EvaluationState, canonical_sha256, sha256_file
from scripts.gliner_evaluation import GlinerPredictor, InferenceConfig, RawPrediction, load_label_map, to_gateway, whitespace_words
from scripts.gliner_predict import describe_model, load_model
from scripts.gliner_predictions import PredictionRecord, PredictionWriter
from scripts.nemotron_pii import is_within_repository
from scripts.presidio_evaluation import ExactMatch, metric_rows, validate_normalized_example

MODEL_REVISION = "bd23e8ef4425fd04e34c5204ab49ffaa706eae79"
THRESHOLDS = (0.3, 0.7)

def _tuples(items: Sequence[ExactMatch]) -> tuple[tuple[str, int, int], ...]:
    return tuple((item.entity_type, item.start, item.end) for item in items)

def _metrics(state: EvaluationState) -> dict[str, Any]:
    totals = state.aggregate(); types = state.entity_types()
    support = Counter({x: state.true_positives[x] + state.false_negatives[x] for x in types})
    predicted = Counter({x: state.true_positives[x] + state.false_positives[x] for x in types})
    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0.0
    recall = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else 0.0
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "tp": totals["tp"], "fp": totals["fp"], "fn": totals["fn"], "per_entity": metric_rows(support, predicted, state.true_positives)}

def _native(record: dict[str, Any]) -> tuple[tuple[str, int, int], ...]:
    rows = record.get("native_entities", [])
    if not isinstance(rows, list): raise ValueError("native entities are invalid")
    return tuple((str(x["native_label"]), int(x["start"]), int(x["end"])) for x in rows if x["native_label"] != "UNMAPPED")

def run(normalized: Path, output_dir: Path, device: str) -> None:
    if is_within_repository(output_dir): raise ValueError("output directory must be outside repository")
    label_map = load_label_map()
    if label_map.model_revision != MODEL_REVISION: raise ValueError("label map model revision differs from OOD protocol")
    rows = [json.loads(line) for line in normalized.read_text(encoding="utf-8").splitlines()]
    model, _ = load_model(label_map, device, "fp32", max_width=24)
    identity_base = {"normalized_sha256": sha256_file(normalized), "model": describe_model(model, label_map, "fp32"), "protocol": {"max_width": 24, "batch_size": 1, "thresholds": list(THRESHOLDS), "email_hybrid": "not_run"}}
    for threshold in THRESHOLDS:
        prediction_path = output_dir / f"gliner_ood_t{threshold}.jsonl"; report_path = output_dir / f"gliner_ood_t{threshold}.json"
        if prediction_path.exists() or report_path.exists(): raise ValueError("OOD output already exists")
        predictor = GlinerPredictor(model, label_map, lambda text: [(text[s:e], s, e) for s, e in whitespace_words(text)], InferenceConfig(threshold, 1, int(model.config.max_len), 24))
        writer = PredictionWriter(prediction_path, {**identity_base, "threshold": threshold}, range(len(rows)), False, False)
        gateway, native, languages = EvaluationState(), EvaluationState(), defaultdict(EvaluationState)
        unmapped = Counter()
        for index, row in enumerate(rows):
            text, truth = validate_normalized_example(row); result = predictor.predict([text])[0]
            predicted = to_gateway(result.predictions, label_map).kept
            native_truth = _native(row); native_pred = tuple((p.label, p.start, p.end) for p in result.predictions)
            batch = Batch([(index, b"")], 0, index == len(rows) - 1)
            score_batch(gateway, batch, [(_tuples(truth), _tuples(predicted))]); score_batch(native, batch, [(native_truth, native_pred)])
            language = str(row.get("metadata", {}).get("language") or "unknown")
            score_batch(languages[language], batch, [(_tuples(truth), _tuples(predicted))])
            unmapped["ground_truth"] += sum(x.entity_type == "UNMAPPED" for x in truth)
            writer.append([PredictionRecord(index, result.chunks, result.duplicates_removed, tuple(result.predictions))])
        report = {"privacy": "aggregate-only; no source text or entity values", "identity": identity_base, "threshold": threshold,
                  "gateway_ontology": _metrics(gateway), "native_label_space": _metrics(native), "unmapped_unsupported_ground_truth": dict(unmapped),
                  "by_language_gateway_ontology": {key: _metrics(value) for key, value in sorted(languages.items())}, "identity_sha256": canonical_sha256(identity_base)}
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run fixed t=0.3/t=0.7 GLiNER OOD evaluation (GPU required).")
    parser.add_argument("--normalized", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    try: run(args.normalized, args.output_dir, args.device)
    except (OSError, ValueError, RuntimeError) as error: print(f"error: {error}", file=sys.stderr); return 2
    return 0
if __name__ == "__main__": raise SystemExit(main())
