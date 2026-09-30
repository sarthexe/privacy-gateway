"""Resumable Presidio baseline on the normalized Nemotron-PII test split.

Detection runs in bounded, checkpointed batches (see ``scripts/batch_evaluation.py``).
Checkpoints live outside the repository and hold no source text or entity values;
an interrupted run continues with ``--resume``. Reports are aggregate-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.batch_evaluation import (
    DEFAULT_BATCH_SIZE,
    SPLIT,
    EvaluationError,
    RunOptions,
    RunResult,
    run_evaluation,
)
from scripts.evaluation_checkpoint import canonical_sha256
from scripts.nemotron_pii import DATASET_ID
from scripts.presidio_evaluation import EVALUATION_TYPE_MAP, metric_rows

REPORT_DIRECTORY = Path(__file__).resolve().parents[1] / "reports" / "pii"


def build_report(result: RunResult) -> dict[str, Any]:
    """Build the aggregate-only report from checkpointed state."""
    state = result.state
    types = state.entity_types()
    support = Counter(
        {key: state.true_positives[key] + state.false_negatives[key] for key in types}
    )
    predictions = Counter(
        {key: state.true_positives[key] + state.false_positives[key] for key in types}
    )
    per_entity = metric_rows(+support, +predictions, +state.true_positives)
    totals = state.aggregate()
    total_support, total_predictions = totals["tp"] + totals["fn"], totals["tp"] + totals["fp"]
    precision = totals["tp"] / total_predictions if total_predictions else 0.0
    recall = totals["tp"] / total_support if total_support else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "dataset": DATASET_ID,
        "split": SPLIT,
        "matching": "exact entity type and half-open offsets",
        "harness": "nervaluate strict",
        "complete_split": state.dataset_exhausted,
        "limit": result.limit,
        "examples_evaluated": state.completed_examples,
        "examples_with_pii": state.examples_with_pii,
        "predictions": total_predictions,
        "ground_truth_entities": total_support,
        "exact_matches": totals["tp"],
        "harness_cross_check": {
            "nervaluate_strict_exact_first": state.harness_exact_matches,
            "nervaluate_strict_detector_order": state.harness_greedy_exact_matches,
        },
        "overall": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": total_support,
            "predictions": total_predictions,
            "exact_matches": totals["tp"],
            "false_positives": totals["fp"],
            "false_negatives": totals["fn"],
        },
        "per_entity": per_entity,
        "evaluation_type_map": dict(sorted(EVALUATION_TYPE_MAP.items())),
        "false_negative_groups": dict(sorted(state.false_negative_groups.items())),
        "false_positive_groups": dict(sorted(state.false_positive_groups.items())),
        "false_negative_samples": state.false_negative_samples,
        "false_positive_samples": state.false_positive_samples,
        "run": {
            "identity_sha256": canonical_sha256(result.identity),
            "detector": result.identity["detector"],
            "dataset_sha256": result.identity["dataset"]["sha256"],
            "elapsed_seconds": round(state.elapsed_seconds, 1),
            "examples_per_second": round(state.completed_examples / state.elapsed_seconds, 1)
            if state.elapsed_seconds
            else None,
            "sessions": state.sessions,
        },
    }


def render_markdown(report: dict[str, Any], command: str) -> str:
    """Render an aggregate-only Markdown report without source text or values."""
    overall = report["overall"]
    run = report["run"]
    scope = (
        "the complete test split"
        if report["complete_split"]
        else f"a {report['examples_evaluated']:,}-example prefix (partial run)"
    )
    batch_sizes = sorted({session["batch_size"] for session in run["sessions"]})
    lines = [
        "# Presidio baseline: Nemotron-PII test split",
        "",
        "## Dataset and methodology",
        "",
        f"- Dataset: `nvidia/Nemotron-PII`, normalized `test.jsonl`; this report covers {scope}.",
        "- Matching: exact entity type and half-open `[start, end)` offsets.",
        "- Scoring: order-independent multiset match of `(type, start, end)`; each prediction "
        "can match at most one identical ground-truth entity.",
        "- Harness: `nervaluate` strict strategy must reproduce the exact-match count on every "
        f"checkpointed batch (batch size {', '.join(map(str, batch_sizes))}).",
        "- No source text, entity values, or raw annotations are persisted in this report.",
        "",
        "## Overall metrics",
        "",
        "| Precision | Recall | F1 | Support | Predictions | Exact matches |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {overall['precision']:.4f} | {overall['recall']:.4f} | {overall['f1']:.4f} | "
            f"{overall['support']} | {overall['predictions']} | {overall['exact_matches']} |"
        ),
        "",
        "Examples evaluated: "
        f"{report['examples_evaluated']}; examples containing PII: {report['examples_with_pii']}.",
        "",
        "## Per-entity metrics",
        "",
        "| Type | Precision | Recall | F1 | Support | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entity_type, metrics in report["per_entity"].items():
        lines.append(
            f"| {entity_type} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['f1']:.4f} | {metrics['support']} | {metrics['false_positives']} | "
            f"{metrics['false_negatives']} |"
        )
    lines.extend(
        [
            "",
            "## Error analysis",
            "",
            "Top false-negative categories: "
            f"`{json.dumps(report['false_negative_groups'], sort_keys=True)}`.",
            "",
            "Top false-positive categories: "
            f"`{json.dumps(report['false_positive_groups'], sort_keys=True)}`.",
            "",
            "## Unsupported and unmapped labels",
            "",
            "`UNMAPPED` and entity types not implemented by Phase 1 are retained in scoring as "
            "ground truth; they cannot produce a Presidio exact match. DATE and TIME are evaluated "
            "through "
            "the documented adapter mapping to Phase 1 `DATE_TIME`.",
            "",
            "## Harness note",
            "",
            "nervaluate 1.2's strict matcher is greedy in prediction order: when Presidio emits "
            "nested same-type spans (common for `DATE_TIME`), an overlapping prediction listed "
            "before the exact one consumes the true entity as *incorrect* and the exact prediction "
            "is scored *spurious*. Exact matching is order-independent, so the cross-check lists "
            "exact matches first. In detector order nervaluate counts "
            f"{report['harness_cross_check']['nervaluate_strict_detector_order']} exact matches "
            f"versus {report['exact_matches']}.",
            "",
            "## Run",
            "",
            f"- Detection time: {run['elapsed_seconds']} s across {len(run['sessions'])} "
            f"session(s); throughput {run['examples_per_second']} examples/s.",
            f"- Detector: Presidio Analyzer {run['detector'].get('presidio_analyzer')}, spaCy "
            f"{run['detector'].get('spacy')}, models `{run['detector'].get('nlp_models')}`.",
            f"- Dataset SHA-256: `{run['dataset_sha256']}`.",
            "",
            "## Limitations",
            "",
            "This is a zero-tuning baseline. Exact spans penalize boundary differences, and "
            "Presidio's "
            "built-in recognizers do not cover many Nemotron PII/PHI categories.",
            "",
            "## Reproduce",
            "",
            f"```bash\n{command}\n```",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(report: dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "presidio_baseline.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    command = (
        "python scripts/evaluate_presidio.py --data-dir /path/to/nemotron-normalized "
        "--workers 4 --resume"
    )
    (report_dir / "presidio_baseline.md").write_text(
        render_markdown(report, command), encoding="utf-8"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumable exact-span Presidio baseline on normalized Nemotron test data."
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="External directory containing test.jsonl."
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=REPORT_DIRECTORY,
        help="Directory for aggregate, PII-free reports.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="External checkpoint file (default: <data-dir>/checkpoints/presidio_test.json).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="Continue from the checkpoint.")
    mode.add_argument(
        "--restart", action="store_true", help="Discard an existing checkpoint and start over."
    )
    parser.add_argument(
        "--limit", type=int, help="Evaluate only the first N examples (pilot runs)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Examples per batch; a checkpoint is written after each batch.",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Detector processes (1 = in-process)."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_path = args.data_dir / f"{SPLIT}.jsonl"
    if not data_path.is_file():
        print("error: --data-dir must contain normalized test.jsonl", file=sys.stderr)
        return 2
    options = RunOptions(
        data_path=data_path,
        checkpoint_path=args.checkpoint or args.data_dir / "checkpoints" / "presidio_test.json",
        resume=args.resume,
        restart=args.restart,
        limit=args.limit,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    try:
        result = run_evaluation(options)
    except ValueError as error:
        # CheckpointError is a ValueError: corrupt or incompatible state is never reused.
        print(f"error: {error}", file=sys.stderr)
        return 2
    except EvaluationError as error:
        print(f"error: {error}; last completed batch is checkpointed", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"error: could not read data or write checkpoint: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(
            "interrupted; progress through the last completed batch is checkpointed. "
            "Rerun with --resume to continue.",
            file=sys.stderr,
        )
        return 130

    report = build_report(result)
    write_reports(report, args.report_dir)
    overall = report["overall"]
    session_rate = (
        result.session_examples / result.session_seconds if result.session_seconds else 0.0
    )
    print(
        f"evaluated examples={report['examples_evaluated']} "
        f"complete_split={report['complete_split']} "
        f"precision={overall['precision']:.4f} recall={overall['recall']:.4f} "
        f"f1={overall['f1']:.4f}; session examples={result.session_examples} "
        f"seconds={result.session_seconds:.1f} examples_per_second={session_rate:.1f} "
        f"startup_seconds={result.startup_seconds:.1f}; reports={args.report_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
