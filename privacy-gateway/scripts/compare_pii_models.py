"""Side-by-side, aggregate-only comparison of the Presidio and GLiNER-PII baselines.

Reads ``reports/pii/presidio_baseline.json`` and ``reports/pii/gliner_baseline.json``
and writes ``reports/pii/model_comparison.md``. It refuses to compare reports built
on different dataset files or matching rules. It deliberately names no overall
winner: the output is per-entity strengths, coverage, cost, and failure modes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPORT_DIRECTORY = Path(__file__).resolve().parents[1] / "reports" / "pii"
# A per-type F1 gap below this is reported as "similar" rather than as a strength.
MATERIAL_F1_GAP = 0.05
PRESIDIO_MODEL = {
    "summary": "rule/regex recognizers + spaCy `en_core_web_sm` 3.8.0 NER",
    "size": "~15 MB spaCy model + ~2.4 MB presidio-analyzer package; no GPU",
}


def _f(value: float) -> str:
    return f"{value:.4f}"


def _delta(value: float) -> str:
    return f"{value:+.4f}"


def check_comparable(presidio: dict[str, Any], gliner: dict[str, Any]) -> None:
    if presidio["run"]["dataset_sha256"] != gliner["run"]["dataset_sha256"]:
        raise ValueError("reports were computed on different normalized test files")
    for key in ("dataset", "split", "matching", "harness", "evaluation_type_map"):
        if presidio[key] != gliner[key]:
            raise ValueError(f"reports differ in {key}; they are not comparable")
    if presidio["ground_truth_entities"] != gliner["ground_truth_entities"]:
        raise ValueError("reports have different ground-truth entity counts")
    if not (presidio["complete_split"] and gliner["complete_split"]):
        raise ValueError("both reports must cover the complete test split")


def render(presidio: dict[str, Any], gliner: dict[str, Any]) -> str:
    check_comparable(presidio, gliner)
    p_overall, g_overall = presidio["overall"], gliner["overall"]
    types = sorted(set(presidio["per_entity"]) | set(gliner["per_entity"]))
    empty = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "support": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "exact_matches": 0,
        "predictions": 0,
    }
    rows, strengths_g, strengths_p, similar = [], [], [], []
    for entity_type in types:
        p = presidio["per_entity"].get(entity_type, empty)
        g = gliner["per_entity"].get(entity_type, empty)
        gap = g["f1"] - p["f1"]
        rows.append(
            f"| {entity_type} | {max(p['support'], g['support']):,} | {_f(p['precision'])} | "
            f"{_f(p['recall'])} | {_f(p['f1'])} | {_f(g['precision'])} | {_f(g['recall'])} | "
            f"{_f(g['f1'])} | {_delta(gap)} |"
        )
        if entity_type == "UNMAPPED":
            continue
        if gap >= MATERIAL_F1_GAP:
            strengths_g.append(entity_type)
        elif gap <= -MATERIAL_F1_GAP:
            strengths_p.append(entity_type)
        else:
            similar.append(entity_type)
    p_run, g_run = presidio["run"], gliner["run"]
    g_runtime, g_model = g_run["runtime"], g_run["model"]
    g_hardware = g_runtime["sessions"][-1]["hardware"]
    p_workers = sorted({s["workers"] for s in p_run["sessions"]})
    coverage = gliner["coverage"]
    unmapped_support = gliner["per_entity"].get("UNMAPPED", empty)["support"]
    total_support = g_overall["support"]
    reachable_support = total_support - unmapped_support
    p_scored = [t for t in types if presidio["per_entity"].get(t, empty)["predictions"]]
    g_scored = [t for t in types if gliner["per_entity"].get(t, empty)["predictions"]]
    unmapped_truth = coverage["unmapped_ground_truth"]
    groups = sorted(set(presidio["false_negative_groups"]) | set(gliner["false_negative_groups"]))
    fp_groups = sorted(
        set(presidio["false_positive_groups"]) | set(gliner["false_positive_groups"])
    )
    g_errors = gliner["error_analysis"]

    def recall_excl_unmapped(report: dict[str, Any]) -> float:
        return report["overall"]["exact_matches"] / reachable_support if reachable_support else 0.0

    lines = [
        "# PII detector comparison: Presidio vs NVIDIA GLiNER-PII",
        "",
        "Both detectors were scored on the **same** normalized Nemotron-PII test file "
        f"(SHA-256 `{g_run['dataset_sha256']}`, {gliner['examples_evaluated']:,} examples, "
        f"{total_support:,} ground-truth entities), with the same ontology mapping, the same "
        "strict rule (type, start and end must all match; each ground-truth entity matches at "
        "most one prediction) and the same `nervaluate` strict cross-check. Neither model was "
        "trained or tuned on the test split. Sources: `presidio_baseline.json`, "
        "`gliner_baseline.json`.",
        "",
        "This comparison deliberately names no overall winner. A single aggregate hides that "
        "the two detectors cover different entity types, cost very different amounts to run, "
        "and fail in different ways.",
        "",
        "## Overall (gateway ontology)",
        "",
        "| Detector | Precision | Recall | F1 | TP | FP | FN | Predictions |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| Presidio (Phase 1) | {_f(p_overall['precision'])} | {_f(p_overall['recall'])} | "
        f"{_f(p_overall['f1'])} | {p_overall['exact_matches']:,} | "
        f"{p_overall['false_positives']:,} | {p_overall['false_negatives']:,} | "
        f"{p_overall['predictions']:,} |",
        f"| GLiNER-PII (t={gliner['threshold']}) | {_f(g_overall['precision'])} | "
        f"{_f(g_overall['recall'])} | {_f(g_overall['f1'])} | {g_overall['exact_matches']:,} | "
        f"{g_overall['false_positives']:,} | {g_overall['false_negatives']:,} | "
        f"{g_overall['predictions']:,} |",
        "",
        f"`UNMAPPED` ground truth ({unmapped_support:,} entities, "
        f"{unmapped_support / total_support:.1%} of support) counts as a false negative for "
        "both detectors because the gateway ontology has no type for it. Recall over the "
        f"{reachable_support:,} entities that have a gateway type: Presidio "
        f"{_f(recall_excl_unmapped(presidio))}, GLiNER-PII {_f(recall_excl_unmapped(gliner))} "
        "(reported for context only; not a substitute for the headline metric).",
        "",
        "## Per entity type",
        "",
        "| Type | Support | Presidio P | Presidio R | Presidio F1 | GLiNER P | GLiNER R | "
        "GLiNER F1 | F1 Δ (GLiNER − Presidio) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *rows,
        "",
        f"- GLiNER-PII materially higher F1 (Δ ≥ {MATERIAL_F1_GAP}): "
        f"{', '.join(strengths_g) or 'none'}.",
        f"- Presidio materially higher F1 (Δ ≤ −{MATERIAL_F1_GAP}): "
        f"{', '.join(strengths_p) or 'none'}.",
        f"- Within ±{MATERIAL_F1_GAP}: {', '.join(similar) or 'none'}.",
        "",
        "## Coverage",
        "",
        "| | Presidio | GLiNER-PII |",
        "| --- | --- | --- |",
        f"| Gateway types it can emit | {len(p_scored)}: {', '.join(p_scored)} | "
        f"{len(g_scored)}: {', '.join(g_scored)} |",
        f"| Native label space | Phase 1 `EntityType` (7 types) | "
        f"{coverage['prompt_labels']} Nemotron-PII labels; {coverage['model_labels_detected']} "
        "returned |",
        f"| Predictions dropped by ontology filtering | n/a (Phase 1 types only) | "
        f"{coverage['predictions_removed_by_ontology_filtering']:,} "
        f"({coverage['removed_predictions_that_exactly_match_native_ground_truth']:,} were "
        "exact native matches) |",
        f"| `UNMAPPED` ground truth detected exactly (native label) | 0 (no such labels) | "
        f"{unmapped_truth.get('exact_native_match', 0):,} of {unmapped_truth.get('total', 0):,} |",
        f"| Native-label strict F1 (all labels, pre-ontology) | n/a | "
        f"{_f(coverage['native_label_overall']['f1'])} |",
        "",
        "Presidio's zero scores for ADDRESS, BANK_ACCOUNT, ID_CARD, ORGANIZATION, "
        "SOCIAL_IDENTIFIER and USERNAME are ontology/recognizer gaps (Phase 1 never emits those "
        "types), not measured detection failures. GLiNER-PII's `UNMAPPED` losses are the "
        "gateway ontology's gap: the model often detects those spans, but they have no gateway "
        "type to be scored as.",
        "",
        "## Runtime and model size",
        "",
        "| | Presidio | GLiNER-PII |",
        "| --- | --- | --- |",
        f"| Model | {PRESIDIO_MODEL['summary']} | `{g_model['model_id']}` @ "
        f"`{g_model['model_revision'][:12]}`, {g_model['encoder']} |",
        f"| Size | {PRESIDIO_MODEL['size']} | {g_model['parameters']:,} parameters; 1.78 GB fp32 "
        f"checkpoint; run as `{g_model['dtype']}` |",
        f"| Hardware | CPU, {', '.join(map(str, p_workers))} worker processes | "
        f"{g_hardware.get('gpu', 'CPU')} ({g_hardware.get('gpu_memory_gib')} GiB), 1 process |",
        f"| Detection time (100k) | {p_run['elapsed_seconds']:,} s | "
        f"{g_runtime['inference_seconds']:,} s (+ {g_runtime['model_load_seconds_total']} s "
        "model load) |",
        f"| Throughput | {p_run['examples_per_second']} examples/s | "
        f"{g_runtime['examples_per_second']} examples/s |",
        "",
        "Both runs used the same laptop (12th Gen Intel Core i5-12450H). GLiNER-PII needs a GPU "
        "to be practical: on this machine's CPU it measured 0.19 examples/s in the pilot.",
        "",
        "## Failure modes",
        "",
        "Presidio-harness categories (identical definitions for both detectors):",
        "",
        "| False-negative group | Presidio | GLiNER-PII |",
        "| --- | ---: | ---: |",
        *[
            f"| {group} | {presidio['false_negative_groups'].get(group, 0):,} | "
            f"{gliner['false_negative_groups'].get(group, 0):,} |"
            for group in groups
        ],
        "",
        "| False-positive group | Presidio | GLiNER-PII |",
        "| --- | ---: | ---: |",
        *[
            f"| {group} | {presidio['false_positive_groups'].get(group, 0):,} | "
            f"{gliner['false_positive_groups'].get(group, 0):,} |"
            for group in fp_groups
        ],
        "",
        "Group meanings: `span_mismatch` = same type, overlapping but different offsets; "
        "`unmapped_source_label` = UNMAPPED ground truth; `unsupported_entity` = a gateway type "
        "Phase 1 does not implement (a Phase 1 label, independent of detector); "
        "`supported_direct` = other misses of Phase 1 types; `detector_recognizer` = "
        "predictions that do not overlap a same-type entity.",
        "",
        "GLiNER-PII finer taxonomy (see `gliner_baseline.md`): "
        + ", ".join(
            f"{name} {count:,}" for name, count in g_errors["false_negative_categories"].items()
        )
        + " (false negatives); "
        + ", ".join(
            f"{name} {count:,}" for name, count in g_errors["false_positive_categories"].items()
        )
        + " (false positives).",
        "",
        "## Implications for a future hybrid detector",
        "",
        *hybrid_observations(presidio, gliner),
        "",
    ]
    return "\n".join(lines)


def _top(counts: dict[str, int], n: int = 3) -> str:
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:n]
    return ", ".join(f"`{k}` {v:,}" for k, v in ranked) or "none"


def hybrid_observations(presidio: dict[str, Any], gliner: dict[str, Any]) -> list[str]:
    """Evidence-backed bullets; every number comes from the two reports."""
    p_types, g_types = presidio["per_entity"], gliner["per_entity"]
    errors = gliner["error_analysis"]
    overlapping = errors.get("false_negative_overlapping_native_labels_by_type", {})
    reach = errors.get("reachability_by_native_label", {})
    unmapped = gliner["coverage"]["unmapped_ground_truth"]
    lines = [
        "- Route or union by entity type rather than choosing one model; the per-type table "
        "shows where each is materially stronger.",
    ]
    for entity_type in ("EMAIL_ADDRESS", "IP_ADDRESS"):
        p, g = p_types.get(entity_type), g_types.get(entity_type)
        if not p or not g:
            continue
        lines.append(
            f"- {entity_type}: Presidio F1 {_f(p['f1'])} (recall {_f(p['recall'])}) vs "
            f"GLiNER-PII F1 {_f(g['f1'])} (recall {_f(g['recall'])}, precision "
            f"{_f(g['precision'])}). Native labels GLiNER predicted over its misses: "
            f"{_top(overlapping.get(entity_type, {}))}."
        )
    for label in ("email", "ipv6", "url", "http_cookie"):
        counts = reach.get(label)
        if not counts:
            continue
        wide = counts.get("wider_than_max_width", 0)
        total = wide + counts.get("reachable", 0) + counts.get("misaligned_word_boundary", 0)
        reachable, reachable_missed = counts.get("reachable", 0), counts.get("reachable_missed", 0)
        if total and wide / total >= 0.1:
            lines.append(
                f"- `{label}`: {wide:,} of {total:,} ground-truth spans are wider than GLiNER's "
                f"12-word `max_width` and all {counts.get('wider_than_max_width_missed', 0):,} "
                f"are missed, while only {reachable_missed:,} of {reachable:,} reachable spans "
                "are missed: a hard architectural limit that a pattern recognizer does not have."
            )
        elif reachable_missed:
            lines.append(
                f"- `{label}`: spans are almost always short enough for GLiNER ({wide:,} too "
                f"wide), yet {reachable_missed:,} of {reachable:,} are missed; the overlap labels "
                "above show flat-NER decoding keeping higher-scoring sub-spans inside the value "
                "instead. A pattern recognizer run before the model would avoid this."
            )
    person = g_types.get("PERSON")
    person_fp = errors["false_positive_categories_by_type"].get("PERSON", {})
    if person:
        lines.append(
            f"- PERSON: GLiNER-PII recall {_f(person['recall'])}, precision "
            f"{_f(person['precision'])}; {person_fp.get('overlapping_prediction', 0):,} of its "
            f"{person['false_positives']:,} false positives overlap a ground-truth entity of "
            "another type (e.g. name parts inside email addresses)."
        )
    lines.extend(
        [
            "- Types Phase 1 Presidio cannot emit at all (ADDRESS, BANK_ACCOUNT, ID_CARD, "
            "ORGANIZATION, SOCIAL_IDENTIFIER, USERNAME) are where a learned detector adds "
            "coverage rather than accuracy.",
            f"- `UNMAPPED` is an ontology gap: GLiNER-PII exactly matched "
            f"{unmapped.get('exact_native_match', 0):,} of {unmapped.get('total', 0):,} such "
            "entities under their native labels; extending the gateway ontology, not changing "
            "models, is what would let a detector be credited for them.",
            "- Caveat: Nemotron-PII is GLiNER-PII's training distribution (train split; the test "
            "split is held out and shares no uid/text with train), so this benchmark favours "
            "GLiNER relative to real gateway traffic. A hybrid decision needs an "
            "out-of-distribution evaluation as well.",
        ]
    )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIRECTORY)
    args = parser.parse_args(argv)
    presidio = json.loads((args.report_dir / "presidio_baseline.json").read_text("utf-8"))
    gliner = json.loads((args.report_dir / "gliner_baseline.json").read_text("utf-8"))
    try:
        markdown = render(presidio, gliner)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    (args.report_dir / "model_comparison.md").write_text(markdown, encoding="utf-8")
    print(f"wrote {args.report_dir / 'model_comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
