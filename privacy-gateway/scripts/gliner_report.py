"""Aggregate-only Markdown rendering for the GLiNER-PII baseline report."""

from __future__ import annotations

from typing import Any

from scripts.gliner_evaluation import UNMAPPED

FN_DESCRIPTIONS = {
    "unmapped_ground_truth_label": "ground-truth label is UNMAPPED in the gateway ontology "
    "(never matchable in this view; see Coverage)",
    "wrong_entity_type": "a scored prediction has the exact span but a different type",
    "wrong_boundaries": "a scored prediction of the same type overlaps with different offsets",
    "overlapping_prediction": "only overlapped by scored predictions of other types and offsets",
    "unmapped_prediction_label": "only overlapped by predictions removed by ontology filtering "
    "(model label maps to UNMAPPED)",
    "missed_entity": "no prediction of any label overlaps it",
}
FP_DESCRIPTIONS = {
    "wrong_entity_type": "exact span of a ground-truth entity of another type (incl. UNMAPPED)",
    "wrong_boundaries": "overlaps a ground-truth entity of the same type, different offsets",
    "overlapping_prediction": "overlaps only ground-truth entities of other types and offsets",
    "false_positive": "overlaps no ground-truth entity",
}


def _f(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _table(header: list[str], rows: list[list[Any]], align: str | None = None) -> list[str]:
    alignment = align or "| --- " + "| ---: " * (len(header) - 1) + "|"
    lines = ["| " + " | ".join(header) + " |", alignment]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return lines


def render_gliner_markdown(report: dict[str, Any]) -> str:
    overall, run, coverage = report["overall"], report["run"], report["coverage"]
    errors, structure, selection = (
        report["error_analysis"],
        report["structure"],
        report["threshold_selection"],
    )
    model, inference, runtime = run["model"], run["inference"], run["runtime"]
    split = report["split"]
    split_name = "test split" if split == "test" else f"{split} subset (train split)"
    scope = (
        f"the complete {split_name}"
        if report["complete_split"]
        else f"the first {report['examples_evaluated']:,} examples of the {split_name} "
        "(partial run)"
    )
    matches_presidio = report.get("dataset_matches_presidio_baseline")
    dataset_note = {
        True: "identical to the Presidio baseline",
        False: "**differs from the Presidio baseline; not comparable**",
        None: "Presidio baseline not found",
    }[matches_presidio]
    lines = [
        f"# NVIDIA GLiNER-PII baseline: Nemotron-PII {split_name}",
        "",
        "Pretrained `nvidia/gliner-PII`, evaluated without any training or fine-tuning, on "
        "the same normalized test file, ground truth, ontology and strict matching as "
        "`presidio_baseline.md`. No source text, entity values, or model inputs are stored.",
        "",
        "## Model",
        "",
        f"- Model: `{model['model_id']}` at revision `{model['model_revision']}` "
        f"(gliner {model['gliner']}, torch {model['torch']}, transformers "
        f"{model['transformers']}).",
        f"- Encoder `{model['encoder']}`, span mode `{model['span_mode']}`, "
        f"{model['parameters']:,} parameters as counted by torch (the model card states "
        f"5.7 × 10^8); inference weights `{model['dtype']}`.",
        f"- Input limit `max_len={model['max_len']}` words; longest predictable span "
        f"`max_width={model['max_width']}` words; `{model['words_splitter_type']}` splitter.",
        "",
        "## Dataset and methodology",
        "",
        f"- Dataset: `nvidia/Nemotron-PII`, normalized `{report['dataset_file']}`; this report "
        f"covers {scope}. SHA-256 `{run['dataset_sha256']}` ({dataset_note}).",
        "- Matching: a prediction is a true positive only if entity type, start and end all "
        "equal a ground-truth entity (half-open offsets); multiset matching, so each "
        "ground-truth entity matches at most one prediction. No overlap or token credit.",
        "- Scoring reuses the Presidio harness (`score_batch`): the `nervaluate` strict "
        "cross-check must reproduce the exact-match count on every 1,000-example batch.",
        "- Ground truth: normalized Nemotron labels via `configs/pii_ontology.yaml`, with DATE "
        "and TIME scored as `DATE_TIME`, exactly as for Presidio. `UNMAPPED` ground truth stays "
        "in the denominator.",
        "- Predictions: every GLiNER label maps through `configs/gliner_label_map.yaml` "
        "(table below). Predictions whose label maps to `UNMAPPED` are removed before gateway "
        "scoring and counted under Coverage; they are not silently discarded.",
        "",
        "## Inference configuration",
        "",
        f"- Prompt: all {coverage['prompt_labels']} native Nemotron-PII labels in one pass "
        "(the model's training vocabulary).",
        f"- `threshold={report['threshold']}`, `flat_ner={inference['flat_ner']}`, "
        f"`multi_label={inference['multi_label']}`, batch size {inference['batch_size']}, "
        f"device `{inference['device_type']}`.",
        f"- Texts over {inference['max_words']} words (the library would silently truncate "
        f"them) are split into word windows overlapping by {2 * inference['chunk_margin_words']} "
        "words; each window keeps predictions starting in its own centre region, so no "
        "predictable entity is cut. Shorter texts are one unchanged call.",
        f"- Chunked examples: {structure['chunked_examples']:,}; model calls (chunks): "
        f"{structure['chunks']:,}.",
        "",
    ]
    lines.extend(_threshold_section(selection))
    lines.extend(
        [
            "## Overall metrics (gateway ontology)",
            "",
            *_table(
                ["Precision", "Recall", "F1", "TP", "FP", "FN", "Support", "Predictions"],
                [
                    [
                        _f(overall["precision"]),
                        _f(overall["recall"]),
                        _f(overall["f1"]),
                        overall["exact_matches"],
                        overall["false_positives"],
                        overall["false_negatives"],
                        overall["support"],
                        overall["predictions"],
                    ]
                ],
                "| ---: " * 8 + "|",
            ),
            "",
            f"Examples evaluated: {report['examples_evaluated']:,}; examples containing PII: "
            f"{report['examples_with_pii']:,}; ground-truth entities: "
            f"{report['ground_truth_entities']:,}; predictions: {report['predictions']:,}; "
            f"exact matches: {report['exact_matches']:,} (nervaluate strict cross-check: "
            f"{report['harness_cross_check']['nervaluate_strict_exact_first']:,}).",
            "",
            "## Per-entity metrics (gateway ontology)",
            "",
            *_table(
                ["Type", "Precision", "Recall", "F1", "TP", "FP", "FN", "Support"],
                [
                    [
                        entity_type,
                        _f(m["precision"]),
                        _f(m["recall"]),
                        _f(m["f1"]),
                        m["exact_matches"],
                        m["false_positives"],
                        m["false_negatives"],
                        m["support"],
                    ]
                    for entity_type, m in report["per_entity"].items()
                ],
            ),
            "",
        ]
    )
    lines.extend(_coverage_section(report))
    lines.extend(_error_section(errors))
    lines.extend(
        [
            "## Structure",
            "",
            *_table(
                ["Slice", "Examples", "Precision", "Recall", "F1"],
                [
                    [
                        f"format={name}",
                        row["examples"],
                        _f(row["precision"]),
                        _f(row["recall"]),
                        _f(row["f1"]),
                    ]
                    for name, row in structure["by_document_format"].items()
                ]
                + [
                    [name, row["examples"], _f(row["precision"]), _f(row["recall"]), _f(row["f1"])]
                    for name, row in structure["by_chunking"].items()
                ],
            ),
            "",
        ]
    )
    lines.extend(_runtime_section(runtime))
    lines.extend(_label_map_section(report))
    lines.extend(
        [
            "## Limitations",
            "",
            "- Nemotron-PII is GLiNER-PII's training source (train split). The test split is "
            "held out and shares no uid or text with train, but it is in-distribution for "
            "GLiNER and out-of-distribution for Presidio; do not read these numbers as "
            "performance on real gateway traffic.",
            "- The threshold was selected on a train-derived subset (in-sample for the model).",
            "- `UNMAPPED` ground truth counts against both detectors; the Coverage section "
            "separates that ontology limit from detection failure.",
            "- Exact spans penalize boundary conventions (e.g. whether a title or trailing "
            "punctuation is included).",
            "",
            "## Reproduce",
            "",
            "```bash",
            "python scripts/gliner_predict.py --data-dir /path/to/nemotron-normalized "
            f"--source test --threshold {report['threshold']} --batch-size "
            f"{inference['batch_size']} --dtype {model['dtype']} --output "
            "/path/to/nemotron-normalized/gliner/test.jsonl --resume",
            "python scripts/evaluate_gliner.py --data-dir /path/to/nemotron-normalized "
            "--predictions /path/to/nemotron-normalized/gliner/test.jsonl "
            "--selection /path/to/nemotron-normalized/gliner/threshold_selection.json",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _threshold_section(selection: dict[str, Any]) -> list[str]:
    if not selection:
        return ["## Threshold selection", "", "Not recorded for this run.", ""]
    source = selection["source"]
    rows = [
        [
            threshold,
            _f(row["gateway"]["precision"]),
            _f(row["gateway"]["recall"]),
            _f(row["gateway"]["f1"]),
            _f(row["native"]["f1"]),
            "**selected**" if float(threshold) == selection["selected_threshold"] else "",
        ]
        for threshold, row in selection["results"].items()
    ]
    lines = [
        "## Threshold selection (development subset only)",
        "",
        f"- Source: {source['selection']['size']:,} records of Nemotron-PII **train** "
        f"(`{source['file_name']}`, SHA-256 `{source['sha256'][:16]}…`), chosen by ranking "
        f'indices by SHA-256 of `"{source["selection"]["seed"]}:<index>"`. The test split '
        "was not read during selection.",
        f"- Rule (fixed before scoring): {selection['rule']}.",
        "- One dev inference pass at the lowest candidate; higher thresholds are the same "
        "predictions filtered by `score > t`, which equals a direct run because the flat-NER "
        "decoder is greedy in descending score order.",
    ]
    verification = selection.get("post_filter_verification")
    if verification:
        lines.append(
            f"- Verified against a direct run at t={verification['direct_threshold']}: "
            f"{verification['examples_identical']}/{verification['examples_compared']} "
            "examples identical "
            f"({verification['predictions_only_in_post_filtered']} / "
            f"{verification['predictions_only_in_direct_run']} predictions differ)."
        )
    lines.extend(
        [
            f"- Caveat: {selection['caveat']}",
            "- NVIDIA's model card reports evaluating at "
            f"t={selection['reference_threshold_model_card']}.",
            "",
            *_table(["t", "Dev precision", "Dev recall", "Dev F1", "Dev native F1", ""], rows),
            "",
            f"Selected threshold: **{selection['selected_threshold']}**. The final test split "
            "was then run once at this threshold. The dev curve is nearly flat and still rising "
            "at the grid's upper edge; because dev records are in-sample (the model is very "
            "confident on them), this procedure likely favours a higher threshold than a "
            "held-out dev set would.",
            "",
        ]
    )
    decision = selection.get("configuration_decision")
    if decision:
        chosen = decision["selected"]
        pilot = decision["pilot_benchmark_64_dev_records"]
        lines.extend(
            [
                "### Inference configuration pilot (development records only)",
                "",
                f"Chosen: `{chosen['device']}`, `{chosen['dtype']}`, batch size "
                f"{chosen['batch_size']}, {chosen['processes']} process. {decision['reason']}",
                "",
                *_table(
                    ["Device:dtype:batch", "Examples/s", "GPU peak MiB", "Identical texts"],
                    [
                        [
                            f"`{name}`",
                            row["examples_per_second"],
                            row["gpu_peak_reserved_mib"] or "n/a",
                            f"{row['identical_texts_vs_fp32_bs1']}/{row['records']}",
                        ]
                        for name, row in pilot.items()
                    ],
                    "| --- | ---: | ---: | ---: |",
                ),
                "",
                "fp16 sensitivity on the same 2,000 dev records (not used for the test run):",
                "",
                *_table(
                    ["t", "fp32 dev F1", "fp16 dev F1"],
                    [
                        [t, _f(selection["results"][t]["gateway"]["f1"]), _f(row["f1"])]
                        for t, row in decision["dev_sensitivity_fp16_bs4"].items()
                    ],
                ),
                "",
            ]
        )
    return lines


def _coverage_section(report: dict[str, Any]) -> list[str]:
    coverage = report["coverage"]
    native = coverage["native_label_overall"]
    unmapped_truth = coverage["unmapped_ground_truth"]
    total_unmapped = unmapped_truth.get("total", 0) or 1
    never = ", ".join(f"`{label}`" for label in coverage["prompt_labels_never_predicted"])
    lines = [
        "## Coverage before ontology filtering",
        "",
        "The gateway view above cannot credit a detection whose label the gateway ontology "
        "does not have. This section scores the model in its **native** label space (GLiNER "
        "label vs raw Nemotron label, same exact span rule) to separate detection failure "
        "from ontology mismatch. These numbers are not comparable to the gateway metrics and "
        "are not mixed into them.",
        "",
        f"- Prompt labels: {coverage['prompt_labels']}; labels the model actually returned: "
        f"{coverage['model_labels_detected']}; never returned: "
        f"{never or 'none'}.",
        f"- Predictions before ontology filtering: "
        f"{coverage['predictions_before_ontology_filtering']:,}; removed by ontology filtering "
        f"(label maps to UNMAPPED): {coverage['predictions_removed_by_ontology_filtering']:,}, "
        f"of which {coverage['removed_predictions_that_exactly_match_native_ground_truth']:,} "
        "exactly match a native ground-truth entity (correct detections the ontology drops).",
        f"- Gateway-mapped labels returned: {len(coverage['gateway_mapped_labels'])}; unmapped "
        f"labels returned: {len(coverage['unmapped_labels'])}.",
        "",
        "Native-label exact match (all 55 labels):",
        "",
        *_table(
            ["Precision", "Recall", "F1", "TP", "FP", "FN"],
            [
                [
                    _f(native["precision"]),
                    _f(native["recall"]),
                    _f(native["f1"]),
                    native["tp"],
                    native["fp"],
                    native["fn"],
                ]
            ],
            "| ---: " * 6 + "|",
        ),
        "",
        f"`UNMAPPED` ground truth ({unmapped_truth.get('total', 0):,} entities) as seen by the "
        "model:",
        "",
        *_table(
            ["Outcome", "Entities", "Share"],
            [
                [key.replace("_", " "), value, f"{value / total_unmapped:.1%}"]
                for key, value in unmapped_truth.items()
                if key != "total"
            ],
        ),
        "",
        "Per native label (gateway type in brackets):",
        "",
        *_table(
            ["Native label", "Gateway", "Precision", "Recall", "F1", "TP", "FP", "FN", "Support"],
            [
                [
                    label,
                    row["gateway_type"],
                    _f(row["precision"]),
                    _f(row["recall"]),
                    _f(row["f1"]),
                    row["exact_matches"],
                    row["false_positives"],
                    row["false_negatives"],
                    row["support"],
                ]
                for label, row in coverage["native_label_per_entity"].items()
            ],
            "| --- | --- " + "| ---: " * 7 + "|",
        ),
        "",
    ]
    return lines


def _error_section(errors: dict[str, Any]) -> list[str]:
    fn, fp = errors["false_negative_categories"], errors["false_positive_categories"]
    fn_total, fp_total = sum(fn.values()) or 1, sum(fp.values()) or 1
    unreachable = errors["structurally_unreachable_ground_truth"]
    unreachable_fn = errors["structurally_unreachable_false_negatives"]
    lines = [
        "## Error analysis (gateway ontology)",
        "",
        "Each unmatched ground-truth entity (false negative) and each unmatched prediction "
        "(false positive) gets exactly one category, checked in the order listed.",
        "",
        *_table(
            ["False-negative category", "Count", "Share", "Meaning"],
            [[k, v, f"{v / fn_total:.1%}", FN_DESCRIPTIONS[k]] for k, v in fn.items()],
            "| --- | ---: | ---: | --- |",
        ),
        "",
        *_table(
            ["False-positive category", "Count", "Share", "Meaning"],
            [[k, v, f"{v / fp_total:.1%}", FP_DESCRIPTIONS[k]] for k, v in fp.items()],
            "| --- | ---: | ---: | --- |",
        ),
        "",
        f"- Overlapping false positives that are nested in / contain a ground-truth span: "
        f"{errors['false_positive_overlaps_that_are_nested']:,}.",
        f"- Prediction pairs overlapping each other: "
        f"{errors['predictions_overlapping_each_other_pairs']:,} (flat-NER decoding forbids "
        "overlap within one model call, so only chunk-window seams could produce any). "
        "Duplicate "
        f"predictions removed: {errors['duplicate_predictions_removed']:,}.",
        "- Structurally unreachable ground truth (no whole-word span of at most `max_width` "
        f"words can equal it): {json_counts(unreachable)}; of these still false negatives: "
        f"{json_counts(unreachable_fn)}.",
        "",
        "Structural reachability by native label (labels with any unreachable ground truth; "
        "native exact-match misses in brackets):",
        "",
        *_table(
            [
                "Native label",
                "Reachable (missed)",
                "Wider than max_width (missed)",
                "Misaligned boundary (missed)",
            ],
            [
                [
                    f"`{label}`",
                    f"{c.get('reachable', 0):,} ({c.get('reachable_missed', 0):,})",
                    f"{c.get('wider_than_max_width', 0):,} "
                    f"({c.get('wider_than_max_width_missed', 0):,})",
                    f"{c.get('misaligned_word_boundary', 0):,} "
                    f"({c.get('misaligned_word_boundary_missed', 0):,})",
                ]
                for label, c in errors.get("reachability_by_native_label", {}).items()
            ],
            "| --- | ---: | ---: | ---: |",
        ),
        "",
        "Native labels predicted over unmatched gateway entities (top 5 per type; shows what "
        "took the span instead, e.g. name parts inside an email address):",
        "",
        *_table(
            ["Type", "Overlapping native labels (count)"],
            [
                [entity_type, ", ".join(f"`{k}` {v:,}" for k, v in counts.items())]
                for entity_type, counts in errors.get(
                    "false_negative_overlapping_native_labels_by_type", {}
                ).items()
            ],
            "| --- | --- |",
        ),
        "",
        "False-negative categories by gateway type:",
        "",
    ]
    categories = list(FN_DESCRIPTIONS)
    lines.extend(
        _table(
            ["Type", *categories],
            [
                [entity_type, *[counts.get(c, 0) for c in categories]]
                for entity_type, counts in errors["false_negative_categories_by_type"].items()
            ],
        )
    )
    fp_categories = list(FP_DESCRIPTIONS)
    lines.extend(["", "False-positive categories by gateway type:", ""])
    lines.extend(
        _table(
            ["Type", *fp_categories],
            [
                [entity_type, *[counts.get(c, 0) for c in fp_categories]]
                for entity_type, counts in errors["false_positive_categories_by_type"].items()
            ],
        )
    )
    lines.extend(["", "Offset-only samples (first per category; uid is Nemotron's):", ""])
    sample_rows = [
        [category, s["uid"], s["example_index"], s["entity_type"], s["start"], s["end"]]
        for category, samples in errors["samples"].items()
        for s in samples[:2]
    ]
    lines.extend(
        _table(
            ["Category", "uid", "Index", "Type", "Start", "End"],
            sample_rows,
            "| --- | --- " + "| ---: " + "| --- " + "| ---: " * 2 + "|",
        )
    )
    lines.append("")
    return lines


def json_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{key} {value:,}" for key, value in counts.items()) or "none"


def _runtime_section(runtime: dict[str, Any]) -> list[str]:
    if not runtime:
        return ["## Runtime", "", "No runtime log was found for this predictions file.", ""]
    sessions = runtime["sessions"]
    hardware = sessions[-1]["hardware"]
    rows = [
        [
            index + 1,
            session["resumed_from"],
            session["examples"],
            session["chunks"],
            session["batch_size"],
            session["model_load_seconds"],
            session["inference_seconds"],
            session.get("examples_per_second"),
            session.get("per_example_latency_ms_mean"),
            session.get("gpu_peak_reserved_mib", "n/a"),
            session.get("resources", {}).get("gpu_util_percent_mean", "n/a"),
            session.get("resources", {}).get("process_cpu_percent_mean", "n/a"),
            session.get("resources", {}).get("rss_mib_max", "n/a"),
        ]
        for index, session in enumerate(sessions)
    ]
    return [
        "## Runtime",
        "",
        f"- Hardware: {hardware.get('gpu', 'CPU only')} "
        f"({hardware.get('gpu_memory_gib', 'n/a')} GiB, CUDA {hardware.get('cuda_runtime')}, "
        f"driver {hardware.get('gpu_driver')}); CPU {hardware['cpu']} "
        f"({hardware['cpu_physical_cores']} cores / {hardware['cpu_logical_cores']} threads), "
        f"{hardware['ram_gib']} GiB RAM; {hardware['platform']}, Python {hardware['python']}.",
        "- Workers: one process, one model instance; batching inside `GLiNER.inference` "
        "(length-sorted batches).",
        f"- Model load time (sum over sessions): {runtime['model_load_seconds_total']} s; "
        f"inference time: {runtime['inference_seconds']:,} s; total runtime: "
        f"{runtime['total_runtime_seconds']:,} s; throughput "
        f"{runtime['examples_per_second']} examples/s over {runtime['examples']:,} examples "
        f"({runtime['chunks']:,} model calls).",
        "",
        *_table(
            [
                "Session",
                "From",
                "Examples",
                "Chunks",
                "Batch",
                "Load s",
                "Inference s",
                "Ex/s",
                "ms/example",
                "GPU peak MiB",
                "GPU util %",
                "CPU %",
                "RSS MiB",
            ],
            rows,
            "| ---: " * 13 + "|",
        ),
        "",
    ]


def _label_map_section(report: dict[str, Any]) -> list[str]:
    mapping = report["label_map"]["mapping"]
    detected = report["coverage"]["predictions_by_model_label"]
    deviations = report["label_map"]["deviations_from_ground_truth_ontology"]
    rows = [
        [
            f"`{label}`",
            target,
            "same as ground-truth ontology" if label not in deviations else deviations[label],
            "filtered (not scored)" if target == UNMAPPED else "scored",
            detected.get(label, 0),
        ]
        for label, target in mapping.items()
    ]
    return [
        "## GLiNER label → gateway ontology mapping",
        "",
        "GLiNER-PII's labels are the native Nemotron-PII labels, so each maps to the same "
        "gateway type the ground truth uses (`configs/pii_ontology.yaml`); the loader rejects "
        "any undocumented divergence. DATE/TIME targets are scored as `DATE_TIME` on both "
        "sides.",
        "",
        *_table(
            ["GLiNER label", "Gateway type", "Basis", "Gateway scoring", "Predictions"],
            rows,
            "| --- | --- | --- | --- | ---: |",
        ),
        "",
    ]
