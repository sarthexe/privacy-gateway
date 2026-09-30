"""Inference-configuration pilot for GLiNER-PII on development records only.

Measures throughput, latency, and memory for device / dtype / batch-size
combinations, and how far each configuration's predictions differ from the
reference (fp32, batch size 1: the unbatched ``predict_entities`` path). Only
aggregate numbers are written; records come from the train-derived dev subset,
never the test split.

    python scripts/gliner_benchmark.py --data-dir D --output D/gliner/benchmark.json \
        --records 200 --configs cuda:fp32:1 cuda:fp32:8 cuda:fp16:8 cpu:fp32:4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.gliner_evaluation import (
    DEFAULT_LABEL_MAP,
    GlinerPredictor,
    InferenceConfig,
    RawPrediction,
    load_label_map,
)
from scripts.gliner_predict import (
    ResourceSampler,
    describe_hardware,
    describe_model,
    iter_texts,
    load_model,
    resolve_source,
    word_splitter,
    write_json_atomic,
)
from scripts.nemotron_pii import is_within_repository

REFERENCE = "cuda:fp32:1"


def agreement(
    reference: list[list[RawPrediction]], candidate: list[list[RawPrediction]], tolerance: float
) -> dict[str, Any]:
    """Compare label/offset sets per text, and score drift on shared predictions."""
    identical = only_reference = only_candidate = shared = 0
    max_score_delta = 0.0
    for left, right in zip(reference, candidate, strict=True):
        left_keys = {(p.label, p.start, p.end): p.score for p in left}
        right_keys = {(p.label, p.start, p.end): p.score for p in right}
        identical += left_keys.keys() == right_keys.keys()
        only_reference += len(left_keys.keys() - right_keys.keys())
        only_candidate += len(right_keys.keys() - left_keys.keys())
        for key in left_keys.keys() & right_keys.keys():
            shared += 1
            max_score_delta = max(max_score_delta, abs(left_keys[key] - right_keys[key]))
    return {
        "texts_with_identical_predictions": identical,
        "texts": len(reference),
        "predictions_only_in_reference": only_reference,
        "predictions_only_in_candidate": only_candidate,
        "shared_predictions": shared,
        "max_score_delta": round(max_score_delta, 6),
        "within_tolerance": only_reference == only_candidate == 0 and max_score_delta <= tolerance,
    }


def parse_config(value: str) -> tuple[str, str, int]:
    device, dtype, batch = value.split(":")
    return device, dtype, int(batch)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--configs", nargs="+", default=[REFERENCE])
    parser.add_argument(
        "--cpu-records", type=int, default=20, help="Smaller sample for slow CPU configs."
    )
    parser.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP)
    args = parser.parse_args(argv)
    if is_within_repository(args.output):
        print("error: --output must be outside the repository", file=sys.stderr)
        return 2
    warnings.filterwarnings("ignore", message="Sentence of length")
    label_map = load_label_map(args.label_map)
    source = resolve_source(args.data_dir, "dev", 2_000)
    texts = [text for _, text in iter_texts(source, 0, args.records)]
    configs = [REFERENCE, *[c for c in args.configs if c != REFERENCE]]

    import torch

    results: dict[str, Any] = {}
    reference: list[list[RawPrediction]] | None = None
    loaded: dict[tuple[str, str], Any] = {}
    for name in configs:
        device, dtype, batch_size = parse_config(name)
        sample = texts if device == "cuda" else texts[: args.cpu_records]
        loaded.clear()  # keep one model resident: the GPU has little memory
        if device == "cuda":
            torch.cuda.empty_cache()
        model, load_seconds = load_model(label_map, device, dtype)
        identity = describe_model(model, label_map, dtype)
        config = InferenceConfig(
            threshold=args.threshold,
            batch_size=batch_size,
            max_words=int(identity["max_len"]),
            margin=int(identity["max_width"]),
        )
        predictor = GlinerPredictor(model, label_map, word_splitter(model), config)
        predictor.predict(sample[:2])  # warm-up (CUDA context, kernels)
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        with ResourceSampler(interval=1.0, gpu=device == "cuda") as sampler:
            started = time.perf_counter()
            outputs = predictor.predict(sample)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
        predictions = [output.predictions for output in outputs]
        row: dict[str, Any] = {
            "device": device,
            "dtype": dtype,
            "batch_size": batch_size,
            "records": len(sample),
            "chunks": sum(output.chunks for output in outputs),
            "model_load_seconds": round(load_seconds, 2),
            "seconds": round(elapsed, 2),
            "examples_per_second": round(len(sample) / elapsed, 3),
            "resources": sampler.summary(),
            "hardware": describe_hardware(device),
        }
        if device == "cuda":
            row["gpu_peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
            row["gpu_peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20)
        if reference is None:
            reference = predictions
        else:
            row["agreement_with_reference"] = agreement(
                reference[: len(sample)], predictions, tolerance=1e-3
            )
        results[name] = row
        print(
            f"{name}: examples_per_second={row['examples_per_second']} "
            f"peak_reserved_mib={row.get('gpu_peak_reserved_mib')} "
            f"agreement={json.dumps(row.get('agreement_with_reference'))}",
            flush=True,
        )
        del model, predictor
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        args.output,
        {"source": source.identity(args.records), "threshold": args.threshold, "results": results},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
