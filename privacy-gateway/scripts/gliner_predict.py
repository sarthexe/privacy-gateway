"""Resumable, offline GLiNER-PII inference on normalized Nemotron-PII records.

Writes offset/label/score predictions only (see ``scripts/gliner_predictions.py``)
to a file outside the repository; scoring is a separate step
(``scripts/evaluate_gliner.py``). Runtime statistics go to ``<output>.runtime.json``.
No source text, model input, or entity value is logged or persisted.

    # Development subset (train split; threshold selection only):
    python scripts/gliner_predict.py --data-dir D --source dev --threshold 0.2 \
        --output D/gliner/dev_t0.2.jsonl
    # Test split pilot, then resume to the full split with the same settings:
    python scripts/gliner_predict.py --data-dir D --source test --threshold T \
        --output D/gliner/test.jsonl --limit 1000
    python scripts/gliner_predict.py ... --resume
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluation_checkpoint import sha256_file
from scripts.gliner_evaluation import (
    DEFAULT_LABEL_MAP,
    GlinerLabelMap,
    GlinerPredictor,
    InferenceConfig,
    load_label_map,
)
from scripts.gliner_predictions import (
    DEV_SELECTION_SEED,
    PredictionRecord,
    PredictionsError,
    PredictionWriter,
    select_dev_indices,
)
from scripts.nemotron_pii import DATASET_ID, is_within_repository

DEV_SUBSET_SIZE = 2_000
DEFAULT_READ_BATCH = 256
# nvidia/gliner-PII's own gliner_config.json value; runs at other widths are overrides.
DEFAULT_MAX_WIDTH = 12


@dataclass(frozen=True)
class Source:
    """Which normalized file and which of its records are evaluated."""

    name: str
    path: Path
    indices: list[int]

    def identity(self, limit: int | None) -> dict[str, Any]:
        selection: dict[str, Any] = {"kind": "all"}
        if self.name == "dev":
            selection = {
                "kind": "keyed_sha256_rank",
                "seed": DEV_SELECTION_SEED,
                "size": len(self.indices),
            }
        return {
            "dataset_id": DATASET_ID,
            "source": self.name,
            "file_name": self.path.name,
            "sha256": sha256_file(self.path),
            "selection": selection,
            "limit": limit,
        }


def resolve_source(data_dir: Path, name: str, dev_size: int) -> Source:
    if name == "test":
        path = data_dir / "test.jsonl"
    elif name == "dev":
        # Never the test split: the development subset is drawn from train.
        path = data_dir / "train.jsonl"
    else:
        raise ValueError("source must be 'test' or 'dev'")
    if not path.is_file():
        raise FileNotFoundError(f"normalized {path.name} not found in --data-dir")
    with path.open("rb") as handle:
        total = sum(1 for _ in handle)
    indices = list(range(total)) if name == "test" else select_dev_indices(total, dev_size)
    return Source(name, path, indices)


def iter_texts(source: Source, skip: int, limit: int | None) -> Iterator[tuple[int, str]]:
    """Yield ``(example_index, text)`` for the selected records after ``skip``."""
    wanted = source.indices[skip : limit if limit is not None else None]
    if not wanted:
        return
    targets = iter(wanted)
    target = next(targets)
    with source.path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index != target:
                continue
            text = json.loads(line).get("text")
            if not isinstance(text, str):
                raise ValueError(f"record {index} has no text field")
            yield index, text
            target = next(targets, -1)
            if target < 0:
                return


# --- Hardware and resource sampling -------------------------------------------------------


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _nvidia_smi(fields: str) -> list[str] | None:
    try:
        output = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    first = output.strip().splitlines()[:1]
    return [part.strip() for part in first[0].split(",")] if first else None


def _cpu_name() -> str:
    if sys.platform == "win32":
        import winreg

        try:
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor()


def describe_hardware(device: str) -> dict[str, Any]:
    import psutil
    import torch

    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": _cpu_name(),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "ram_gib": round(psutil.virtual_memory().total / 2**30, 1),
        "torch_threads": torch.get_num_threads(),
        "device": device,
    }
    if device.startswith("cuda"):
        properties = torch.cuda.get_device_properties(0)
        smi = _nvidia_smi("driver_version,power.limit")
        info.update(
            {
                "gpu": properties.name,
                "gpu_memory_gib": round(properties.total_memory / 2**30, 2),
                "gpu_compute_capability": f"{properties.major}.{properties.minor}",
                "cuda_runtime": torch.version.cuda,
                "gpu_driver": smi[0] if smi else None,
            }
        )
    return info


class ResourceSampler:
    """Background sampler of CPU, process RSS, and GPU utilisation (no data access)."""

    def __init__(self, interval: float = 2.0, gpu: bool = False) -> None:
        import psutil

        self._process = psutil.Process()
        self._psutil = psutil
        self._interval, self._gpu = interval, gpu
        self._stop = threading.Event()
        self.samples: list[dict[str, float]] = []
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> ResourceSampler:
        self._process.cpu_percent(None)
        self._psutil.cpu_percent(None)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            sample = {
                "process_cpu_percent": self._process.cpu_percent(None),
                "system_cpu_percent": self._psutil.cpu_percent(None),
                "rss_mib": self._process.memory_info().rss / 2**20,
            }
            if self._gpu:
                smi = _nvidia_smi("utilization.gpu,memory.used")
                if smi:
                    sample["gpu_util_percent"] = float(smi[0])
                    sample["gpu_memory_used_mib"] = float(smi[1])
            self.samples.append(sample)

    def summary(self) -> dict[str, float]:
        result: dict[str, float] = {"samples": len(self.samples)}
        keys = sorted({key for sample in self.samples for key in sample})
        for key in keys:
            values = [sample[key] for sample in self.samples if key in sample]
            result[f"{key}_mean"] = round(sum(values) / len(values), 1)
            result[f"{key}_max"] = round(max(values), 1)
        return result


# --- Model --------------------------------------------------------------------------------


def load_model(
    label_map: GlinerLabelMap, device: str, dtype: str, max_width: int = DEFAULT_MAX_WIDTH
) -> tuple[Any, float]:
    """Load the pinned revision from the Hugging Face cache (never into the repo).

    ``max_width`` (longest predictable span, in words) is passed to
    ``GLiNER.from_pretrained`` as a config override; the default equals the
    checkpoint's own value. For this model's ``markerV0`` span layer the width
    only shapes the candidate-span grid and adds no weights.
    """
    from gliner import GLiNER

    started = time.perf_counter()
    model = GLiNER.from_pretrained(
        label_map.model_id,
        revision=label_map.model_revision,
        map_location=device,
        dtype=None if dtype == "fp32" else dtype,
        max_width=max_width,
    )
    require_max_width(model, max_width)
    model.eval()
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()
    return model, time.perf_counter() - started


def require_max_width(model: Any, max_width: int) -> None:
    """Fail if the loaded model does not use the requested span width."""
    configured = getattr(model.config, "max_width", None)
    if configured != max_width:
        raise RuntimeError(
            f"requested max_width={max_width} but the loaded model uses {configured}"
        )


def describe_model(model: Any, label_map: GlinerLabelMap, dtype: str) -> dict[str, Any]:
    config = model.config
    parameters = sum(parameter.numel() for parameter in model.parameters())
    return {
        "model_id": label_map.model_id,
        "model_revision": label_map.model_revision,
        "gliner": _version("gliner"),
        "torch": _version("torch"),
        "transformers": _version("transformers"),
        "encoder": getattr(config, "model_name", None),
        "span_mode": getattr(config, "span_mode", None),
        "max_len": getattr(config, "max_len", None),
        "max_width": getattr(config, "max_width", None),
        "words_splitter_type": getattr(config, "words_splitter_type", None),
        "parameters": parameters,
        "dtype": dtype,
    }


def word_splitter(model: Any) -> Any:
    """The model's own splitter, so chunk boundaries match its tokenization."""
    return model.data_processor.words_splitter


# --- Runtime log --------------------------------------------------------------------------


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def runtime_path(output: Path) -> Path:
    return output.with_name(output.name + ".runtime.json")


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


# --- Run identity -------------------------------------------------------------------------


def inference_config(
    model_identity: dict[str, Any], threshold: float, batch_size: int
) -> InferenceConfig:
    """Derive inference settings from the loaded model's effective configuration."""
    return InferenceConfig(
        threshold=threshold,
        batch_size=batch_size,
        max_words=int(model_identity["max_len"]),
        # Windows overlap by 2 * max_width words so no predictable entity is cut.
        margin=int(model_identity["max_width"]),
    )


def build_identity(
    source_identity: dict[str, Any],
    label_map: GlinerLabelMap,
    model_identity: dict[str, Any],
    config: InferenceConfig,
    device: str,
) -> dict[str, Any]:
    """Header identity; ``model.max_width`` is the effective width, so runs at
    different widths never share (or resume) a predictions file."""
    return {
        "source": source_identity,
        "label_map": label_map.as_identity(),
        "model": model_identity,
        "inference": {
            **config.as_identity(),
            "batch_size": config.batch_size,
            "device_type": device.split(":")[0],
        },
    }


# --- CLI ----------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline GLiNER-PII inference (offsets only).")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source", choices=("test", "dev"), required=True)
    parser.add_argument("--output", type=Path, required=True, help="External predictions JSONL.")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=8, help="GLiNER inference batch size.")
    parser.add_argument(
        "--read-batch",
        type=int,
        default=DEFAULT_READ_BATCH,
        help="Records per durable write (length-sorted into model batches).",
    )
    parser.add_argument("--limit", type=int, help="Only the first N selected records.")
    parser.add_argument("--dev-size", type=int, default=DEV_SUBSET_SIZE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument(
        "--max-width",
        type=int,
        default=DEFAULT_MAX_WIDTH,
        help="Longest predictable span in words, passed to GLiNER.from_pretrained "
        f"(default {DEFAULT_MAX_WIDTH}, the checkpoint's value). Also sets the chunk overlap.",
    )
    parser.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--restart", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if is_within_repository(args.output):
        print("error: --output must be outside the repository", file=sys.stderr)
        return 2
    if not 0.0 <= args.threshold < 1.0 or args.batch_size < 1 or args.read_batch < 1:
        print("error: invalid threshold, batch size, or read batch", file=sys.stderr)
        return 2
    if args.max_width < 1:
        print("error: --max-width must be a positive number of words", file=sys.stderr)
        return 2
    try:
        source = resolve_source(args.data_dir, args.source, args.dev_size)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    limit = None if args.limit is None else min(args.limit, len(source.indices))
    target = len(source.indices) if limit is None else limit
    label_map = load_label_map(args.label_map)

    # Library warnings (e.g. truncation) are counted structurally, never printed with data.
    warnings.filterwarnings("ignore", message="Sentence of length")
    try:
        model, load_seconds = load_model(label_map, args.device, args.dtype, args.max_width)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    model_identity = describe_model(model, label_map, args.dtype)
    config = inference_config(model_identity, args.threshold, args.batch_size)
    identity = build_identity(source.identity(None), label_map, model_identity, config, args.device)
    try:
        writer = PredictionWriter(
            args.output, identity, source.indices, resume=args.resume, restart=args.restart
        )
    except (PredictionsError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    predictor = GlinerPredictor(model, label_map, word_splitter(model), config)

    log_path = runtime_path(args.output)
    runtime: dict[str, Any] = {"sessions": []}
    if args.resume and log_path.is_file():
        runtime = json.loads(log_path.read_text(encoding="utf-8"))
    hardware = describe_hardware(args.device)
    session: dict[str, Any] = {
        "resumed_from": writer.completed,
        "hardware": hardware,
        "model_load_seconds": round(load_seconds, 2),
        "max_width": model_identity["max_width"],
        "batch_size": args.batch_size,
        "read_batch": args.read_batch,
        "processes": 1,
        "examples": 0,
        "chunks": 0,
        "inference_seconds": 0.0,
    }
    runtime["sessions"].append(session)
    batch_latencies: list[float] = []
    print(
        f"model ready: load_seconds={load_seconds:.1f} device={args.device} dtype={args.dtype} "
        f"max_width={model_identity['max_width']} completed={writer.completed}/{target}",
        flush=True,
    )

    use_gpu = args.device.startswith("cuda")
    if use_gpu:
        import torch

        torch.cuda.reset_peak_memory_stats()
    session_start = time.perf_counter()

    def flush_runtime(sampler: ResourceSampler) -> None:
        session["wall_seconds"] = round(time.perf_counter() - session_start, 2)
        session["inference_seconds"] = round(session["inference_seconds"], 3)
        session["examples_per_second"] = (
            round(session["examples"] / session["inference_seconds"], 2)
            if session["inference_seconds"]
            else None
        )
        session["batch_latency_seconds"] = {
            "p50": _percentile(batch_latencies, 0.5),
            "p95": _percentile(batch_latencies, 0.95),
            "max": max(batch_latencies) if batch_latencies else None,
        }
        session["per_example_latency_ms_mean"] = (
            round(1000 * session["inference_seconds"] / session["examples"], 2)
            if session["examples"]
            else None
        )
        session["resources"] = sampler.summary()
        if use_gpu:
            import torch

            session["gpu_peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
            session["gpu_peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20)
        runtime["completed_examples"] = writer.completed
        runtime["target_examples"] = target
        write_json_atomic(log_path, runtime)

    try:
        with ResourceSampler(gpu=use_gpu) as sampler:
            pending: list[tuple[int, str]] = []

            def run_pending() -> None:
                if not pending:
                    return
                started = time.perf_counter()
                results = predictor.predict([text for _, text in pending])
                if use_gpu:
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                writer.append(
                    [
                        PredictionRecord(
                            index,
                            result.chunks,
                            result.duplicates_removed,
                            tuple(result.predictions),
                        )
                        for (index, _), result in zip(pending, results, strict=True)
                    ]
                )
                session["examples"] += len(pending)
                session["chunks"] += sum(result.chunks for result in results)
                session["inference_seconds"] += elapsed
                batch_latencies.append(elapsed)
                pending.clear()
                flush_runtime(sampler)
                rate = session["examples"] / max(session["inference_seconds"], 1e-9)
                print(
                    f"progress: completed={writer.completed}/{target} "
                    f"examples_per_second={rate:.2f} "
                    f"eta_seconds={(target - writer.completed) / max(rate, 1e-9):.0f}",
                    flush=True,
                )

            for item in iter_texts(source, writer.completed, limit):
                pending.append(item)
                if len(pending) >= args.read_batch:
                    run_pending()
            run_pending()
            flush_runtime(sampler)
    except KeyboardInterrupt:
        print("interrupted; completed batches are durable. Rerun with --resume.", file=sys.stderr)
        return 130
    print(
        f"done: completed={writer.completed}/{target} "
        f"inference_seconds={session['inference_seconds']:.1f} output={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
