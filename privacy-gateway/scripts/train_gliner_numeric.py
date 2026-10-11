"""Fine-tune the pinned nvidia/gliner-PII checkpoint on the numeric dataset (offline).

Loads exactly the pinned Hugging Face snapshot with gliner==0.2.29, refuses any
other architecture, and writes checkpoints plus a final model to an external run
directory. Runs are resumable; resuming a run whose identity changed is refused.
Nothing here logs text, entity values, or examples: the trainer reports loss only.

gliner 0.2.29 notes handled here:
- ``BaseGLiNER.train_model`` cannot resume, so the Trainer is built directly.
- ``freeze_component`` only warns on unknown names; freezing is checked strictly.
- There is no GLiNER-level gradient checkpointing; it is enabled on the encoder.
- ``Trainer.training_step`` skips CUDA OOM batches with a zero loss; this run aborts.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import random
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluation_checkpoint import canonical_sha256, sha256_file
from scripts.gliner_numeric_common import (
    DEFAULT_CONFIG,
    EXPERIMENT_ID,
    OTHER_NUMERIC,
    PINNED_MODEL_ID,
    PINNED_MODEL_REVISION,
    TOKEN_WINDOWING_METHOD,
    GlinerInputSizer,
    NumericConfig,
    NumericExperimentError,
    check_output_path,
    load_config,
    package_version,
    require_training_packages,
    resolve_tokenizer_sequence_limit,
    verify_architecture,
    verify_snapshot_revision,
)
from scripts.gliner_predict import require_max_width, write_json_atomic
from scripts.prepare_gliner_numeric_finetune import DEFAULT_OUTPUT_DIR as DEFAULT_DATASET_DIR

DEFAULT_RUNS_DIR = Path.home() / ".cache" / "privacy-gateway" / "gliner-numeric-finetune" / "runs"
LINEAGE_FILE = "numeric_finetune_lineage.json"
RUN_IDENTITY_FILE = "run_identity.json"
RUN_RECORD_FILE = "training_run.json"
MODEL_KEYS = ("tokenized_text", "ner", "ner_labels")
_SKIPPED_BATCH_PREFIX = "Skipping batch due to"


# --- Dataset ----------------------------------------------------------------------------


def load_manifest(dataset_dir: Path, config: NumericConfig) -> dict[str, Any]:
    """The dataset must come from this experiment's data and label config, unmodified."""
    path = dataset_dir / "manifest.json"
    if not path.is_file():
        raise NumericExperimentError("dataset manifest.json is missing")
    manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("experiment") != EXPERIMENT_ID:
        raise NumericExperimentError("dataset was built for another experiment")
    for key, section in (("data_config_sha256", "data"), ("labels_config_sha256", "labels")):
        if manifest.get(key) != config.section_sha256(section):
            raise NumericExperimentError(f"dataset {section} config differs from this config")
    if manifest.get("label_vocabulary", {}).get("training_sha256") != canonical_sha256(
        list(config.training_labels)
    ):
        raise NumericExperimentError("dataset label vocabulary differs from this config")
    for name in ("train.jsonl", "dev.jsonl"):
        expected = manifest.get("files", {}).get(name, {}).get("sha256")
        if not expected or sha256_file(dataset_dir / name) != expected:
            raise NumericExperimentError(f"dataset file {name} does not match its manifest")
    windowing = manifest.get("format", {}).get("subword_windowing", {})
    if (
        windowing.get("method") != TOKEN_WINDOWING_METHOD
        or not isinstance(windowing.get("effective_limit"), int)
        or windowing["effective_limit"] < 1
    ):
        raise NumericExperimentError(
            "dataset was not built with the required tokenizer-aware windowing; rebuild it"
        )
    return manifest


def read_records(
    path: Path, labels: Sequence[str], max_width: int, limit: int | None = None
) -> list[dict[str, Any]]:
    """Read and validate GLiNER span records; keep only the keys the processor uses."""
    allowed = set(labels)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="\n") as handle:
        for line_number, line in enumerate(handle, start=1):
            if limit is not None and len(records) >= limit:
                break
            record = json.loads(line)
            tokens, ner, prompt = (record.get(key) for key in MODEL_KEYS)
            if not (
                isinstance(tokens, list) and tokens and all(isinstance(t, str) for t in tokens)
            ):
                raise NumericExperimentError(f"{path.name} line {line_number}: invalid tokens")
            if not (isinstance(prompt, list) and prompt and set(prompt) <= allowed):
                raise NumericExperimentError(f"{path.name} line {line_number}: invalid labels")
            if len(set(prompt)) != len(prompt):
                raise NumericExperimentError(f"{path.name} line {line_number}: duplicate labels")
            if not isinstance(ner, list):
                raise NumericExperimentError(f"{path.name} line {line_number}: invalid ner")
            for item in ner:
                valid = (
                    isinstance(item, list)
                    and len(item) == 3
                    and all(isinstance(value, int) for value in item[:2])
                    and 0 <= item[0] <= item[1] < len(tokens)
                    and item[1] - item[0] < max_width
                    and item[2] in prompt
                )
                if not valid:
                    raise NumericExperimentError(f"{path.name} line {line_number}: invalid span")
            records.append({key: record[key] for key in MODEL_KEYS})
    if not records:
        raise NumericExperimentError(f"{path.name} has no records")
    return records


# --- Settings and identity --------------------------------------------------------------


def resolve_settings(
    config: NumericConfig, profile_name: str | None, args: argparse.Namespace
) -> dict[str, Any]:
    """Profile values, then pilot overrides, then explicit CLI overrides."""
    training = config.training
    settings = config.profile(profile_name)
    settings["profile"] = profile_name or training["default_profile"]
    settings["pilot"] = bool(args.pilot)
    settings["limit_train_examples"] = None
    settings["limit_dev_examples"] = int(training["eval_dev_examples"])
    if args.pilot:
        pilot = training["pilot"]
        settings.update(
            max_steps=int(pilot["max_steps"]),
            save_steps=int(pilot["save_steps"]),
            eval_steps=int(pilot["eval_steps"]),
            logging_steps=int(pilot["logging_steps"]),
            limit_train_examples=int(pilot["limit_train_examples"]),
            limit_dev_examples=int(pilot["limit_dev_examples"]),
        )
    for key in (
        "max_steps",
        "limit_train_examples",
        "limit_dev_examples",
        "gradient_accumulation_steps",
        "per_device_train_batch_size",
    ):
        value = getattr(args, key, None)
        if value is not None:
            if value < 1:
                raise NumericExperimentError(f"--{key.replace('_', '-')} must be positive")
            settings[key] = value
    if args.fp16:
        settings["fp16"] = True
    if args.bf16:
        settings["bf16"] = True
    if args.gradient_checkpointing is not None:
        settings["gradient_checkpointing"] = args.gradient_checkpointing
    if settings["fp16"] and settings["bf16"]:
        raise NumericExperimentError("choose at most one of fp16 and bf16")
    if settings["gradient_checkpointing"] and "text_encoder" in settings["freeze_components"]:
        raise NumericExperimentError("gradient checkpointing needs a trainable encoder")
    settings["use_cpu"] = bool(args.use_cpu)
    return settings


def training_arguments_kwargs(
    settings: Mapping[str, Any], checkpoints_dir: Path, seed: int, report_to: str
) -> dict[str, Any]:
    """Keyword arguments for gliner.training.TrainingArguments (no trackers)."""
    if report_to != "none":
        raise NumericExperimentError("report_to must stay none")
    return {
        "output_dir": str(checkpoints_dir),
        "learning_rate": settings["learning_rate"],
        "weight_decay": settings["weight_decay"],
        "others_lr": settings["others_lr"],
        "others_weight_decay": settings["others_weight_decay"],
        "focal_loss_alpha": settings["focal_loss_alpha"],
        "focal_loss_gamma": settings["focal_loss_gamma"],
        "loss_reduction": settings["loss_reduction"],
        "negatives": settings["negatives"],
        "masking": settings["masking"],
        "lr_scheduler_type": settings["lr_scheduler_type"],
        "warmup_ratio": settings["warmup_ratio"],
        "per_device_train_batch_size": settings["per_device_train_batch_size"],
        "per_device_eval_batch_size": settings["per_device_eval_batch_size"],
        "gradient_accumulation_steps": settings["gradient_accumulation_steps"],
        "max_grad_norm": settings["max_grad_norm"],
        "max_steps": settings["max_steps"],
        "eval_strategy": "steps",
        "eval_steps": settings["eval_steps"],
        "save_strategy": "steps",
        "save_steps": settings["save_steps"],
        "save_total_limit": settings["save_total_limit"],
        "logging_strategy": "steps",
        "logging_steps": settings["logging_steps"],
        "load_best_model_at_end": False,
        "prediction_loss_only": True,
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "report_to": report_to,
        "fp16": settings["fp16"],
        "bf16": settings["bf16"],
        "gradient_checkpointing": False,
        "use_cpu": settings["use_cpu"],
        "seed": seed,
        "data_seed": seed,
    }


def run_identity(
    config: NumericConfig,
    settings: Mapping[str, Any],
    manifest: Mapping[str, Any],
    dataset_dir: Path,
) -> dict[str, Any]:
    """Everything that must be unchanged for a resume to continue the same run."""
    base = config.raw["base_model"]
    return {
        "experiment": EXPERIMENT_ID,
        "config_sha256": config.sha256,
        "seed": config.seed,
        "settings": dict(settings),
        "base_model": {"model_id": base["model_id"], "revision": base["revision"]},
        "dataset": {
            "manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
            "train_sha256": manifest["files"]["train.jsonl"]["sha256"],
            "dev_sha256": manifest["files"]["dev.jsonl"]["sha256"],
        },
        "training_labels_sha256": canonical_sha256(list(config.training_labels)),
    }


def prepare_run_directory(
    output_dir: Path, identity: Mapping[str, Any], resume: bool, hf_cache: Path | None
) -> Path | None:
    """Create or reopen a run; return the checkpoint to resume from, if any."""
    check_output_path(output_dir, hf_cache)
    identity_path = output_dir / RUN_IDENTITY_FILE
    checkpoints = output_dir / "checkpoints"
    if identity_path.exists():
        stored = json.loads(identity_path.read_text(encoding="utf-8"))
        if stored != json.loads(json.dumps(identity)):
            raise NumericExperimentError("run directory belongs to a different run identity")
        if not resume:
            raise NumericExperimentError("run directory exists; pass --resume to continue it")
        if (output_dir / "final_model").exists():
            raise NumericExperimentError("run already has a final model")
        found = sorted(
            (path for path in checkpoints.glob("checkpoint-*") if path.is_dir()),
            key=lambda path: int(path.name.rsplit("-", 1)[1]),
        )
        return found[-1] if found else None
    if output_dir.exists() and any(output_dir.iterdir()):
        raise NumericExperimentError("run directory is not empty and has no run identity")
    if resume:
        raise NumericExperimentError("--resume given but the run directory is new")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(identity_path, dict(identity))
    return None


# --- Runtime (torch / gliner) -----------------------------------------------------------


def seed_everything(seed: int, deterministic: bool) -> None:
    import numpy
    import torch

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def resolve_base_snapshot(model_id: str, revision: str) -> Path:
    """Download (or reuse) exactly the pinned snapshot from the Hugging Face cache."""
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(repo_id=model_id, revision=revision))
    verify_snapshot_revision(snapshot, revision)
    return snapshot


def weight_files(snapshot: Path) -> list[Path]:
    files = sorted(snapshot.glob("*.safetensors")) + sorted(snapshot.glob("pytorch_model*.bin"))
    if not files:
        raise NumericExperimentError("pinned snapshot has no weight file")
    return files


def freeze_strictly(model: Any, components: Sequence[str]) -> None:
    available = model._get_freezable_components()
    for name in components:
        if name not in available:
            raise NumericExperimentError("component to freeze is not present in the model")
        model.freeze_component(name)
        if any(parameter.requires_grad for parameter in available[name].parameters()):
            raise NumericExperimentError("component did not freeze")


def enable_encoder_checkpointing(model: Any) -> None:
    encoder = model._get_freezable_components().get("text_encoder")
    enable = getattr(encoder, "gradient_checkpointing_enable", None)
    if enable is None or not getattr(encoder, "supports_gradient_checkpointing", False):
        raise NumericExperimentError("the text encoder does not support gradient checkpointing")
    enable(gradient_checkpointing_kwargs={"use_reentrant": False})


def parameter_counts(model: Any) -> dict[str, int]:
    total = trainable = 0
    for parameter in model.parameters():
        total += parameter.numel()
        trainable += parameter.numel() if parameter.requires_grad else 0
    return {"total": total, "trainable": trainable}


def drop_subword_overflow(
    model: Any, records: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int, int]:
    """Drop examples whose prompt + words exceed the tokenizer limit (no truncation)."""
    processor = model.data_processor
    tokenizer = processor.transformer_tokenizer
    components = model._get_freezable_components()
    text_encoder = components.get("text_encoder") if isinstance(components, Mapping) else None
    limit = resolve_tokenizer_sequence_limit(tokenizer, text_encoder)
    sizer = GlinerInputSizer(processor, limit)
    kept = []
    for record in records:
        if sizer.fits(record["tokenized_text"], record["ner_labels"]):
            kept.append(record)
    return kept, len(records) - len(kept), limit


class _SkippedBatchCounter(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if str(record.msg).startswith(_SKIPPED_BATCH_PREFIX):
            self.count += 1


def _abort_on_skipped_batches(counter: _SkippedBatchCounter) -> Any:
    from transformers import TrainerCallback

    class AbortOnSkippedBatch(TrainerCallback):  # type: ignore[misc]
        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            if counter.count:
                raise RuntimeError(
                    "gliner skipped a batch after running out of memory; lower the batch "
                    "size, freeze the encoder, or enable gradient checkpointing"
                )
            return control

    return AbortOnSkippedBatch()


def _hardware(use_cpu: bool) -> dict[str, Any]:
    import platform

    import torch

    info: dict[str, Any] = {"platform": platform.platform(), "python": platform.python_version()}
    if torch.cuda.is_available() and not use_cpu:
        properties = torch.cuda.get_device_properties(0)
        info.update(
            gpu=properties.name,
            gpu_memory_gib=round(properties.total_memory / 2**30, 2),
            gpu_compute_capability=f"{properties.major}.{properties.minor}",
            cuda_runtime=torch.version.cuda,
        )
    else:
        info["device"] = "cpu"
    return info


def train(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    versions = require_training_packages()
    from huggingface_hub import constants

    hf_cache = Path(constants.HF_HUB_CACHE)
    settings = resolve_settings(config, args.profile, args)
    manifest = load_manifest(args.dataset_dir, config)
    if "limited_source_rows_smoke_build" in manifest.get("warnings", []) and not args.pilot:
        raise NumericExperimentError("a smoke-build dataset can only be used with --pilot")
    output_dir = args.output_dir or DEFAULT_RUNS_DIR / (
        f"{settings['profile']}{'-pilot' if args.pilot else ''}"
    )
    identity = run_identity(config, settings, manifest, args.dataset_dir)
    resume_from = prepare_run_directory(output_dir, identity, args.resume, hf_cache)
    record_path = output_dir / RUN_RECORD_FILE
    record: dict[str, Any] = {
        "status": "starting",
        "pilot": settings["pilot"],
        "identity_sha256": canonical_sha256(identity),
        "identity": identity,
        "packages": versions,
        "resumed_from": resume_from.name if resume_from else None,
        "started_unix": time.time(),
        "privacy": "loss metrics and aggregate counts only",
    }
    write_json_atomic(record_path, record)

    base = config.raw["base_model"]
    if (base["model_id"], base["revision"]) != (PINNED_MODEL_ID, PINNED_MODEL_REVISION):
        record["warnings"] = ["base_model_is_not_the_pinned_gliner_pii_revision"]
    seed_everything(config.seed, bool(config.training["deterministic_algorithms"]))
    from transformers import set_seed

    set_seed(config.seed)
    snapshot = resolve_base_snapshot(base["model_id"], base["revision"])
    record["base_snapshot"] = {
        "revision": snapshot.name,
        "weights": {
            path.name: (sha256_file(path) if config.training["hash_base_weights"] else None)
            for path in weight_files(snapshot)
        },
    }

    from gliner import GLiNER

    max_width = int(config.training["max_width"])
    model = GLiNER.from_pretrained(str(snapshot), map_location="cpu", max_width=max_width)
    require_max_width(model, max_width)
    record["architecture"] = verify_architecture(model, base)
    freeze_strictly(model, settings["freeze_components"])
    if settings["gradient_checkpointing"]:
        enable_encoder_checkpointing(model)
    record["parameters"] = parameter_counts(model)

    labels = config.training_labels
    train_records = read_records(
        args.dataset_dir / "train.jsonl", labels, max_width, settings["limit_train_examples"]
    )
    dev_records = read_records(
        args.dataset_dir / "dev.jsonl", labels, max_width, settings["limit_dev_examples"]
    )
    overflow: dict[str, Any] = {}
    built_token_limit = int(manifest["format"]["subword_windowing"]["effective_limit"])
    for name, rows in (("train", train_records), ("dev", dev_records)):
        kept, dropped, limit = drop_subword_overflow(model, rows)
        if limit != built_token_limit:
            raise NumericExperimentError(
                "loaded model token limit differs from the dataset builder limit; rebuild it"
            )
        overflow[name] = {"examples": len(rows), "dropped": dropped, "token_limit": limit}
        if dropped / len(rows) > float(config.training["max_subword_overflow_fraction"]):
            raise NumericExperimentError(f"too many {name} examples exceed the token limit")
        rows[:] = kept
    record["subword_overflow"] = overflow
    record["examples"] = {"train": len(train_records), "dev": len(dev_records)}

    from gliner.training import Trainer, TrainingArguments

    kwargs = training_arguments_kwargs(
        settings, output_dir / "checkpoints", config.seed, str(config.training["report_to"])
    )
    supported = {item.name for item in dataclasses.fields(TrainingArguments)}
    unsupported = sorted(set(kwargs) - supported)
    if unsupported:
        raise NumericExperimentError(
            "installed transformers lacks TrainingArguments: " + ", ".join(unsupported)
        )
    training_args = TrainingArguments(**kwargs)
    counter = _SkippedBatchCounter()
    logging.getLogger("gliner.training.trainer").addHandler(counter)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_records,
        "eval_dataset": dev_records,
        "data_collator": model._create_data_collator(),
        "callbacks": [_abort_on_skipped_batches(counter)],
    }
    tokenizer_key = "tokenizer" if versions["transformers"].startswith("4.") else "processing_class"
    trainer_kwargs[tokenizer_key] = model.data_processor.transformer_tokenizer
    trainer = Trainer(**trainer_kwargs)

    record.update(status="training", hardware=_hardware(settings["use_cpu"]))
    write_json_atomic(record_path, record)
    try:
        result = trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
    except BaseException:
        record.update(status="failed", failed_unix=time.time())
        write_json_atomic(record_path, record)
        raise

    final_dir = output_dir / "final_model"
    trainer.save_model(str(final_dir))
    lineage = {
        "experiment": EXPERIMENT_ID,
        "pilot": settings["pilot"],
        "base_model": identity["base_model"],
        "identity_sha256": record["identity_sha256"],
        "dataset_manifest_sha256": identity["dataset"]["manifest_sha256"],
        "global_step": int(trainer.state.global_step),
        "training_labels": list(labels),
        "training_only_labels": [OTHER_NUMERIC],
        "other_numeric": "training-only prompt label; maps to no gateway type",
        "max_width": max_width,
        "gliner_version": versions["gliner"],
        "production_ready": False,
    }
    write_json_atomic(final_dir / LINEAGE_FILE, lineage)
    record.update(
        status="completed",
        completed_unix=time.time(),
        global_step=int(trainer.state.global_step),
        train_metrics={
            key: value for key, value in result.metrics.items() if isinstance(value, int | float)
        },
        log_history=[
            {key: value for key, value in entry.items() if isinstance(value, int | float)}
            for entry in trainer.state.log_history
        ],
        final_model=str(final_dir),
        final_model_files={
            path.name: sha256_file(path) for path in sorted(final_dir.iterdir()) if path.is_file()
        },
        accelerate=package_version("accelerate"),
    )
    write_json_atomic(record_path, record)
    return record


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune pinned GLiNER-PII (offline).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, help="External run directory.")
    parser.add_argument("--profile", help="Training profile from the config.")
    parser.add_argument("--pilot", action="store_true", help="Tiny run from the pilot settings.")
    parser.add_argument("--resume", action="store_true", help="Continue an existing run.")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--limit-train-examples", type=int)
    parser.add_argument("--limit-dev-examples", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--per-device-train-batch-size", type=int)
    parser.add_argument("--fp16", action="store_true", help="Mixed precision (T4: fp16 only).")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--use-cpu", action="store_true", help="Debugging only.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        record = train(args)
    except NumericExperimentError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        f"training {record['status']} step={record.get('global_step')} "
        f"final_model={record.get('final_model')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
