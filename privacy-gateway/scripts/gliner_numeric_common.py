"""Shared, torch-free pieces of the offline GLiNER numeric fine-tuning experiment.

Config validation, the 55-label + ``other_numeric`` training vocabulary, the list
of frozen benchmark splits that must never be trained on, output-path guards, and
checks that a loaded checkpoint is the pinned architecture. Nothing here imports
gliner, torch, or transformers.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from scripts.evaluation_checkpoint import canonical_sha256
from scripts.gliner_evaluation import UNMAPPED, GlinerLabelMap, load_label_map
from scripts.nemotron_pii import REPOSITORY_ROOT, is_within_repository, load_ontology
from scripts.presidio_evaluation import SUPPORTED_TYPES

DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "gliner_numeric_finetune.yaml"
EXPERIMENT_ID = "gliner-numeric-finetune/v1"
OTHER_NUMERIC = "other_numeric"
OTHER_NUMERIC_CLASS = "OTHER_NUMERIC"
PINNED_MODEL_ID = "nvidia/gliner-PII"
PINNED_MODEL_REVISION = "bd23e8ef4425fd04e34c5204ab49ffaa706eae79"
PINNED_NEMOTRON_REVISION = "b70ffaf5ff39e079776134c5bf4381f00a9fd1ed"
REQUIRED_GLINER_VERSION = "0.2.29"
# gliner 0.2.29's own pin: transformers>=4.51.3,<5.17.0.
TRANSFORMERS_RANGE = ((4, 51, 3), (5, 17, 0))
PROTOCOL_MAX_WIDTH = 24
PROTOCOL_THRESHOLDS = (0.3, 0.7)
MODEL_MAX_WORDS = 384
FREEZABLE_COMPONENTS = frozenset({"text_encoder"})

# Frozen benchmark rows. Argilla publishes one split, and it is the benchmark.
FORBIDDEN_TRAINING_SPLITS: Mapping[str, frozenset[str] | None] = {
    "ai4privacy": frozenset({"validation"}),
    "gretel": frozenset({"test"}),
    "argilla": None,
    "nemotron": frozenset({"test"}),
}
OOD_BENCHMARK_SHA256 = {
    "ai4privacy": "575c7105a3ad5d27cbea7a467bebc360b08edfcdebadd71a45d5962a12472fca",
    "gretel": "2b628ffc09effc56475300038c648252a5faf9eb4f8877c9221ef31e5ca727f9",
    "argilla": "551e98472d8c0df3e5ec1590f0b62960d3be667266eadf65518715fd534ee6cf",
}
SOURCE_DATASETS = {
    "nemotron_train": "nemotron",
    "ai4privacy_train": "ai4privacy",
    "gretel_train": "gretel",
}

_PROFILE_KEYS = {
    "freeze_components": list,
    "gradient_checkpointing": bool,
    "fp16": bool,
    "bf16": bool,
    "learning_rate": float,
    "others_lr": float,
    "weight_decay": float,
    "others_weight_decay": float,
    "per_device_train_batch_size": int,
    "per_device_eval_batch_size": int,
    "gradient_accumulation_steps": int,
    "max_steps": int,
    "warmup_ratio": float,
    "lr_scheduler_type": str,
    "max_grad_norm": float,
    "save_steps": int,
    "eval_steps": int,
    "save_total_limit": int,
    "logging_steps": int,
    "focal_loss_alpha": float,
    "focal_loss_gamma": float,
    "loss_reduction": str,
    "negatives": float,
    "masking": str,
}
_LABEL_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class NumericExperimentError(ValueError):
    """Experiment inputs are unsafe or inconsistent. Messages contain no source data."""


def is_forbidden_split(dataset: str, split: str) -> bool:
    """True when a dataset split is a frozen benchmark that training must never read."""
    if dataset not in FORBIDDEN_TRAINING_SPLITS:
        raise NumericExperimentError("unknown training source dataset")
    forbidden = FORBIDDEN_TRAINING_SPLITS[dataset]
    return forbidden is None or split in forbidden


def lf_sha256(path: Path) -> str:
    """SHA-256 of a text file with CRLF folded to LF, matching published artifact hashes."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _release(version: str) -> tuple[int, ...]:
    match = re.match(r"^(\d+(?:\.\d+)*)", version)
    if match is None:
        raise NumericExperimentError("package version is not a release number")
    return tuple(int(part) for part in match.group(1).split("."))


def require_training_packages(versions: Mapping[str, str | None] | None = None) -> dict[str, str]:
    """Fail clearly unless the exact gliner release and a compatible stack are installed."""
    found = (
        dict(versions)
        if versions is not None
        else {
            name: package_version(name)
            for name in ("gliner", "torch", "transformers", "accelerate", "numpy")
        }
    )
    missing = sorted(name for name, version in found.items() if version is None)
    if missing:
        raise NumericExperimentError(
            "missing packages for fine-tuning: "
            + ", ".join(missing)
            + "; install the gliner-finetune extra"
        )
    if found["gliner"] != REQUIRED_GLINER_VERSION:
        raise NumericExperimentError(
            f"gliner=={REQUIRED_GLINER_VERSION} is required; found {found['gliner']}"
        )
    low, high = TRANSFORMERS_RANGE
    transformers = _release(str(found["transformers"]))
    if not low <= transformers < high:
        raise NumericExperimentError(
            f"transformers {found['transformers']} is outside gliner 0.2.29's supported range"
        )
    return {name: str(version) for name, version in found.items()}


def check_output_path(path: Path, hf_cache: Path | None = None) -> None:
    """Experiment outputs live outside the package tree and never inside the HF cache."""
    if is_within_repository(path):
        raise NumericExperimentError("output path must be outside the repository")
    if hf_cache is not None:
        try:
            path.resolve().relative_to(hf_cache.resolve())
        except ValueError:
            return
        raise NumericExperimentError("output path must not be inside the Hugging Face cache")


@dataclass(frozen=True)
class NumericConfig:
    """Validated experiment configuration and its canonical hash."""

    raw: Mapping[str, Any]
    sha256: str
    label_map: GlinerLabelMap

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def data(self) -> Mapping[str, Any]:
        return self.raw["data"]  # type: ignore[no-any-return]

    @property
    def training(self) -> Mapping[str, Any]:
        return self.raw["training"]  # type: ignore[no-any-return]

    @property
    def evaluation(self) -> Mapping[str, Any]:
        return self.raw["evaluation"]  # type: ignore[no-any-return]

    @property
    def focus_labels(self) -> tuple[str, ...]:
        return tuple(self.raw["labels"]["focus"])

    @property
    def competitor_labels(self) -> tuple[str, ...]:
        return tuple(self.raw["labels"]["numeric_competitors"])

    @property
    def native_labels(self) -> tuple[str, ...]:
        return self.label_map.labels

    @property
    def training_labels(self) -> tuple[str, ...]:
        """The 55 native labels in label-map order, then the training-only label."""
        return (*self.label_map.labels, OTHER_NUMERIC)

    def profile(self, name: str | None = None) -> dict[str, Any]:
        profiles = self.training["profiles"]
        chosen = name or self.training["default_profile"]
        if chosen not in profiles:
            raise NumericExperimentError("unknown training profile")
        return dict(profiles[chosen])

    def section_sha256(self, key: str) -> str:
        return canonical_sha256(self.raw[key])


def extended_label_map(label_map: GlinerLabelMap) -> GlinerLabelMap:
    """Native label map plus ``other_numeric`` mapped to no gateway type."""
    if OTHER_NUMERIC in label_map.mapping:
        raise NumericExperimentError("other_numeric must not be part of the native label map")
    return GlinerLabelMap(
        model_id=label_map.model_id,
        model_revision=label_map.model_revision,
        labels=(*label_map.labels, OTHER_NUMERIC),
        mapping={**label_map.mapping, OTHER_NUMERIC: UNMAPPED},
        deviations=dict(label_map.deviations),
    )


def _require(condition: object, message: str) -> None:
    if not condition:
        raise NumericExperimentError(message)


def _number(value: object, kind: type) -> bool:
    if isinstance(value, bool):
        return kind is bool
    if kind is float:
        return isinstance(value, int | float) and math.isfinite(float(value))
    return isinstance(value, kind)


def _validate_profile(name: str, profile: object) -> None:
    _require(isinstance(profile, Mapping), f"training profile {name} must be a mapping")
    assert isinstance(profile, Mapping)
    _require(set(profile) == set(_PROFILE_KEYS), f"training profile {name} has unexpected keys")
    for key, kind in _PROFILE_KEYS.items():
        _require(_number(profile[key], kind), f"training profile {name}.{key} has the wrong type")
    _require(not (profile["fp16"] and profile["bf16"]), f"profile {name} sets fp16 and bf16")
    frozen = profile["freeze_components"]
    _require(
        all(isinstance(item, str) for item in frozen) and set(frozen) <= FREEZABLE_COMPONENTS,
        f"profile {name} freezes an unknown component",
    )
    _require(
        not (profile["gradient_checkpointing"] and "text_encoder" in frozen),
        f"profile {name} enables gradient checkpointing on a frozen encoder",
    )
    for key in (
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "max_steps",
        "save_steps",
        "eval_steps",
        "save_total_limit",
        "logging_steps",
    ):
        _require(profile[key] >= 1, f"profile {name}.{key} must be positive")
    for key in ("learning_rate", "others_lr"):
        _require(0 < profile[key] < 1, f"profile {name}.{key} must be in (0, 1)")
    _require(0 <= profile["warmup_ratio"] <= 1, f"profile {name}.warmup_ratio must be in [0, 1]")
    _require(profile["loss_reduction"] in {"sum", "mean"}, f"profile {name} loss_reduction")


def validate_config(raw: object, label_map: GlinerLabelMap, ontology: Mapping[str, str]) -> None:
    """Reject any configuration that would change the frozen protocol or vocabulary."""
    _require(isinstance(raw, Mapping), "config must be a mapping")
    assert isinstance(raw, Mapping)
    _require(raw.get("experiment") == EXPERIMENT_ID, "config experiment id is not recognized")
    _require(_number(raw.get("seed"), int), "config seed must be an integer")

    base = raw.get("base_model")
    _require(isinstance(base, Mapping), "config base_model must be a mapping")
    assert isinstance(base, Mapping)
    pinned = (base.get("model_id"), base.get("revision")) == (
        label_map.model_id,
        label_map.model_revision,
    )
    _require(
        pinned or base.get("allow_unpinned_base") is True,
        "base model differs from the pinned label map; set allow_unpinned_base to change it",
    )
    _require(
        base.get("gliner_version") == REQUIRED_GLINER_VERSION,
        "config must target gliner 0.2.29",
    )
    for key in ("architecture", "span_mode", "encoder", "words_splitter_type"):
        _require(isinstance(base.get(key), str), f"config base_model.{key} is required")

    labels = raw.get("labels")
    _require(isinstance(labels, Mapping), "config labels must be a mapping")
    assert isinstance(labels, Mapping)
    _require(
        labels.get("label_map") == "configs/gliner_label_map.yaml",
        "the experiment must use the existing 55-label GLiNER label map",
    )
    _require(len(label_map.labels) == 55, "the native GLiNER vocabulary must have 55 labels")
    _require(labels.get("training_only") == [OTHER_NUMERIC], "only other_numeric may be added")
    _require(OTHER_NUMERIC not in ontology, "other_numeric must not enter the gateway ontology")
    _require(OTHER_NUMERIC not in label_map.mapping, "other_numeric must not enter the label map")
    for key in ("focus", "numeric_competitors"):
        values = labels.get(key)
        _require(
            isinstance(values, list) and values and set(values) <= set(label_map.labels),
            f"config labels.{key} must be native labels",
        )
        assert isinstance(values, list)
        _require(len(set(values)) == len(values), f"config labels.{key} has duplicates")
    _require(
        not set(labels["focus"]) & set(labels["numeric_competitors"]),
        "focus and competitor labels overlap",
    )

    data = raw.get("data")
    _require(isinstance(data, Mapping), "config data must be a mapping")
    assert isinstance(data, Mapping)
    _require(data.get("max_width") == PROTOCOL_MAX_WIDTH, "data.max_width must stay 24")
    _require(
        _number(data.get("max_words"), int) and 1 <= data["max_words"] <= MODEL_MAX_WORDS,
        "data.max_words must be within the model's 384-word limit",
    )
    _require(data["max_words"] > PROTOCOL_MAX_WIDTH, "data.max_words must exceed max_width")
    fraction: Any = data.get("dev_fraction")
    _require(_number(fraction, float) and 0 < fraction < 0.5, "data.dev_fraction out of range")
    _require(
        isinstance(data.get("reserve_threshold_selection_subset"), bool),
        "data.reserve_threshold_selection_subset must be a boolean",
    )
    sources = data.get("sources")
    _require(
        isinstance(sources, Mapping) and set(sources) == set(SOURCE_DATASETS),
        "data.sources must list exactly the supported training sources",
    )
    assert isinstance(sources, Mapping)
    for name, dataset in SOURCE_DATASETS.items():
        source = sources[name]
        _require(isinstance(source, Mapping), f"data.sources.{name} must be a mapping")
        _require(isinstance(source.get("enabled"), bool), f"data.sources.{name}.enabled")
        split = source.get("split")
        _require(isinstance(split, str) and bool(split), f"data.sources.{name}.split")
        _require(
            not is_forbidden_split(dataset, split),
            f"data.sources.{name} points at a frozen benchmark split",
        )
    nemotron = sources["nemotron_train"]
    _require(
        nemotron.get("repository") == "nvidia/Nemotron-PII"
        and nemotron.get("revision") == PINNED_NEMOTRON_REVISION,
        "Nemotron source must stay pinned",
    )

    balancing = data.get("balancing")
    _require(isinstance(balancing, Mapping), "data.balancing must be a mapping")
    assert isinstance(balancing, Mapping)
    _require(
        _number(balancing.get("focus_oversample_factor"), int)
        and 1 <= balancing["focus_oversample_factor"] <= 10,
        "data.balancing.focus_oversample_factor must be in [1, 10]",
    )
    low: Any = balancing.get("min_focus_share")
    high: Any = balancing.get("max_focus_share")
    _require(
        _number(low, float) and _number(high, float) and 0 <= low <= high < 0.5,
        "focus share bounds must satisfy 0 <= min <= max < 0.5",
    )

    hard = data.get("hard_negatives")
    _require(isinstance(hard, Mapping), "data.hard_negatives must be a mapping")
    assert isinstance(hard, Mapping)
    for key in ("train_count", "dev_count"):
        _require(_number(hard.get(key), int) and hard[key] >= 0, f"hard_negatives.{key}")
    for key in (
        "card_shaped_fraction",
        "multi_span_fraction",
        "unlabeled_quantity_fraction",
        "native_context_fraction",
    ):
        _require(_number(hard.get(key), float) and 0 <= hard[key] <= 1, f"hard_negatives.{key}")
    contexts = hard.get("other_numeric_contexts")
    _require(
        isinstance(contexts, list) and contexts and all(isinstance(c, str) and c for c in contexts),
        "hard_negatives.other_numeric_contexts must be non-empty strings",
    )
    native_contexts = hard.get("native_contexts")
    _require(isinstance(native_contexts, Mapping), "hard_negatives.native_contexts")
    assert isinstance(native_contexts, Mapping)
    _require(
        set(native_contexts.values()) <= set(labels["focus"]) - {"credit_debit_card"},
        "native contexts must map to focus labels other than credit_debit_card",
    )

    training = raw.get("training")
    _require(isinstance(training, Mapping), "config training must be a mapping")
    assert isinstance(training, Mapping)
    _require(training.get("max_width") == PROTOCOL_MAX_WIDTH, "training.max_width must stay 24")
    _require(training.get("report_to") == "none", "training.report_to must be none")
    _require(training.get("dataloader_num_workers") == 0, "dataloader workers must be 0")
    for key in ("deterministic_algorithms", "hash_base_weights"):
        _require(isinstance(training.get(key), bool), f"training.{key} must be a boolean")
    overflow: Any = training.get("max_subword_overflow_fraction")
    _require(
        _number(overflow, float) and 0 <= overflow < 0.1,
        "training.max_subword_overflow_fraction must be in [0, 0.1)",
    )
    _require(
        _number(training.get("eval_dev_examples"), int) and training["eval_dev_examples"] >= 1,
        "training.eval_dev_examples must be positive",
    )
    profiles = training.get("profiles")
    _require(isinstance(profiles, Mapping) and bool(profiles), "training.profiles is required")
    assert isinstance(profiles, Mapping)
    _require(training.get("default_profile") in profiles, "training.default_profile is unknown")
    for name, profile in profiles.items():
        _validate_profile(str(name), profile)
    pilot = training.get("pilot")
    _require(isinstance(pilot, Mapping), "training.pilot must be a mapping")
    assert isinstance(pilot, Mapping)
    for key in (
        "max_steps",
        "limit_train_examples",
        "limit_dev_examples",
        "save_steps",
        "eval_steps",
        "logging_steps",
    ):
        _require(_number(pilot.get(key), int) and pilot[key] >= 1, f"training.pilot.{key}")

    evaluation = raw.get("evaluation")
    _require(isinstance(evaluation, Mapping), "config evaluation must be a mapping")
    assert isinstance(evaluation, Mapping)
    _require(
        tuple(evaluation.get("thresholds") or ()) == PROTOCOL_THRESHOLDS,
        "evaluation thresholds must stay 0.3 and 0.7",
    )
    _require(evaluation.get("max_width") == PROTOCOL_MAX_WIDTH, "evaluation max_width must be 24")
    _require(evaluation.get("batch_size") == 1, "evaluation batch_size must stay 1")
    _require(evaluation.get("dtype") == "fp32", "evaluation dtype must stay fp32")
    gateway_types = set(ontology.values()) - {UNMAPPED} | {"DATE_TIME"}
    for key in ("target_types", "highlighted_types"):
        values = evaluation.get(key)
        _require(
            isinstance(values, list) and set(values) <= gateway_types,
            f"evaluation.{key} must be gateway types",
        )
    gate = evaluation.get("safety_gate")
    _require(isinstance(gate, Mapping), "evaluation.safety_gate must be a mapping")
    assert isinstance(gate, Mapping)
    _require(
        isinstance(gate.get("major_types"), list) and set(gate["major_types"]) <= SUPPORTED_TYPES,
        "safety_gate.major_types must be gateway-supported types",
    )
    for key in (
        "supported_only_f1_max_drop",
        "major_type_f1_max_drop",
        "native_label_f1_max_drop",
        "other_numeric_absorbed_max_fraction",
        "tp_lost_to_other_numeric_max_fraction",
    ):
        _require(_number(gate.get(key), float) and 0 <= gate[key] < 1, f"safety_gate.{key}")
    for key in ("major_type_min_support", "native_label_min_support"):
        _require(_number(gate.get(key), int) and gate[key] >= 1, f"safety_gate.{key}")


def load_config(
    path: Path = DEFAULT_CONFIG,
    label_map: GlinerLabelMap | None = None,
    ontology: Mapping[str, str] | None = None,
) -> NumericConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    label_map = load_label_map() if label_map is None else label_map
    ontology = load_ontology() if ontology is None else ontology
    validate_config(raw, label_map, ontology)
    return NumericConfig(raw=raw, sha256=canonical_sha256(raw), label_map=label_map)


def require_label_identifiers(labels: Sequence[str]) -> None:
    for label in labels:
        _require(bool(_LABEL_PATTERN.fullmatch(label)), "labels must be lower-case identifiers")


def verify_snapshot_revision(snapshot_dir: Path, revision: str) -> None:
    """A Hugging Face cache snapshot directory is named by its full commit hash."""
    _require(
        re.fullmatch(r"[0-9a-f]{40}", revision) is not None,
        "the pinned revision must be a full commit hash",
    )
    _require(snapshot_dir.name == revision, "loaded snapshot does not match the pinned revision")
    _require((snapshot_dir / "gliner_config.json").is_file(), "snapshot has no gliner_config.json")


def verify_architecture(model: Any, base: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse a checkpoint whose class, span layer, encoder, or splitter differs."""
    config = model.config
    found = {
        "architecture": type(model).__name__,
        "span_mode": getattr(config, "span_mode", None),
        "encoder": getattr(config, "model_name", None),
        "words_splitter_type": getattr(config, "words_splitter_type", None),
    }
    for key, value in found.items():
        _require(value == base[key], f"loaded checkpoint {key} differs from the experiment")
    return found
