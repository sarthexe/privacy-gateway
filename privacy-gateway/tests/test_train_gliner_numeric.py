from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import gliner_numeric_common as common
from scripts import prepare_gliner_numeric_finetune as prepare
from scripts import train_gliner_numeric as train
from scripts.gliner_numeric_common import (
    OTHER_NUMERIC,
    PINNED_MODEL_REVISION,
    NumericExperimentError,
    load_config,
    validate_config,
)
from scripts.nemotron_pii import REPOSITORY_ROOT, load_ontology

CONFIG = load_config()
ONTOLOGY = load_ontology()
GOOD_VERSIONS = {
    "gliner": "0.2.29",
    "torch": "2.11.0+cu128",
    "transformers": "5.16.1",
    "accelerate": "1.13.0",
    "numpy": "2.4.6",
}


def raw_config() -> dict[str, Any]:
    return copy.deepcopy(dict(CONFIG.raw))


def args(**overrides: Any) -> argparse.Namespace:
    parsed = train.parse_args([])
    for key, value in overrides.items():
        setattr(parsed, key, value)
    return parsed


# --- Config, vocabulary, and revision checks ---------------------------------------------


def test_default_config_is_valid_and_pins_the_base_model() -> None:
    base = CONFIG.raw["base_model"]
    assert (base["model_id"], base["revision"]) == ("nvidia/gliner-PII", PINNED_MODEL_REVISION)
    assert (base["model_id"], base["revision"]) == (
        CONFIG.label_map.model_id,
        CONFIG.label_map.model_revision,
    )
    assert CONFIG.training_labels == (*CONFIG.label_map.labels, OTHER_NUMERIC)
    assert OTHER_NUMERIC not in ONTOLOGY
    assert CONFIG.training["report_to"] == "none"
    assert not any(
        profile["fp16"] or profile["bf16"] for profile in CONFIG.training["profiles"].values()
    )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("base_model", "revision"), "main", "pinned"),
        (("base_model", "gliner_version"), "0.2.30", "0.2.29"),
        (("labels", "training_only"), ["other_numeric", "misc"], "only other_numeric"),
        (("labels", "label_map"), "configs/other.yaml", "55-label"),
        (("data", "max_width"), 12, "max_width"),
        (("training", "report_to"), "wandb", "report_to"),
        (("training", "max_width"), 30, "max_width"),
        (("evaluation", "thresholds"), [0.5], "thresholds"),
        (("evaluation", "batch_size"), 8, "batch_size"),
        (("data", "sources", "nemotron_train", "revision"), "main", "pinned"),
        (("data", "hard_negatives", "native_contexts"), {"card": "credit_debit_card"}, "native"),
    ],
)
def test_config_validation_rejects_protocol_changes(
    path: tuple[str, ...], value: Any, message: str
) -> None:
    raw = raw_config()
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(NumericExperimentError, match=message):
        validate_config(raw, CONFIG.label_map, ONTOLOGY)


def test_unpinned_base_requires_explicit_opt_in() -> None:
    raw = raw_config()
    raw["base_model"]["revision"] = "0" * 40
    with pytest.raises(NumericExperimentError, match="allow_unpinned_base"):
        validate_config(raw, CONFIG.label_map, ONTOLOGY)
    raw["base_model"]["allow_unpinned_base"] = True
    validate_config(raw, CONFIG.label_map, ONTOLOGY)


def test_other_numeric_must_stay_out_of_ontology_and_label_map() -> None:
    with pytest.raises(NumericExperimentError, match="ontology"):
        validate_config(raw_config(), CONFIG.label_map, {**ONTOLOGY, OTHER_NUMERIC: "UNMAPPED"})
    extended = common.extended_label_map(CONFIG.label_map)
    assert extended.gateway_type(OTHER_NUMERIC) == "UNMAPPED"
    with pytest.raises(NumericExperimentError):
        common.extended_label_map(extended)


def test_profiles_reject_checkpointing_a_frozen_encoder_and_dual_precision() -> None:
    raw = raw_config()
    raw["training"]["profiles"]["t4_head_only"]["gradient_checkpointing"] = True
    with pytest.raises(NumericExperimentError, match="frozen encoder"):
        validate_config(raw, CONFIG.label_map, ONTOLOGY)
    raw = raw_config()
    raw["training"]["profiles"]["full_encoder"].update(fp16=True, bf16=True)
    with pytest.raises(NumericExperimentError, match="fp16 and bf16"):
        validate_config(raw, CONFIG.label_map, ONTOLOGY)
    raw = raw_config()
    raw["training"]["profiles"]["full_encoder"]["freeze_components"] = ["labels_encoder"]
    with pytest.raises(NumericExperimentError, match="unknown component"):
        validate_config(raw, CONFIG.label_map, ONTOLOGY)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"gliner": "0.2.28"}, "gliner==0.2.29"),
        ({"transformers": "5.17.0"}, "outside"),
        ({"transformers": "4.40.0"}, "outside"),
        ({"accelerate": None}, "missing packages"),
    ],
)
def test_package_versions_are_enforced(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(NumericExperimentError, match=message):
        common.require_training_packages({**GOOD_VERSIONS, **changes})
    assert common.require_training_packages(GOOD_VERSIONS)["gliner"] == "0.2.29"


def test_snapshot_must_be_the_pinned_revision(tmp_path: Path) -> None:
    snapshot = tmp_path / PINNED_MODEL_REVISION
    snapshot.mkdir()
    (snapshot / "gliner_config.json").write_text("{}", encoding="utf-8")
    common.verify_snapshot_revision(snapshot, PINNED_MODEL_REVISION)
    other = tmp_path / ("a" * 40)
    other.mkdir()
    with pytest.raises(NumericExperimentError, match="does not match"):
        common.verify_snapshot_revision(other, PINNED_MODEL_REVISION)
    with pytest.raises(NumericExperimentError, match="full commit"):
        common.verify_snapshot_revision(snapshot, "main")


def fake_model(**config: Any) -> Any:
    base = CONFIG.raw["base_model"]
    values = {
        "span_mode": base["span_mode"],
        "model_name": base["encoder"],
        "words_splitter_type": base["words_splitter_type"],
        **config,
    }
    cls = type(base["architecture"], (), {})
    model = cls()
    model.config = SimpleNamespace(**values)
    return model


def test_architecture_swaps_are_refused() -> None:
    base = CONFIG.raw["base_model"]
    assert common.verify_architecture(fake_model(), base)["architecture"] == base["architecture"]
    with pytest.raises(NumericExperimentError, match="encoder"):
        common.verify_architecture(fake_model(model_name="microsoft/deberta-v3-small"), base)
    with pytest.raises(NumericExperimentError, match="span_mode"):
        common.verify_architecture(fake_model(span_mode="token_level"), base)


def test_freezing_is_strict() -> None:
    class Parameter:
        requires_grad = True

    class Component:
        def __init__(self) -> None:
            self.items = [Parameter()]

        def parameters(self) -> list[Parameter]:
            return self.items

    component = Component()

    class Model:
        def _get_freezable_components(self) -> dict[str, Component]:
            return {"text_encoder": component}

        def freeze_component(self, name: str) -> None:
            for parameter in component.parameters():
                parameter.requires_grad = False

    train.freeze_strictly(Model(), ["text_encoder"])
    assert not component.items[0].requires_grad
    with pytest.raises(NumericExperimentError, match="not present"):
        train.freeze_strictly(Model(), ["decoder"])


# --- Settings ----------------------------------------------------------------------------


def test_pilot_and_cli_overrides() -> None:
    settings = train.resolve_settings(CONFIG, None, args(pilot=True))
    pilot = CONFIG.training["pilot"]
    assert settings["profile"] == CONFIG.training["default_profile"]
    assert settings["max_steps"] == pilot["max_steps"]
    assert settings["limit_train_examples"] == pilot["limit_train_examples"]
    overridden = train.resolve_settings(
        CONFIG, "full_encoder", args(max_steps=3, gradient_accumulation_steps=2, fp16=True)
    )
    assert (overridden["max_steps"], overridden["gradient_accumulation_steps"]) == (3, 2)
    assert overridden["fp16"] and not overridden["bf16"]
    assert overridden["gradient_checkpointing"]
    full = train.resolve_settings(CONFIG, None, args())
    assert full["limit_train_examples"] is None and not full["pilot"]


@pytest.mark.parametrize(
    ("profile", "overrides", "message"),
    [
        (None, {"fp16": True, "bf16": True}, "fp16 and bf16"),
        (None, {"gradient_checkpointing": True}, "trainable encoder"),
        (None, {"max_steps": 0}, "positive"),
        ("missing", {}, "unknown training profile"),
    ],
)
def test_invalid_settings_are_rejected(
    profile: str | None, overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(NumericExperimentError, match=message):
        train.resolve_settings(CONFIG, profile, args(**overrides))


def test_training_arguments_disable_trackers_and_seed_everything(tmp_path: Path) -> None:
    settings = train.resolve_settings(CONFIG, "full_encoder", args())
    kwargs = train.training_arguments_kwargs(settings, tmp_path, 17, "none")
    assert kwargs["report_to"] == "none"
    assert kwargs["seed"] == kwargs["data_seed"] == 17
    assert kwargs["gradient_checkpointing"] is False
    assert kwargs["dataloader_num_workers"] == 0
    assert kwargs["fp16"] is False and kwargs["bf16"] is False
    assert kwargs["load_best_model_at_end"] is False
    with pytest.raises(NumericExperimentError):
        train.training_arguments_kwargs(settings, tmp_path, 17, "tensorboard")


# --- Run directory and resume ------------------------------------------------------------


def identity(**changes: Any) -> dict[str, Any]:
    return {"experiment": "x", "settings": {"max_steps": 10}, **changes}


def test_run_directory_lifecycle(tmp_path: Path) -> None:
    run = tmp_path / "run"
    assert train.prepare_run_directory(run, identity(), resume=False, hf_cache=None) is None
    with pytest.raises(NumericExperimentError, match="--resume"):
        train.prepare_run_directory(run, identity(), resume=False, hf_cache=None)
    with pytest.raises(NumericExperimentError, match="different run identity"):
        train.prepare_run_directory(run, identity(seed=2), resume=True, hf_cache=None)
    for step in (5, 10, 9):
        (run / "checkpoints" / f"checkpoint-{step}").mkdir(parents=True)
    resumed = train.prepare_run_directory(run, identity(), resume=True, hf_cache=None)
    assert resumed is not None and resumed.name == "checkpoint-10"
    (run / "final_model").mkdir()
    with pytest.raises(NumericExperimentError, match="final model"):
        train.prepare_run_directory(run, identity(), resume=True, hf_cache=None)


def test_run_directory_guards(tmp_path: Path) -> None:
    with pytest.raises(NumericExperimentError, match="outside the repository"):
        train.prepare_run_directory(REPOSITORY_ROOT / "runs", identity(), False, None)
    cache = tmp_path / "hf"
    with pytest.raises(NumericExperimentError, match="Hugging Face cache"):
        train.prepare_run_directory(cache / "hub" / "run", identity(), False, cache)
    stale = tmp_path / "stale"
    stale.mkdir()
    (stale / "x").write_text("x", encoding="utf-8")
    with pytest.raises(NumericExperimentError, match="no run identity"):
        train.prepare_run_directory(stale, identity(), False, None)
    with pytest.raises(NumericExperimentError, match="new"):
        train.prepare_run_directory(tmp_path / "fresh", identity(), True, None)


def test_missing_training_packages_fail_before_any_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing() -> None:
        raise NumericExperimentError("missing packages for fine-tuning: gliner")

    monkeypatch.setattr(train, "require_training_packages", missing)
    run = tmp_path / "run"
    code = train.main(
        ["--dataset-dir", str(tmp_path / "data"), "--output-dir", str(run), "--pilot"]
    )
    assert code == 2 and not run.exists()


# --- Dataset integrity -------------------------------------------------------------------


def small_dataset(tmp_path: Path) -> Path:
    rows = [
        {
            "uid": f"u{index}",
            "text": f"Card 4111 1111 1111 {index:04d} was used.",
            "spans": str([{"start": 5, "end": 24, "label": "credit_debit_card"}]),
        }
        for index in range(30)
    ]
    spec = prepare.SourceSpec("nemotron_train", "nemotron", "train", "r", lambda: rows, 5_000)
    out = tmp_path / "data"
    prepare.build_dataset(CONFIG, [spec], out, synthetic_counts=(10, 4))
    return out


def test_manifest_and_files_must_match_the_config(tmp_path: Path) -> None:
    dataset = small_dataset(tmp_path)
    manifest = train.load_manifest(dataset, CONFIG)
    assert manifest["experiment"] == CONFIG.raw["experiment"]
    raw = raw_config()
    raw["data"]["dev_fraction"] = 0.1
    changed = common.NumericConfig(raw, common.canonical_sha256(raw), CONFIG.label_map)
    with pytest.raises(NumericExperimentError, match="data config"):
        train.load_manifest(dataset, changed)
    with (dataset / "train.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("{}\n")
    with pytest.raises(NumericExperimentError, match="does not match"):
        train.load_manifest(dataset, CONFIG)


def test_records_are_validated_and_stripped(tmp_path: Path) -> None:
    dataset = small_dataset(tmp_path)
    records = train.read_records(dataset / "train.jsonl", CONFIG.training_labels, 24, limit=5)
    assert len(records) == 5
    assert all(set(record) == set(train.MODEL_KEYS) for record in records)
    bad = tmp_path / "bad.jsonl"
    base = {"tokenized_text": ["a", "b"], "ner_labels": ["ssn"]}
    for ner in ([[0, 5, "ssn"]], [[0, 1, "email"]], [[1, 0, "ssn"]]):
        bad.write_text(json.dumps({**base, "ner": ner}) + "\n", encoding="utf-8")
        with pytest.raises(NumericExperimentError, match="invalid span"):
            train.read_records(bad, CONFIG.training_labels, 24)
    bad.write_text(
        json.dumps({**base, "ner": [], "ner_labels": ["not_a_label"]}) + "\n", encoding="utf-8"
    )
    with pytest.raises(NumericExperimentError, match="invalid labels"):
        train.read_records(bad, CONFIG.training_labels, 24)


def test_run_identity_changes_with_settings(tmp_path: Path) -> None:
    dataset = small_dataset(tmp_path)
    manifest = train.load_manifest(dataset, CONFIG)
    pilot = train.run_identity(
        CONFIG, train.resolve_settings(CONFIG, None, args(pilot=True)), manifest, dataset
    )
    full = train.run_identity(
        CONFIG, train.resolve_settings(CONFIG, None, args()), manifest, dataset
    )
    assert pilot != full
    assert pilot["base_model"]["revision"] == PINNED_MODEL_REVISION
    assert json.dumps(pilot).count("4111") == 0
