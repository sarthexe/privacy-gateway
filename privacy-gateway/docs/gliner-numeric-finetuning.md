# GLiNER numeric fine-tuning experiment (offline)

Research only. This experiment fine-tunes the pinned `nvidia/gliner-PII` checkpoint
(`bd23e8ef4425fd04e34c5204ab49ffaa706eae79`, gliner 0.2.29) so it separates
ambiguous numeric entities, especially `CREDIT_CARD` vs `BANK_ACCOUNT` vs `ID_CARD`,
without losing the 55-label GLiNER-PII vocabulary. It changes nothing in the
gateway: no detection, tokenization, vault, API, routing, policy, ontology, or
label-map code or config is touched, and no model is promoted automatically.

| File | Role |
|---|---|
| `configs/gliner_numeric_finetune.yaml` | Every experiment setting, validated on load |
| `scripts/gliner_numeric_common.py` | Torch-free config validation, vocabulary, leakage and identity guards |
| `scripts/prepare_gliner_numeric_finetune.py` | Dataset builder (no model) |
| `scripts/train_gliner_numeric.py` | Resumable trainer |
| `scripts/evaluate_gliner_numeric_finetune.py` | `predict`, `compare`, `summarize` |

Install on the training machine (CUDA torch first, from the PyTorch index):

```bash
pip install -e ".[evaluation,gliner-evaluation,gliner-finetune]"
```

## Data

**Sources.** Only non-benchmark splits are read:

| Source | Default | Notes |
|---|---|---|
| `nvidia/Nemotron-PII` `train` @ `b70ffaf5…` | on | Validated with `nemotron_pii.normalize_example`; native labels from its spans |
| AI4Privacy `train` | off | Through the pinned `ood_pii` adapter |
| Gretel `train` | off | Through the pinned `ood_pii` adapter |

**Never trained on** (enforced in config validation, the builder, and the
adapter wrapper): AI4Privacy `validation`, Gretel `test`, any Argilla split
(its only split is the benchmark), and Nemotron `test`. Turning on an OOD train
split makes that benchmark partly in-distribution. It therefore needs
`--benchmark-normalized` with the pinned benchmark files (their SHA-256 is
checked), and any training text whose casefolded, whitespace-collapsed form
matches a benchmark text is dropped. The manifest then carries an
`ood_train_sources_enabled…` warning.

**Offsets.** Character offsets from the source are kept exactly. Each span must
start and end on a word boundary of gliner's whitespace splitter
(`\w+(?:[-_]\w+)*|\S`) and be at most 24 words wide. If any span fails that, or
one span carries two labels, the whole example is dropped and counted by
reason. Nothing is re-aligned, trimmed, or guessed. A malformed source
annotation (bad offsets, unknown Nemotron label) aborts the build. OOD examples
with an UNMAPPED span are dropped, since that span would otherwise become an
unlabeled false negative.

**Windows.** Texts longer than 384 words (the model's `max_len`) are cut into
consecutive windows that never split an entity.

**Record format** (gliner 0.2.29 `UniEncoderSpanProcessor`):

```json
{"tokenized_text": ["Order", "12345", "shipped"],
 "ner": [[1, 1, "other_numeric"]],
 "ner_labels": ["…55 native labels…", "other_numeric"],
 "source": "synthetic", "kind": "synthetic", "group": "<16-hex group hash>"}
```

`ner` uses inclusive word indices. `ner_labels` fixes the prompted label set, so
gliner does not sample labels. Labels in the prompt but absent from `ner` are
negatives. Natural examples prompt the 55 native labels; synthetic examples
prompt the 55 plus `other_numeric`. OOD examples prompt only the native labels
their adapter can express. Label order is shuffled per record with a seeded RNG.

**Split.** Examples are grouped by union-find over stable source IDs (Nemotron
`uid`, OOD `uid`/`id`) and near-duplicate text keys. A group goes to dev when
`sha256(seed|dev|group)` falls below `dev_fraction` (5%). Any group containing a
row from the existing 2,000-row Nemotron threshold-selection subset
(`select_dev_indices`, seed `privacy-gateway/gliner-dev/v1`) is reserved and
excluded from both train and dev. The split does not depend on row order.

**Outputs** (default `~/.cache/privacy-gateway/gliner-numeric-finetune/data`;
paths inside the package tree are refused):

| File | Content |
|---|---|
| `train.jsonl` | Training records in a seeded global order, so a prefix is a fair limit |
| `dev.jsonl` | Dev records for eval loss |
| `dev_normalized.jsonl` | The same dev windows as `{text, entities, native_entities}` for strict scoring |
| `manifest.json` | Aggregates only: counts, label frequencies, rejection reasons, revisions, hashes, seed, config hashes |

The manifest never contains text, entity values, or source IDs (a test checks
this). Builder stdout prints counts only.

## Label vocabulary and OTHER_NUMERIC

The 55 native labels are read from `configs/gliner_label_map.yaml` unchanged.
`other_numeric` is a training-only prompt label for contextual numeric
identifiers that are not PII in this vocabulary (invoice, order, transaction
reference, ticket, tracking, confirmation, case, document, and record numbers,
version-like dotted numbers, codes, and so on). It is not in
`configs/pii_ontology.yaml` or the label map, and config validation fails if it
appears in either. At evaluation it maps to no gateway type
(`UNMAPPED`), is removed before every score, and is reported separately.

## Synthetic hard negatives

Deterministic: each example comes from `random.Random("<experiment>|<seed>|<stream>|<index>")`,
with separate train and dev streams. Dev examples whose text matches a train
example are dropped. Every example is built piece by piece, so offsets are
known exactly and re-checked by the aligner. Each example contains one to three
numeric values and no other PII.

| Family | Shape | Label |
|---|---|---|
| `reference_*` | plain, leading zeros, 20–30-digit integers, 3/4-digit groups with space or dash, `INV-`/`ORD_`/`DOC/` prefixes, mixed alphanumerics | `other_numeric` |
| `card_shaped` | 13–19 digits grouped like cards; half pass Luhn, half fail | `other_numeric` |
| `ip_lookalike` | dotted 3–5 part numbers after "firmware version", "build number", … | `other_numeric` |
| `date_lookalike` | `20240311-0042`-style batch and lot codes | `other_numeric` |
| `postcode_lookalike` | 4–5 digit or ZIP+4-shaped error, part, item, and model codes | `other_numeric` |
| `multi_span` | two or three of the above in one sentence | `other_numeric` |
| `contrast_reference` / `contrast_native` | the same value after a reference phrase, then after "employee ID" / "customer ID" | `other_numeric` / `employee_id`, `customer_id` |
| `unlabeled_quantity` | small counts ("12 boxes") | none |

Text varies in delimiter (`: `, ` #`, ` is `, tab, newline, …), wrapping
punctuation, trailing punctuation, extra whitespace, context casing, and
position (start, middle, end).

Luhn policy: Luhn never decides a label. Card-shaped values are `other_numeric`
whether or not they pass, and synthetic data never produces `credit_debit_card`.
Positive card examples come only from annotated training data. Luhn validity is
recorded only as an aggregate count. The contrast pairs make context, not shape,
the deciding signal.

IP, date/time, and postcode hard cases are look-alike negatives only. The
positives for `ipv4`, `ipv6`, `date`, `date_time`, `time`, `postcode`, and
`street_address` come from the natural rehearsal data, so the source
annotation convention is preserved.

## Balancing

All natural windows are kept: this is the 55-label rehearsal set. Focus labels
are `credit_debit_card`, `account_number`, `bank_routing_number`, `customer_id`,
`employee_id`, `national_id`, and `swift_bic`. The focus share is the fraction
of labeled train mentions that are focus labels or `other_numeric`. Counting
mentions rather than windows matters: most Nemotron windows hold many entities,
and about 30% contain at least one focus label.

1. Add all synthetic examples. Only if natural plus synthetic already exceeds
   `max_focus_share` (0.40), drop synthetic examples from the end.
2. Add copies of natural windows that contain a focus label, one round at a
   time (up to `focus_oversample_factor - 1 = 2` extra copies each) in a seeded
   order. Stop before the share would pass 0.40.
3. If the final share is below `min_focus_share` (0.15), record a warning.

The manifest records `focus_share`, `focus_example_share`, copy and synthetic
counts, and per-label mention counts for train, dev, and each train kind
(`train_natural`, `train_oversampled`, `train_synthetic`).

Smoke build: first 200 Nemotron train rows, 200 synthetic train, 40 synthetic
dev, run twice with byte-identical outputs.

| Quantity | Value |
|---|---|
| Rows read / accepted | 200 / 195 (4 misaligned span boundaries, 1 span wider than 24 words) |
| Train records | 499 (187 natural, 112 oversampled, 200 synthetic) |
| Dev records | 53 (13 natural, 40 synthetic) |
| Focus share (mentions) | 0.205 (479 / 2,335) |
| Synthetic card-shaped (train) | 4 Luhn-valid, 4 Luhn-invalid, all `other_numeric` |

The smoke build overweights synthetic data. In the full build, about 8,000
synthetic examples sit beside the full Nemotron train split, so the cap rarely
binds. Read the full counts from that run's manifest.

## Training

`train_gliner_numeric.py`:

- requires `gliner==0.2.29`, `transformers` within gliner's pin
  (`>=4.51.3,<5.17.0`), `accelerate`, and `torch`;
- downloads or reuses exactly the pinned snapshot with
  `snapshot_download(revision=…)`, checks that the snapshot directory name is
  that commit, and records SHA-256 hashes of the weight files;
- loads with `GLiNER.from_pretrained(<snapshot>, max_width=24)` and refuses any
  class other than `UniEncoderSpanGLiNER`, span mode other than `markerV0`,
  encoder other than `microsoft/deberta-v3-large`, or a non-whitespace splitter.
  There is no fallback to a smaller model;
- seeds Python, NumPy, torch, and transformers;
- checks the dataset manifest against the config's `data` and `labels` hashes
  and verifies the train and dev file hashes;
- validates every record (word bounds, width ≤ 24, labels in the 56-label
  vocabulary, each `ner` label present in `ner_labels`);
- runs the real tokenizer over prompt plus words, and drops examples that would
  exceed the tokenizer's `model_max_length` rather than letting gliner truncate them
  silently. Above 1% dropped, the run stops;
- writes `run_identity.json`, `checkpoints/checkpoint-N/`, `final_model/`
  (with `numeric_finetune_lineage.json`, `production_ready: false`), and
  `training_run.json` (status, package versions, hardware, parameter counts,
  overflow counts, loss history, final file hashes). Output paths inside the
  package tree or the Hugging Face cache are refused;
- uses `report_to: none`, no trackers, fp32 unless `--fp16`/`--bf16` is given
  explicitly, and `dataloader_num_workers: 0`.

| Profile | What trains | Memory notes |
|---|---|---|
| `t4_head_only` (default) | span, prompt, and LSTM layers; DeBERTa-v3-large frozen | Batch 1 × 8 accumulation, fp32. The safest T4 choice; least forgetting risk |
| `full_encoder` | everything, encoder gradient checkpointing | 445M params with AdamW in fp32 is about 7 GB before activations. Not validated on a T4: run `--pilot` first, and add `--fp16` if needed (T4 has no bf16) |

Overrides: `--profile`, `--pilot`, `--max-steps`, `--limit-train-examples`,
`--limit-dev-examples`, `--gradient-accumulation-steps`,
`--per-device-train-batch-size`, `--fp16`, `--bf16`,
`--[no-]gradient-checkpointing`. They all become part of the run identity, so
a resume with different settings is refused. `--resume` continues from the
highest-numbered checkpoint. The final model is the last step, not a "best"
checkpoint.

`--pilot` uses `training.pilot` (10 steps, 64 train / 16 dev examples, saves at
step 5) and is the only mode allowed on a `--limit-source-rows` smoke dataset.

### gliner 0.2.29 training API blockers and workarounds

| Blocker in 0.2.29 | Workaround |
|---|---|
| `BaseGLiNER.train_model()` calls `trainer.train()` with no `resume_from_checkpoint` | Build `gliner.training.Trainer` directly, as `train_model` does, and pass the checkpoint |
| `freeze_component()` only warns on an unknown component | Check the name against `_get_freezable_components()` and verify `requires_grad` is off |
| No GLiNER-level gradient checkpointing; the HF flag would call it on the wrapper | Keep `TrainingArguments.gradient_checkpointing=False`; call `gradient_checkpointing_enable(use_reentrant=False)` on the DeBERTa encoder |
| `Trainer.training_step` catches CUDA OOM, logs "Skipping batch", and returns loss 0 | A log handler counts those warnings and a callback aborts the run |
| The processor truncates over-long prompt-plus-text input with only a warning | Pre-tokenize and drop over-limit examples, counted and capped |
| Native `max_width` is 12 | `from_pretrained(max_width=24)` to match the frozen protocol; checked after load |
| Checkpoint format follows `save_safetensors` in the installed transformers | Record final file hashes, whichever format is written |

## Evaluation

Protocol: thresholds 0.3 and 0.7 only, `max_width` 24, batch size 1, fp32, the
model's whitespace splitter, `max_len` windows with a 24-word margin, the
strict exact-span scorer (`batch_evaluation.score_batch`), and the existing
DATE/TIME → DATE_TIME mapping. No threshold tuning.

`predict` prompts the fine-tuned model with 55 + `other_numeric`
(`--labels auto`). The baseline uses the 55 labels, which matches the frozen
phase-1.5 predictions in `artifacts/phase-1.5-ood/<dataset>/gliner_ood_t{0.3,0.7}.jsonl.gz`.
Those files can be the baseline directly. Running the baseline with
`--labels native+other_numeric` gives a same-prompt control.

`compare` checks that both files are for the same normalized file (SHA-256),
the same threshold, the same protocol, fp32, and the pinned lineage. It then
reports, for each model:

- **full strict:** all gateway GT (UNMAPPED GT counts as FN for both models);
- **supported only:** GT and predictions restricted to the gateway-supported
  types;
- **native label space:** native GT vs native predictions, with UNMAPPED and
  `other_numeric` excluded from both;
- **per type:** precision, recall, F1, TP, FP, FN. The highlighted types are
  `CREDIT_CARD`, `BANK_ACCOUNT`, `ID_CARD`, `IP_ADDRESS`, `DATE_TIME`;
- **`other_numeric`:** prediction count, mapped GT spans it overlaps or matches
  exactly, the absorbed fraction, and (on dev) precision and recall against
  synthetic `other_numeric` GT;
- **forgetting:** baseline F1, candidate F1, and delta for every one of the 55
  native labels (`null` when the dataset has no support).

Safety gate (config `evaluation.safety_gate`). Each condition below raises a flag:

| Flag | Condition |
|---|---|
| `supported_only_f1_drop` | supported-only F1 falls by more than 0.005 |
| `major_type_f1_drop` | a supported type with at least 50 GT falls by more than 0.02 F1 |
| `native_label_f1_drop` | a native label with at least 30 GT falls by more than 0.03 F1 |
| `other_numeric_absorbs_supported_entities` | `other_numeric` overlaps more than 0.5% of mapped GT |
| `true_positives_lost_to_other_numeric` | more than 0.1% of baseline TPs are lost where `other_numeric` now overlaps |

Verdicts:

- `flagged`: at least one flag;
- `no_target_improvement`: no target type improved;
- `no_gate_flags_review_required`: no flags and at least one target type improved.

There is no "accepted" verdict. A `CREDIT_CARD` gain never offsets a flag,
and a human decides whether anything goes further.

Run the gate on all three benchmarks at both thresholds, and on
`dev_normalized.jsonl` for in-distribution forgetting.

## Commands

The examples use a Linux T4 machine with the repo at `privacy-gateway/` and
external storage at `$W` (for example `/kaggle/working/gnf`). `$OOD` holds the
pinned normalized benchmarks (`ai4privacy_validation.jsonl`, `gretel_test.jsonl`,
`argilla_train.jsonl`).

Prepare (full):

```bash
python scripts/prepare_gliner_numeric_finetune.py --output-dir $W/data
```

Smoke train on a T4 (first command to run; about 10 optimizer steps):

```bash
python scripts/prepare_gliner_numeric_finetune.py --output-dir $W/smoke-data \
  --limit-source-rows 2000 --synthetic-train 400 --synthetic-dev 80
python scripts/train_gliner_numeric.py --dataset-dir $W/smoke-data \
  --output-dir $W/runs/t4-head-only-pilot --pilot
```

Full train (default `t4_head_only` profile; resumable):

```bash
python scripts/train_gliner_numeric.py --dataset-dir $W/data --output-dir $W/runs/t4-head-only
# after an interruption:
python scripts/train_gliner_numeric.py --dataset-dir $W/data --output-dir $W/runs/t4-head-only --resume
```

Evaluate (fine-tuned model; repeat for each benchmark and `$W/data/dev_normalized.jsonl`):

```bash
python scripts/evaluate_gliner_numeric_finetune.py predict \
  --normalized $OOD/ai4privacy_validation.jsonl \
  --model $W/runs/t4-head-only/final_model --output-dir $W/eval/ai4privacy/finetuned
# baseline for datasets without frozen predictions (e.g. dev):
python scripts/evaluate_gliner_numeric_finetune.py predict \
  --normalized $W/data/dev_normalized.jsonl --model base --output-dir $W/eval/dev/baseline
```

Compare and summarize:

```bash
for t in 0.3 0.7; do
  python scripts/evaluate_gliner_numeric_finetune.py compare \
    --normalized $OOD/ai4privacy_validation.jsonl \
    --baseline artifacts/phase-1.5-ood/ai4privacy/gliner_ood_t$t.jsonl.gz \
    --candidate $W/eval/ai4privacy/finetuned/predictions_t$t.jsonl \
    --output $W/eval/reports/ai4privacy_t$t.json
done
python scripts/evaluate_gliner_numeric_finetune.py summarize $W/eval/reports/*.json \
  --output $W/eval/reports/summary.json
```

`artifacts/` is the outer repository folder, so run `compare` from the outer
repository root or give an absolute path. `summarize` writes `summary.json` and
`summary.md`.

## Privacy

No script logs or persists source text, entity values, or source IDs outside the
external dataset files themselves. Manifests, run records, prediction files
(offsets, labels, and scores only), and reports are aggregate- or offset-only.
Error messages name a file, a line or row number, and a fixed reason.

## Limitations

- The default profile freezes the encoder, so gains may be modest. The
  `full_encoder` profile can learn more, but carries more forgetting risk and
  is unverified on a T4.
- Synthetic contexts are English and template-based. Gains on them may not
  transfer to real documents.
- Nemotron's annotation conventions define the natural positives. OOD
  benchmarks label numeric types differently, so strict F1 can fall even when
  the model behaves sensibly. Read per-type and forgetting tables, not only the
  headline number.
