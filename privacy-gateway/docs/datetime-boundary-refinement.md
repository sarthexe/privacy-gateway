# Offline DATE_TIME candidate boundary refinement

This isolated experiment reads original source text and exact persisted GLiNER
predictions. Phone/email/card experiments, production, model, thresholds,
max_width, ontology and strict scorer are unchanged. Credit-card experiments
remain negative offline results; neither card variant is promoted.

## Fixed source-only rules

- Calendar patterns: numeric year-first dates with four-digit year and one
  consistent `-`, `/` or `.` separator; four-digit-year-last numeric dates only
  when their month/day order is uniquely determined (equal day/month is allowed);
  fixed English month-name/abbreviation dates in day-month-year or month-day-year
  form, optionally with a valid ordinal and internal comma.
- No locale/language/dataset selection, two-digit-year inference, missing-year
  inference, bare weekdays, relative-date understanding, or broad NLP.
- Times: hour:minute, optional seconds/fraction (up to nine digits), optional
  AM/PM or a.m./p.m.; hour-only requires AM/PM. Validate clock ranges.
- Immediate dates plus times, separated by `T` or horizontal whitespace, form a
  single maximal candidate. Do not split an attached date/time into smaller values.
- Preserve attached Z, fixed uppercase timezone abbreviations, and signed offsets
  HHMM/HH:MM, including named-zone offsets. Abbreviations are opaque suffixes:
  do not infer location or convert timezones. Validate offset hour/minute ranges.
- Reject ambiguous numeric dates, invalid calendars/clocks/offsets, mixed date
  separators, connected numeric fragments, and repeated timezone attachments.
  Reject unknown uppercase suffixes that could be timezone tags rather than
  trimming them. Do not back off to a bare date before a malformed attached clock.
- Calendar validation uses the proleptic Gregorian calendar, years 1–9999.
  Leap seconds are not supported. These are coverage limits, not GT exclusions.
- The regex extracts a complete structural core. A provisional raw envelope adds
  up to three immediately adjacent peripheral opening/closing punctuation marks.
  Refine only that generated envelope to its unique validated core; preserve all
  digits, date/time attachment, AM/PM and timezone characters.
- Never refine a model prediction, infer a core from GT, expand toward GT, or
  suppress a conflict. Append only new canonical DATE_TIME spans absent from
  existing model date/time aliases. Score 1.0 denotes rule acceptance, not model
  confidence; persisted thresholds are not changed.

## Interpretation of diagnostics

Raw/core boundary relations use GT only after candidate decisions: exact,
candidate inside GT, GT inside candidate, partial overlap or no DATE_TIME overlap.
Report per-class raw/valid/rejected/refined/unchanged/ambiguous counts and per-class
exact model duplicates at each threshold. Embedded components consumed by a
maximal composite are not separately appended.

Recovery accounting separates recovered FN with existing nonexact DATE_TIME
overlap from other recovered FN; appended FP are measured separately. A core
matching GT when its raw punctuation envelope does not is a boundary-conformance
observation, not proof that trimming alone fixes the model. Existing model
boundary FP remain because all original predictions must survive.

## Verification and reproduction

All normalized and prediction bytes must match frozen evidence. Reproduce both
aggregate scoring views and DATE_TIME baseline metrics. Verify preservation of
every original model object, label, score, boundary, order and multiplicity.
Export only aggregates, never dates/times, snippets, offsets or source IDs.

From `privacy-gateway/`:

```sh
python -m pytest tests/test_gliner_datetime_boundary_refinement.py \
  tests/test_gliner_phone_hybrid_experiment.py \
  tests/test_gliner_phone_boundary_refinement.py \
  tests/test_gliner_credit_card_experiment.py \
  tests/test_gliner_email_hybrid.py tests/test_analyze_gliner_ood_errors.py -q
python -m ruff check scripts/gliner_datetime_boundary_refinement.py \
  tests/test_gliner_datetime_boundary_refinement.py
python -m py_compile scripts/gliner_datetime_boundary_refinement.py \
  tests/test_gliner_datetime_boundary_refinement.py
```

For each dataset/split pair ai4privacy/validation, gretel/test, argilla/train:

```sh
python scripts/gliner_datetime_boundary_refinement.py \
  --dataset argilla \
  --normalized /tmp/ood-reconstruction/argilla/argilla_train.jsonl \
  --predictions /tmp/ood-reconstruction/predictions/argilla/gliner_ood_t0.3.jsonl \
  --predictions /tmp/ood-reconstruction/predictions/argilla/gliner_ood_t0.7.jsonl \
  --baseline-report ../.local/outputs/phase15-ood-root-cause.json \
  --input-reference ../.local/outputs/phone-boundary-refinement-experiment.json \
  --report /tmp/datetime-results/argilla.json
```

Use new external report paths; never overwrite inputs. Leave all new work
uncommitted for review. Do not promote the experiment into production.

## Measured results

All six original baselines reproduced exactly, in both overall scoring views and
DATE_TIME-specific counts. Both thresholds were evaluated together per dataset:
AI4Privacy 188.195 seconds (47,725 examples), Gretel 19.474 seconds (5,000),
Argilla 7.372 seconds (2,096). These are measured concurrent-run timings, not
standalone performance guarantees.

| Dataset | Threshold | Baseline TP/FP/FN | Variant TP/FP/FN | Delta TP/FP/FN | DATE_TIME F1 baseline → variant |
|---|---:|---|---|---|---|
| AI4Privacy | 0.3 | 31148/26003/3602 | 31937/26243/2813 | +789/+240/-789 | 0.677860 → 0.687335 |
| AI4Privacy | 0.7 | 30434/21258/4316 | 31349/21527/3401 | +915/+269/-915 | 0.704148 → 0.715518 |
| Gretel | 0.3 | 3665/1425/140 | 3724/1545/81 | +59/+120/-59 | 0.824058 → 0.820807 |
| Gretel | 0.7 | 3649/1157/156 | 3719/1286/86 | +70/+129/-70 | 0.847521 → 0.844268 |
| Argilla | 0.3 | 451/70/22 | 451/72/22 | 0/+2/0 | 0.907445 → 0.905622 |
| Argilla | 0.7 | 451/33/22 | 451/35/22 | 0/+2/0 | 0.942529 → 0.940563 |

| Dataset | Threshold | Recovered same-type mismatch FN | Other recovered FN | Added FP | Core exact / raw nonexact |
|---|---:|---:|---:|---:|---:|
| AI4Privacy | 0.3 | 591 | 198 | 240 | 213 |
| AI4Privacy | 0.7 | 539 | 376 | 269 | 257 |
| Gretel | 0.3 | 57 | 2 | 120 | 32 |
| Gretel | 0.7 | 52 | 18 | 129 | 41 |
| Argilla | 0.3 | 0 | 0 | 2 | 0 |
| Argilla | 0.7 | 0 | 0 | 2 | 0 |

**Observations:** AI4Privacy precision/recall/F1 improve at both thresholds.
Gretel recall improves but precision/F1 decline; 98/105 appended FP are cores
inside longer GT. Argilla has no recovery. No core partial-overlap candidates
remain. This is a mixed cross-dataset result, not a robust general improvement.

**Hypotheses, not evaluated:** A generic source/model-anchored eligibility and
completeness policy might reduce new subspan FP. A separate paired raw/refined
ablation could isolate peripheral boundary refinement's contribution.

**Recommendation:** Retain as an offline experiment and evidence. Do not promote
the current variant. Another separately specified DATE_TIME variant is justified,
but no second variant or post-evaluation rule tuning was performed here.

**Limitations:** Raw envelopes are provisional peripheral expansions around
recognized structural cores, not original model predictions or an independently
inferred broad detector output. Raw/core exact-match differences do not isolate
the causal effect of trimming. Full source composites may disagree with GT's
component annotations. A strict FP, especially one without DATE_TIME GT overlap,
does not by itself establish semantically non-date content. No annotation was
changed to accommodate these differences.

198 tests passed in 1.12 seconds: 45 new DATE_TIME tests and 153 frozen tests.
Ruff and py_compile passed. Every original prediction's order, multiplicity,
label, score and boundaries survive. All nine dataset/prediction hashes match
frozen evidence and remain unchanged. All preexisting tracked implementation,
mapping, scoring, production and frozen experiment files are byte-identical.
Only project-memory bookkeeping changed among preexisting tracked files.
HEAD remains unchanged; new work is uncommitted.

Complete precision/recall/F1, deltas, both overall views, per-class candidate
accounting, raw/core relations, input fingerprints and exact executed commands:

- `.local/outputs/datetime-boundary-refinement.json`
- `.local/outputs/datetime-boundary-refinement.html`
- `.local/outputs/datetime-preimplementation.json` (frozen-file audit baseline)

These paths are relative to the workspace root. CLI outputs first went to new
external `/tmp/datetime-results/{ai4privacy,gretel,argilla}.json` paths, then were
consolidated promptly in the workspace. Repeat runs must choose fresh external
report paths because overwriting is intentionally forbidden.