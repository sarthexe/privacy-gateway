# Offline PHONE_NUMBER hybrid — Variant A

This script is an isolated experiment, not a production detector. It reads the
exact normalized OOD files and both existing threshold prediction files. It
does not import the GLiNER runtime, run inference, change model configuration,
write predictions, or alter the strict scorer. Ground truth is used only for
evaluation, never candidate generation, validation, or merging.

## Dependency and region policy

The experiment uses the already installed `phonenumbers` library (the initial
run uses version 9.0.40). Its version is recorded in every report. No production
dependency declarations were changed.

The default is **no region**. Only a leading `+` identifies a region-independent
international number. A national/local candidate is rejected unless the caller
explicitly provides `--region US`, or another uppercase supported region.
There is no inference from locale, language, dataset, prose, or model labels.
An IDD prefix such as `00` does not imply a region in the default policy.

## Extraction and validation

The broad regex finds digit strings with optional leading `+` or `(`, horizontal
spaces, tabs, parentheses, dots, slashes, hyphens, and optional extension markers.
At least seven decimal digits are required for a raw candidate. Newlines are not
consumed. Exact spans are deduplicated. Spans longer than 128 characters are
retained in raw statistics but rejected before parsing.

The parser must successfully parse the candidate and return `is_valid_number`.
Validity means compatibility with numbering-plan metadata, not that a number
is assigned, reachable, or actually PII. Only the parser normalizes; the source
text and original `[start,end)` offsets remain unchanged. Raw and validated
candidate spans are retained separately in memory, never exported.

This first broad extractor can combine adjacent numeric groups or include
punctuation/extension text that differs from annotation boundaries. It does not
split, trim, repair, or expand candidates. Those are limitations to measure,
not reasons to consult GT during extraction.

## Append-only merge

Every baseline native prediction, including unsupported and unmapped labels,
is retained with its original order, offsets, label, score, and multiplicity.
Validated candidates are appended with native label `phone_number` unless an
existing PHONE_NUMBER prediction has the same offsets. Existing fax aliases
mapped to PHONE_NUMBER also prevent duplicates. Non-phone predictions with the
same offsets do not prevent adding a different-type phone candidate.

The synthetic score `1.0` is only a validation acceptance marker, not model
confidence. Deterministic candidates are not filtered by the model threshold.
There is no suppression or arbitration. Existing phone boundary errors remain
FP even if an additional exact candidate recovers FN.

## Evaluation and report semantics

Both thresholds are evaluated in one dataset stream. The unchanged `score_batch`
strict scorer evaluates full and gateway-supported-only views. Baseline metrics
must reproduce the saved benchmark; PHONE_NUMBER metrics and all six deltas are
reported separately.

Boundary counts classify each candidate once, prioritizing exact, candidate
contained by GT, GT contained by candidate, partial overlap, then no phone-GT
overlap. Raw and validated candidate tables are separate. They count candidates,
not overlapping pairs or GT entities, and sum to their respective candidate
totals. These counts are shared between the two thresholds; added-candidate
counts differ because duplicate baseline phone predictions are skipped.

Per-record checks preserve every baseline prediction and the complete ordered
non-phone sequence. Aggregate checks enforce zero removals, zero changed
non-phone predictions, matching prediction-count deltas, and unchanged input
hashes. Recovered FN evidence distinguishes existing nonexact phone overlaps,
other native prediction overlaps, and no native prediction overlap. Extra FP
coverage diagnostics are overlap evidence, not semantic adjudication.

Reports contain only aggregate metrics, counts, policy, parser version, input
hashes, and runtime. No values, snippets, raw offsets, or example IDs are written.
Existing outputs cannot be overwritten; CLI reports must be outside the repository.

## Commands

From `privacy-gateway/`:

```sh
python -m pytest tests/test_gliner_phone_hybrid_experiment.py -q
python scripts/gliner_phone_hybrid_experiment.py \
  --normalized /tmp/ood-reconstruction/ai4privacy/ai4privacy_validation.jsonl \
  --predictions /tmp/ood-reconstruction/predictions/ai4privacy/gliner_ood_t0.3.jsonl \
  --predictions /tmp/ood-reconstruction/predictions/ai4privacy/gliner_ood_t0.7.jsonl \
  --report /tmp/phone-hybrid-results/ai4privacy.json
```

Repeat with `gretel/gretel_test.jsonl` and `argilla/argilla_train.jsonl`,
their corresponding prediction directories, and distinct report paths.
The six initial conditions omit `--region`. No second variant is implemented.