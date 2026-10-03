---
name: OOD ground-truth integrity
description: Evaluation decisions for inconsistent public-dataset metadata and invalid annotations.
---

Treat public benchmark annotation boundaries and labels as immutable ground truth. AI4Privacy value metadata is not authoritative when it disagrees with the source slice. Invalid offsets or labels must not be clipped, reconstructed, repaired, or silently removed.

**Why:** The user explicitly confirmed that accepting authoritative valid offsets while ignoring inconsistent value metadata is correct. Repairing invalid spans or retaining only an example's valid spans would change the benchmark ground truth and make evaluation results misleading.

**How to apply:** Keep normalization fail-closed by default. Any permitted exclusion must be an explicit opt-in that removes the whole example, preserves original source indices for retained examples, and records source-independent reasons and verifiable counts/hashes. Never include source text or entity values in exclusion audits.

## Ontology fingerprint representations

OOD benchmark reproducibility records use the SHA256 of the ontology YAML file bytes. Normalization metadata uses the SHA256 of the canonical JSON label mapping. These are different fingerprints even though both fields are named `ontology_sha256`.

**Why:** Comparing the two representations directly produces a false ontology-mismatch diagnosis even when the ontology is unchanged.

**How to apply:** Verify benchmark fingerprints against raw file bytes and normalization fingerprints against the canonical mapping. Never edit the ontology to make these two different fingerprints equal.

## Restoring ephemeral normalized inputs

Keep pinned benchmark adapters unchanged when restoring missing temporary inputs. A temporary restoration driver can preload an immutable label map rather than reload it for each row, but its output must match the previously recorded byte-level SHA256 exactly.

**Why:** Temporary evaluation inputs can disappear across environment resets. Repeated label-map loading dominated reconstruction time; preloading the same mapping restored byte-identical data without modifying the frozen adapter or regenerating predictions.

**How to apply:** Reuse the pinned source revision and the original whole-example exclusion policy. Verify normalized bytes and decompressed persisted prediction bytes against the frozen evidence before evaluation. An optimization is not permission to repair annotations or change the benchmark.

## Credit-card validity and annotation coverage

Keep checksum-invalid CREDIT_CARD GT in the benchmark. Analyze checksum-valid, checksum-invalid, and unable-to-validate GT separately. Zero-GT-support CREDIT_CARD predictions are coverage diagnostics, not confirmed semantic detector failures, and must not drive arbitration rules.

**Why:** The user explicitly required these distinctions for offline candidate generation and type arbitration.

**How to apply:** Fix candidate and arbitration rules from source/model-output evidence before evaluation. Use GT only afterward for scoring and consequences; never use a checksum to exclude otherwise valid annotations.

## Frozen phone experiments

Both PHONE_NUMBER candidate generation and phone boundary refinement are reviewed and frozen for offline experimentation. Keep them unchanged during separately scoped detector experiments.

**Why:** The user explicitly declared both phone variants frozen before beginning the credit-card experiment.

**How to apply:** Implement new detector trials in isolated files and verify the frozen phone files remain unchanged.

## Credit-card experiment disposition

The CREDIT_CARD experiment is a NEGATIVE RESULT and must remain offline-only.
Do not promote either existing CREDIT_CARD variant.

**Why:** The user explicitly specified this disposition before the DATE_TIME experiment.

**How to apply:** Preserve the card experiment during separately scoped work;
do not treat semantic checksum validity as evidence supporting production promotion.

## Frozen previous date/time experiment controls

The previous append-only DATE_TIME experiment must preserve every model prediction, including
existing DATE_TIME predictions. Only newly generated candidates may differ.
Do not use GT to choose boundaries or language/dataset identity to choose date formats.
Preserve legitimate timezone suffixes, AM/PM and date-time attachment even when GT
uses different boundaries.

**Why:** The user explicitly required source-only boundary quality experiments,
not benchmark-specific annotation matching or production changes.

**How to apply:** Generate and refine from original source text before GT diagnostics.
Keep decisions deterministic and generic, verify append-only preservation and
frozen input bytes, and leave new experiments uncommitted for review.

## Closed date/time experiments

DATE_TIME experimentation is CLOSED. Preserve both previous offline variants
and do not implement another DATE_TIME heuristic.

**Why:** The user explicitly closed DATE_TIME after the anchored experiment
produced limited AI4Privacy gain and no Gretel/Argilla change.

**How to apply:** Treat the date/time variants as frozen evidence, not a starting
point for another heuristic. CREDIT_CARD must not be revisited yet; phone
variants remain frozen.

## Location anchored experiment stopping rule

For LOCATION, anchor each proposed replacement to an existing model prediction,
preserve the original separately, and retain label, score, order and multiplicity.
No document-wide extraction, gazetteer, geographic hierarchy inference,
capitalisation-only expansion, PERSON↔LOCATION suppression, GT-selected
boundaries or dataset-specific rules.

**Why:** The user explicitly requested a generic source-only offline test of
LOCATION truncation repair, not broad new prediction generation.

**How to apply:** Keep inference, production, persisted predictions, scorer,
ontology and other experiments unchanged; leave new work uncommitted. If the
effect is tiny or dataset-specific, STOP LOCATION experimentation rather than
adding more heuristics.

LOCATION experimentation is now stopped. Retain the anchored result as offline
negative/limited evidence only; do not promote it or implement another variant.

**Why:** The target truncation cohort had no recovery, and the tiny net gain
was dataset-specific, triggering the user's explicit stopping rule.

**How to apply:** Preserve the result for review without treating aggregate
F1 improvement as grounds to restart LOCATION heuristics.

## PERSON analysis-only phase

PERSON work is root-cause/type-conflict analysis only until a separate controlled
experiment is authorized. Do not build a PERSON recognizer or implement
cross-type suppression/arbitration in this phase. LOCATION is closed and its
anchored offline experiment must remain unchanged.

**Why:** The user explicitly required analysis of persisted predictions before
any PERSON experiment, and warned that overlap is not proven semantic confusion.

**How to apply:** Keep inference, production, prediction artifacts, strict scorer,
ontology and frozen experiments unchanged. Use GT only for diagnostics, export
aggregate counts without source values/offsets/example IDs, and leave work
uncommitted. Proposed future experiments must be generic and source/model-
anchored, without GT decisions, gazetteers, capitalization alone or broad name
extraction.