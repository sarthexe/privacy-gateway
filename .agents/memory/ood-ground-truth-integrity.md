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