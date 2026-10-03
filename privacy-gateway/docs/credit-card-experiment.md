# Offline CREDIT_CARD generation and conservative type arbitration

This is a separately scoped experiment. Phone A/B and email experiments, model,
thresholds, max_width, production, ontology, persisted predictions, and scorer
remain frozen. Only this experiment reads the source and existing model outputs.
No model runtime is imported; scoring uses the existing strict harness.

## Rules fixed before evaluation

- Extract maximal ASCII-digit/horizontal-space/hyphen spans, bounded by non-word
  characters; never cross lines. No splitting a long numeric run into PANs.
- Normalize separators only in memory for validation. Keep original digit-to-digit
  source boundaries unchanged. Accept 13–19 digits, with a 128-character cap.
- Contiguous digits are allowed. Separated numbers must use either single
  hyphens or horizontal whitespace, never both. At most eight groups, each
  one to six digits. This includes common 4-4-4-4 and 4-6-5 formats.
- Reject connected signs, masking, and dot/slash-connected numeric fragments.
- Candidate A accepts Luhn-valid spans, or checksum-invalid spans immediately
  preceded by an explicit card label within 64 characters on the same line.
  Fixed labels: credit/debit/payment card, optionally number/no.; card number/no.;
  cc number/no. Optional punctuation `:`, `=`, `#`, `-` may precede the value.
- There is no issuer-prefix inference, assignment claim, locale inference,
  dataset rule or GT-derived acceptance. The all-zero value can pass Luhn;
  this checksum is structural evidence, not evidence of a real card.

## Arbitration B — exact rule

Begin with precisely the same accepted candidates as A. For each candidate:

1. Inspect only original non-CREDIT_CARD predictions directly overlapping it.
2. Require both Luhn validity and the explicit source card context above.
3. Every such overlap must have identical boundaries and native label
   `account_number`, `national_id`, `customer_id`, or `employee_id`, mapped to
   BANK_ACCOUNT or ID_CARD.
4. Only then suppress those exact competitors. Preserve every other original
   prediction and its order, score, boundaries, label and multiplicity.
5. If any overlap is partial/containing, protected/other type, or lacks priority
   evidence, reject arbitration as ambiguous and keep A's append-only outcome.

No unrelated suppression, no cross-type score threshold, no relabeling, no GT.
Existing CREDIT_CARD predictions (including cvv aliases) are never suppressed;
exact same-type source spans prevent duplicate additions. Appended score 1.0
is an acceptance marker, not model confidence. Model thresholds are unchanged.
Duplicate model competitors are counted individually when suppressed.

## Diagnostics and checksum coverage

GT is used only after decisions for scoring and consequences. Every CREDIT_CARD
annotation remains in the benchmark. Partition source slices into checksum-valid,
checksum-invalid, or unable to validate as a full PAN. The unchanged ontology maps
cvv to CREDIT_CARD; short CVV annotations fall into unable-to-validate. Invalid
grouping, masked/non-ASCII values and unsupported lengths also fall there.

Report partitions by native label, strict TP/FN in each partition, acceptance-tier
TP/FP, and a Luhn-only-additions ablation that keeps all baseline predictions and
all GT. This measures the contextual checksum-invalid branch without re-inference.

Report full and seven-type supported-only strict metrics plus CREDIT_CARD metrics
for baseline/A/B. Suppression consequences include exact GT matches lost versus
nonmatching predictions removed, by canonical type. Zero-GT support, especially
AI4Privacy, is a separate annotation-coverage diagnostic; its strict FP must not
be called confirmed semantic errors or used to choose arbitration rules.

Counters distinguish candidate/model overlap pairs, conflicted candidates,
resolved pairs/candidates, ambiguous decisions, individual suppressed predictions,
native/canonical suppression categories, additions, duplicates and survivors.
“Predictions changed” excludes explicitly reported deletion/suppression and is
zero: no existing object is relabeled or rewritten.

## Limits and integrity

PAN-like IDs/accounts can pass Luhn; context is narrow and lexical. Checksum-invalid
GT may not have context, and CVV/masked GT cannot be recovered by this full-PAN
generator. A maximal regex can join neighboring groups. This is deliberate,
measured coverage loss, not permission to split using GT. Exact competing spans
may themselves be correct unsupported GT; all such losses must be disclosed.

CLI reports are new external files; source data and prediction artifacts are
read-only. Baseline metrics must reproduce frozen diagnostics. Both fixed
threshold identities must otherwise match. Final repository fingerprint auditing
protects all frozen code, mappings, benchmarks and phone/email files. Reports
contain aggregates only, never card values, snippets, offsets or example IDs.

## Commands

From `privacy-gateway/`:

```sh
python -m pytest tests/test_gliner_credit_card_experiment.py -q
python scripts/gliner_credit_card_experiment.py \
  --dataset argilla \
  --normalized /tmp/ood-reconstruction/argilla/argilla_train.jsonl \
  --predictions /tmp/ood-reconstruction/predictions/argilla/gliner_ood_t0.3.jsonl \
  --predictions /tmp/ood-reconstruction/predictions/argilla/gliner_ood_t0.7.jsonl \
  --baseline-report ../.local/outputs/phase15-ood-root-cause.json \
  --input-reference ../.local/outputs/phone-boundary-refinement-experiment.json \
  --report /tmp/credit-card-results/argilla.json
```

Repeat with ai4privacy/validation and gretel/test, without changing rules.
Leave this experiment uncommitted for review.

## Completed evaluation — interpretation

All six baseline conditions reproduce the frozen full, supported-only and card
metrics. There is no inference, production modification or prediction regeneration.
Restoring temporary normalized files is not prediction regeneration: all three
normalized hashes and six decompressed prediction hashes match frozen evidence.

| Dataset | GT | Checksum-valid | Checksum-invalid | Unable to validate |
|---|---:|---:|---:|---:|
| AI4Privacy | 0 | 0 | 0 | 0 |
| Gretel | 663 | 432 | 218 | 13 |
| Argilla | 180 | 19 | 113 | 48 |

Argilla's 48 unable cases are mapped CVV GT, not discarded annotations. Across
Argilla's 132 PAN annotations, 113 fail Luhn: global Luhn-only GT filtering would
materially alter this benchmark and is prohibited.

| Dataset / threshold | Baseline card TP/FP/FN | A and B card TP/FP/FN | Baseline F1 | A/B F1 |
|---|---|---|---:|---:|
| AI4Privacy / 0.3 | 0/1921/0 | 0/2198/0 | 0 | 0 |
| AI4Privacy / 0.7 | 0/880/0 | 0/1165/0 | 0 | 0 |
| Gretel / 0.3 | 555/68/108 | 633/88/30 | 0.863142 | 0.914740 |
| Gretel / 0.7 | 546/39/117 | 632/59/31 | 0.875000 | 0.933530 |
| Argilla / 0.3 | 102/101/78 | 112/190/68 | 0.532637 | 0.464730 |
| Argilla / 0.7 | 94/60/86 | 107/152/73 | 0.562874 | 0.487472 |

Gretel recovers 78/86 FN with 20 added FP at both thresholds. Argilla recovers
10/13 FN but adds 89/92 FP, lowering precision and F1. Do not keep A as a general
default based on these results. The contextual checksum-invalid addition tier
recovers one FN per Gretel threshold and one at Argilla 0.7, with no added FP in
those conditions; it does not solve the checksum-valid false-positive burden.

B suppresses two exact native national_id predictions per AI4Privacy threshold,
and one exact account_number prediction at Gretel 0.3. All five condition-level
suppressions are nonmatching under the strict benchmark; no exact GT TP is lost.
AI4Privacy remains a zero-card-support diagnostic, not semantic validation for
arbitration. In GT-supported datasets there is only one resolved conflict, with
no CREDIT_CARD or supported-only improvement. Do not keep B as a general default;
its evidence is too limited.

No rules were tuned against outcomes. A context-window word-boundary conformance
fix was checked against all 312,404 raw candidates across 54,821 source records:
zero context decisions changed. Boundary diagnostic labels use CREDIT_CARD,
even though their span-relation helper is shared with the frozen phone experiment.

The complete allowlisted JSON and standalone table report are saved at
`.local/outputs/credit-card-experiment.json` and
`.local/outputs/credit-card-experiment.html` from the workspace root. They include
all three variants, both scoring views, P/R/F1, partition TP/FN, acceptance tiers,
Luhn-only ablation, suppression consequences, exact commands, hashes and invariants.
No plaintext values, snippets, source offsets or example IDs are exported.