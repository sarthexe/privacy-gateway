# PERSON-in-email suppression experiment

Offline scoring experiment. It does not change the production detector, the GLiNER
model, the threshold, `max_width`, the ontology, or the strict scorer.

Persisted predictions stay on disk unchanged. The scorer sees a temporary view.

## Email validity

A candidate is a span returned by the frozen `recognize_email_addresses` helper in
`scripts/gliner_email_hybrid.py`. This experiment imports that function and does
not change it.

The helper is a deterministic expression. It accepts a local part of dot-separated
atoms, `@`, domain labels, and an alphabetic TLD of 2 to 63 characters. Lookarounds
leave surrounding punctuation outside the span. It does not call DNS or any
network service. Text the helper does not emit is not a candidate, including
invalid addresses and strings the pattern does not uniquely accept. There is no
second parser and no repair step.

## Strict eligibility

A prediction is removed from the temporary view only when all of the following hold:

1. Its pinned label map type is `PERSON` (`first_name` or `last_name`).
2. The same source record contains at least one valid email candidate.
3. The PERSON span is strictly inside that candidate:
   `email_start < person_start < person_end < email_end`.
4. The span does not touch either email boundary and does not extend outside it.
5. The decision uses the source text and the model prediction only.

Overlap alone is not enough. Boundary contact is not enough. An invalid email is
not enough. No prediction is added. `EMAIL_ADDRESS` predictions and every
non-PERSON prediction stay as they are, including labels, offsets, and scores.

The frozen email hybrid uses a weaker containment test (`<=`) and also suppresses
`USERNAME` and `ORGANIZATION`, and it inserts email predictions. This experiment
does not call that hybrid.

## Loose diagnostic

For analysis only, a PERSON prediction is a loose-overlap candidate when its
half-open span overlaps any valid email candidate. Touching endpoints do not
count as overlap. Strict containment implies loose overlap. The loose rule does
not remove anything.

`loose_only_not_strict` is the set the loose rule would add on top of strict
containment. That gap is the reason strict containment is the rule under test.

## Ground truth

Eligibility does not read ground truth. After the view is fixed, evaluation uses
the existing strict exact-span scorer (`score_batch`): type, start, and end must
all match.

Overlap counters are not a partition. One removed prediction can be an exact
PERSON match and also overlap an EMAIL span. An EMAIL overlap is not described as
a PERSON annotation error.

Table columns:

| Column | Meaning |
| --- | --- |
| Suppressed PERSON | PERSON predictions removed from the thresholded view |
| Suppressed TP | removed predictions that exactly match PERSON ground truth |
| Suppressed FP | removed predictions that overlap no ground truth |
| Suppressed other/overlap | removed predictions that are not exact PERSON matches but overlap some ground truth |
| Net TP, Net FP, Net F1 | PERSON rows from the strict scorer, suppression minus baseline |

`strict_scorer_fp` is Suppressed PERSON minus Suppressed TP. It includes the
other/overlap column. Those rows are false positives in the scorer because the
type and offsets do not match a PERSON entity. They are not evidence that the
span was a false person mention.

## Scoring views

Thresholds stay at `0.3` and `0.7`. A stored score is kept when `score > threshold`,
the same comparison the decoder uses.

- **Full strict.** Every normalized entity after the existing DATE/TIME adapter,
  including `UNMAPPED`.
- **Gateway-supported-only.** Truth and predictions limited to Phase 1 types:
  PERSON, EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD, IP_ADDRESS, DATE_TIME,
  LOCATION.

PERSON metrics are taken from the full-strict state. The supported-only state
must report the same PERSON counts; the runner rejects the condition if it does
not.

## Reconstruction

Normalized files were not in the tree. They were rebuilt with the existing
`scripts.ood_pii.normalize` adapter and the same JSONL encoding as
`scripts/prepare_ood_pii.py`. `prepare_ood_pii.py` aborts on the first adapter
error, so it cannot emit the published AI4Privacy file by itself. The
reconstruction keeps that adapter's fail-closed behavior and drops a whole
example only when the adapter raises. Source `example_index` values are kept.
That reproduces the published row counts and SHA-256 digests:

| Dataset | Rows | SHA-256 |
| --- | ---: | --- |
| AI4Privacy validation | 47,725 (47,728 source, 3 skipped) | `575c7105a3ad5d27cbea7a467bebc360b08edfcdebadd71a45d5962a12472fca` |
| Gretel test | 5,000 | `2b628ffc09effc56475300038c648252a5faf9eb4f8877c9221ef31e5ca727f9` |
| Argilla train | 2,096 | `551e98472d8c0df3e5ec1590f0b62960d3be667266eadf65518715fd534ee6cf` |

Command, from the git root:

```bash
python .local/work/person-email-suppression/reconstruct_ood.py
```

Outputs live under `.local/work/person-email-suppression/`, which is gitignored.
They are not committed.

The canonical ontology hash of `configs/pii_ontology.yaml` on this commit is
`80c75a020b8b73d4b254f12cb40622f59224298799eea6d4472e81a144d45495`. The ontology
file was not modified.

## Evaluation command

Predictions must already exist. The command refuses a missing file and does not
call GLiNER. Reports are aggregate JSON outside the package tree.

```bash
python scripts/gliner_person_email_suppression.py \
  --dataset ai4privacy \
  --normalized /path/to/ai4privacy_validation.jsonl \
  --predictions /path/to/gliner_ood_t0.3.jsonl \
  --threshold 0.3 \
  --report /path/to/ai4privacy_t0.3.json
```

Repeat for Gretel and Argilla at `0.3` and `0.7`. Prediction records must use
normalized line order (`example_index` 0..N-1), which is what
`gliner_ood_evaluate.py` writes. The AI4Privacy JSONL `example_index` field still
carries the source index and can contain gaps.

## This workspace

Persisted OOD GLiNER prediction files are not on this machine. The Nemotron file
`gliner_width_24.jsonl` is a different dataset and was not scored. Inference was
not run. The six conditions therefore have no measured PERSON, full-strict, or
gateway-supported numbers in this run.

The rule is not a production change. It is not adopted from this run, because the
benchmark evidence required to keep or reject it was not available here.
