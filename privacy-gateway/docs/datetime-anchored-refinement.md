# Final offline anchored DATE_TIME experiment (Variant B)

The previous mixed-result append-only experiment is frozen. Reuse its structural
parser and strict scoring helpers without modifying them. No inference,
production changes, threshold/max_width changes, ontology/scorer changes,
cross-type suppression or broad new source detector rules.

## Eligibility and exact rules

1. Use the existing ontology adapter to select native model labels whose
   evaluation type is DATE_TIME. All other predictions remain identical objects.
2. Inspect only the anchor plus 32 source characters on each side. Reject
   anchors longer than 256 characters. Reject a core touching an artificial
   neighborhood edge. Do not scan the rest of the document for candidates.
3. Reuse the frozen calendar/clock/AM-PM/timezone grammar, Gregorian validation,
   ambiguity checks, and maximal complete structural cores.
4. Require exactly one valid core overlapping the anchor. Multiple overlapping
   cores, no valid core, malformed/ambiguous input or unsupported identity remain
   unchanged and are counted as rejected refinements.
5. A calendar component must already be entirely within the model anchor.
   A clock must already contain its source hour:minute component, or a complete
   hour-plus-AM/PM value. Do not add a calendar component to a time-only anchor,
   infer a year/month/date, or complete missing hour/minute digits.
6. Remove only peripheral whitespace and `()[]{}"'“”‘’.,;:!?` outside the core.
   Never remove digits, semantic words, AM/PM or valid timezone information.
7. A parsed clock may retain/complete explicit adjacent seconds, fractions,
   AM/PM or timezone suffixes. No semantic leftward expansion or prose extension.
   Source information is preserved even if the GT annotation excludes it.
8. Structural punctuation is handled by selecting source boundaries only.
   Never rewrite text, substitute separators or manufacture a normalized value.

No GT, dataset name, language or geographic identity enters refinement.
Rules are fixed before evaluation, not tuned against outcomes.

## Replacement semantics and trace

Keep the original prediction tuple separately in memory. Build a temporary
scoring view with exactly one slot per original prediction in the same order:
the original object, or one refined prediction retaining the original native
label and score. Never deduplicate or silently delete collisions. At most one
replacement per DATE_TIME anchor; no source-only additions.

An in-memory index-to-decision trace binds each replacement to exactly one
original. These indices, source offsets and values are never exported.
Rejections are a subset of unchanged predictions. Aggregate accounting:

- original count equals refined-view count;
- replacements + unchanged equals original count;
- changed equals replacements; added/deleted/non-DATE_TIME changes equal zero.

## Diagnostics

GT is used only after source/model decisions finish. Report exclusive relations
for all original/anchored DATE_TIME predictions and both sides of replacements:
exact, inside GT, contains GT, partial, no DATE_TIME overlap.

Gross recovered/lost TP use multiset exact-match capacity. Net TP improvement
is recovered minus lost; fixed prediction count implies ΔFP = ΔFN = -ΔTP.
Exact boundary relation counts can include duplicate predictions, so they are
not interchangeable with allocated strict TP.

Separate changed nonexact→exact, nonexact→nonexact, exact→nonexact,
introduced/resolved inside-GT cases, and previous appended versus current
changed inside-GT counts. These affected populations have different denominators:
also show all baseline/anchored inside-GT counts rather than claiming that
replacement counts alone prove a comparable FP rate.

## Reproduction

From privacy-gateway/, run the new suite plus frozen DATE_TIME, phone,
credit-card, email and analyzer suites; run Ruff and py_compile on the two new
Python files. Synthetic fixtures only.

For each dataset/split pair ai4privacy/validation, gretel/test, argilla/train:

```sh
python scripts/gliner_datetime_anchored_refinement.py \
  --dataset argilla \
  --normalized /tmp/ood-reconstruction/argilla/argilla_train.jsonl \
  --predictions /tmp/ood-reconstruction/predictions/argilla/gliner_ood_t0.3.jsonl \
  --predictions /tmp/ood-reconstruction/predictions/argilla/gliner_ood_t0.7.jsonl \
  --previous-report ../.local/outputs/datetime-boundary-refinement.json \
  --input-reference ../.local/outputs/phone-boundary-refinement-experiment.json \
  --report /tmp/datetime-anchored-results/argilla.json
```

Use fresh external report paths; overwriting is forbidden. Retain aggregate
reports promptly in .local/outputs. Compare persisted baseline, frozen previous
metrics and B on all six conditions; previous metrics are reused, not regenerated.
Audit all frozen tracked file bytes and input/prediction hashes. Leave B
uncommitted. This is the final DATE_TIME trial: if it does not preserve useful
recoveries while reducing new subspan FP, stop and implement no further heuristic.

## Measured six-condition result

Both thresholds were evaluated together for each dataset: AI4Privacy 191.656
seconds (47,725 examples), Gretel 19.862 (5,000), Argilla 6.949 (2,096).
Timings are concurrent-run observations, not isolated performance guarantees.

| Dataset | Threshold | Baseline TP/FP/FN | Previous TP/FP/FN | B TP/FP/FN | B ΔTP/FP/FN vs baseline | F1 baseline → previous → B |
|---|---:|---|---|---|---|---|
| AI4Privacy | 0.3 | 31148/26003/3602 | 31937/26243/2813 | 31192/25959/3558 | +44/-44/-44 | .677860 → .687335 → .678817 |
| AI4Privacy | 0.7 | 30434/21258/4316 | 31349/21527/3401 | 30468/21224/4282 | +34/-34/-34 | .704148 → .715518 → .704935 |
| Gretel | 0.3 | 3665/1425/140 | 3724/1545/81 | 3665/1425/140 | 0/0/0 | .824058 → .820807 → .824058 |
| Gretel | 0.7 | 3649/1157/156 | 3719/1286/86 | 3649/1157/156 | 0/0/0 | .847521 → .844268 → .847521 |
| Argilla | 0.3 | 451/70/22 | 451/72/22 | 451/70/22 | 0/0/0 | .907445 → .905622 → .907445 |
| Argilla | 0.7 | 451/33/22 | 451/35/22 | 451/33/22 | 0/0/0 | .942529 → .940563 → .942529 |

B ΔF1 versus baseline: +0.000957552 / +0.000786655 for AI4Privacy,
zero for the other four conditions. Full precision, recall, exact floating-point
deltas versus both comparators and both overall strict views are in the exports.

| Dataset | Threshold | Original = scoring-view count | Replaced = changed | Unchanged total | Rejected (subset of unchanged) | DATE_TIME anchors unchanged |
|---|---:|---:|---:|---:|---:|---:|
| AI4Privacy | 0.3 | 384481 | 96 | 384385 | 28165 | 57055 |
| AI4Privacy | 0.7 | 295013 | 81 | 294932 | 22876 | 51611 |
| Gretel | 0.3 | 28798 | 0 | 28798 | 872 | 5090 |
| Gretel | 0.7 | 25605 | 0 | 25605 | 608 | 4806 |
| Argilla | 0.3 | 6944 | 0 | 6944 | 251 | 521 |
| Argilla | 0.7 | 5831 | 0 | 5831 | 214 | 484 |

Added/deleted predictions and non-DATE_TIME changes are zero in all six cases.
All originals survive separately; only the temporary scoring view differs.

| Dataset | Threshold | Previous new inside-GT FP | B changed inside-GT FP | All baseline → B inside-GT | Changed exact | Changed FP | Gross recovered FN | Lost TP |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| AI4Privacy | 0.3 | 21 | 0 | 1396 → 1392 | 66 | 30 | 66 | 22 |
| AI4Privacy | 0.7 | 22 | 0 | 1162 → 1159 | 55 | 26 | 55 | 21 |
| Gretel | 0.3 | 98 | 0 | 221 → 221 | 0 | 0 | 0 | 0 |
| Gretel | 0.7 | 105 | 0 | 174 → 174 | 0 | 0 | 0 | 0 |
| Argilla | 0.3 | 0 | 0 | 0 → 0 | 0 | 0 | 0 | 0 |
| Argilla | 0.7 | 0 | 0 | 0 → 0 | 0 | 0 | 0 | 0 |

AI4Privacy changed-core relation counts (threshold 0.3/0.7): exact 66/55,
inside-GT 0/0, contains-GT 22/21, partial 0/0, no-overlap 8/5.
Nonexact→nonexact changes: 8/5. Exact→nonexact changes: 22/21.
Thus changed FP totals 30/26 are not 30/26 newly introduced FP: 22/21
were previously exact and 8/5 were already nonexact. All other datasets have
zero changed predictions and zero changed-core relation counts.

## Interpretation and limits

**Factual answer:** Anchoring retains a small subset of useful AI4Privacy
boundary recoveries and introduces no new inside-GT cores. It does not retain
the prior Gretel recoveries. AI4Privacy's gross 66/55 recoveries are only 8.4%/6.0%
of the previous 789/915; lost TP reduce the net improvement to 44/34.
Gretel and Argilla return to baseline by making no replacements, not through
successful boundary repair.

The old parser's calendar/clock/composite/timezone rules jointly yielded the
previous recoveries and FP. That report lacks a per-rule causal ablation.
Its peripheral-envelope exact-core/raw-nonexact counts (AI4Privacy 213/257,
Gretel 32/41, Argilla 0/0) demonstrate conformance differences, not that
punctuation trimming caused the improvement. Previous inside-GT cores are
subspan outcomes, not evidence to alter parsing rules.

GT-inside-candidate changed FP may reflect structural-completeness versus
annotation-boundary disagreement. Never trim legitimate source information to
avoid this loss. The fixed grammar is not broad multilingual/relative-date NLP;
unsupported/incomplete anchors and bounded neighborhoods limit recovery.
Strict FP do not by themselves prove semantic non-date content.

Retain B as a limited offline result, not a production promotion. This final
DATE_TIME trial is closed; no further heuristic or post-evaluation tuning was
implemented. Work remains uncommitted for review.

**Verification:** 238 tests passed in 1.37 seconds (40 new, 198 frozen), Ruff
and py_compile passed. All six baselines reproduce exactly. All nine input/
prediction hashes match frozen evidence and remain unchanged. All 259
preexisting tracked non-memory files remain byte-identical; only the project
memory scope note changed among the 260 preexisting tracked files.

Workspace-root exports:

- `.local/outputs/datetime-anchored-refinement.json`
- `.local/outputs/datetime-anchored-refinement.html`
- `.local/outputs/datetime-anchored-preimplementation.json`
- `.local/outputs/datetime-anchored-audit.json`

The aggregate exports include exact executed evaluation/validation commands.
CLI results were first written to new external
`/tmp/datetime-anchored-results/{ai4privacy,gretel,argilla}.json` paths.