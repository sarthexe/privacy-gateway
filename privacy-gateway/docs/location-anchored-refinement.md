# Offline LOCATION anchored boundary refinement

An isolated persisted-prediction experiment, not a production detector.
DATE_TIME is closed; phone/card/email implementations and all frozen evidence
remain unchanged. No inference, regenerated predictions, gazetteer, external
knowledge, document-wide extraction, hierarchy inference, cross-type suppression,
ontology/scorer/threshold/max_width changes, or GT-selected boundaries.

## Evidence and fixed source-only policy

AI4Privacy baseline LOCATION TP/FP/FN at thresholds .3/.7:
31076/16236/3948 and 29528/12802/5496. The frozen diagnostic has
2266/1975 prediction-inside-GT pairs, versus 170/144 GT-inside-prediction and
8/5 partial pairs. Pair counts are nonexclusive and are not repairable-anchor counts.

Inspect only the existing model span plus 32 characters on each side, with a
256-character anchor limit. This window permits short connected token fragments
and immediate delimiter closure, not unbounded prose or new whitespace-separated
words. Reject longer anchors and completion touching an artificial window edge.

1. Only native labels normalized by the existing ontology to LOCATION qualify.
2. Require name-like alphabetic/combining-mark content already in the anchor.
3. Remove peripheral whitespace, outer parentheses/brackets/braces/quotes,
   unmatched terminal closers, and terminal comma/semicolon/colon/!/?.
   Preserve internal commas, owned delimiters and existing multi-token structure.
4. Reject serialized bracket/object collections with commas, colons, semicolons
   or string quoting. Such syntax is not an unambiguous wrapper around one name.
5. Reject terminal periods: punctuation versus name abbreviation is ambiguous.
6. Complete only a contiguous lexical token made of letters/Unicode marks and
   ASCII/Unicode hyphens (-/‐/‑) or apostrophes ('/’/ʼ) directly between letters.
   Require at least two alphabetic characters of existing support for completion.
   No case/capitalisation signal, new whitespace-separated word or geographic
   semantic inference is used. Existing multi-token anchors may have their first
   or last token completed without adding another token.
7. Reject newly completed apostrophe-s possessives, numeric/underscore/slash/
   en-dash/em-dash attachments, unsupported or incomplete qualifier expansion.
   Preserve possessives already completely inside the original model span.
8. If the model already includes an internal opening parenthesis and qualifier
   content, append only immediately adjacent missing closing parentheses.
   Do not add qualifier words across whitespace.
9. Reject proposed external parenthetical/comma qualifiers rather than absorbing
   nearby names, adjectives, PERSON, ORGANIZATION or hierarchy.

The policy is fixed before evaluation. decide(text, anchor) accepts no GT,
dataset, language, gazetteer or other prediction-label inputs.

## One-to-one scoring view

Keep the original model tuple unchanged in memory. A separate view retains its
order, multiplicity, labels and scores; each LOCATION slot has zero or one
replacement. Rejected/unchanged and all non-LOCATION slots retain the original
objects. No append, silent deletion or deduplication, including collisions.
The index-to-decision trace is in memory only and is never exported.

Rejections are a subset of unchanged predictions. Original count equals scoring
view count and equals replacements plus unchanged. Changed equals replacements.

## Diagnostic definitions

GT enters only after decisions. Score both full-strict and supported-only views
with the frozen strict scorer. Report LOCATION P/R/F1 and exact count deltas.

- Exclusive relations: exact, prediction inside GT, GT inside prediction,
  partial, no overlap.
- Reproduce the frozen nonexclusive boundary-pair diagnostic: all nonexact
  same-type overlapping pairs for which the GT or prediction is unmatched,
  retaining multiplicity. Require baseline pair counts to match frozen evidence.
- The truncation cohort comprises original anchors participating in a qualifying
  prediction-inside-GT pair. Compare each replacement with those original longer
  GT spans after refinement; report converted-to-exact and remaining nonexact.
- Report changed cohort FP and previously exact→FP separately. Originally short
  nonexact predictions are already strict FP, not newly introduced FP.
- Multiset capacity distinguishes gross recovered/lost TP from boundary-exact
  duplicates. Net ΔTP = recovered minus lost; fixed count implies ΔFP = ΔFN = −ΔTP.

Exports contain only aggregates, never names, snippets, offsets or example IDs.

## Commands

From privacy-gateway/:

```sh
python -m pytest tests/test_gliner_location_anchored_refinement.py \
  tests/test_gliner_datetime_anchored_refinement.py \
  tests/test_gliner_datetime_boundary_refinement.py \
  tests/test_gliner_phone_hybrid_experiment.py \
  tests/test_gliner_phone_boundary_refinement.py \
  tests/test_gliner_credit_card_experiment.py \
  tests/test_gliner_email_hybrid.py tests/test_analyze_gliner_ood_errors.py -q --tb=short
python -m ruff check scripts/gliner_location_anchored_refinement.py \
  tests/test_gliner_location_anchored_refinement.py
python -m py_compile scripts/gliner_location_anchored_refinement.py \
  tests/test_gliner_location_anchored_refinement.py
```

Evaluate ai4privacy/validation, gretel/test, argilla/train:

```sh
for pair in "ai4privacy validation" "gretel test" "argilla train"; do
  set -- $pair
  dataset=$1
  split=$2
  python scripts/gliner_location_anchored_refinement.py \
    --dataset "$dataset" \
    --normalized "/tmp/ood-reconstruction/$dataset/${dataset}_${split}.jsonl" \
    --predictions "/tmp/ood-reconstruction/predictions/$dataset/gliner_ood_t0.3.jsonl" \
    --predictions "/tmp/ood-reconstruction/predictions/$dataset/gliner_ood_t0.7.jsonl" \
    --baseline-report ../.local/outputs/phase15-ood-root-cause.json \
    --input-reference ../.local/outputs/phone-boundary-refinement-experiment.json \
    --report "/tmp/location-results/$dataset.json"
done
```

Fresh external output paths only; never overwrite inputs. Byte-identical
restoration from cached pinned data and decompression of existing prediction
archives is not inference or prediction regeneration. Keep adapters unchanged.
Leave all new work uncommitted. If recovery is tiny or dataset-specific, stop
LOCATION experimentation rather than adding another heuristic.

## Completed six-condition evaluation

**Disposition: STOP LOCATION experimentation.** Retain this as offline negative/
limited evidence only, pending review. Do not promote it into production or add
another LOCATION heuristic.

293 tests passed in 1.54 seconds (55 new LOCATION tests plus frozen regressions).
Ruff and py_compile passed. Evaluation elapsed times, including both thresholds:
AI4Privacy 191.93 seconds, Gretel 19.63 seconds, Argilla 6.96 seconds. These were
concurrent processes, not isolated timing benchmarks.

### LOCATION metrics

All P/R/F1 values and deltas below use the 0–1 scale, not percentage points.
Counts are exact; displayed floating-point values are rounded to nine decimals.

| Dataset / threshold | Baseline TP/FP/FN | B TP/FP/FN | ΔTP/FP/FN |
|---|---|---|---|
| AI4Privacy .3 | 31076/16236/3948 | 31077/16235/3947 | +1/−1/−1 |
| AI4Privacy .7 | 29528/12802/5496 | 29529/12801/5495 | +1/−1/−1 |
| Gretel .3 | 203/1640/23 | 203/1640/23 | 0/0/0 |
| Gretel .7 | 188/1444/38 | 188/1444/38 | 0/0/0 |
| Argilla .3 | 234/342/30 | 234/342/30 | 0/0/0 |
| Argilla .7 | 221/272/43 | 221/272/43 | 0/0/0 |

| Dataset / threshold | Baseline P/R/F1 | B P/R/F1 | ΔP/R/F1 |
|---|---|---|---|
| AI4Privacy .3 | .656831248/.887277296/.754858142 | .656852384/.887305847/.754882433 | +.000021136/+.000028552/+.000024291 |
| AI4Privacy .7 | .697566738/.843079032/.763451147 | .697590361/.843107583/.763477002 | +.000023624/+.000028552/+.000025855 |
| Gretel .3 | .110146500/.898230088/.196230063 | same | 0/0/0 |
| Gretel .7 | .115196078/.831858407/.202368138 | same | 0/0/0 |
| Argilla .3 | .406250000/.886363636/.557142857 | same | 0/0/0 |
| Argilla .7 | .448275862/.837121212/.583883752 | same | 0/0/0 |

### Replacement accounting

Rejected is a subset of unchanged; ambiguous is a subset of rejected.
Zero additions, deletions and non-LOCATION changes in all six conditions.

| Dataset / threshold | Original LOCATION | Replacements/changed | Unchanged LOCATION | Rejected | Ambiguous |
|---|---:|---:|---:|---:|---:|
| AI4Privacy .3 | 47312 | 57 | 47255 | 12430 | 9671 |
| AI4Privacy .7 | 42330 | 34 | 42296 | 10860 | 8568 |
| Gretel .3 | 1843 | 2 | 1841 | 731 | 664 |
| Gretel .7 | 1632 | 2 | 1630 | 659 | 598 |
| Argilla .3 | 576 | 0 | 576 | 196 | 112 |
| Argilla .7 | 493 | 0 | 493 | 184 | 102 |

At each AI4Privacy threshold there are 3 nonexact→exact transitions and gross
FN recoveries, but 2 exact→nonexact transitions and lost TP/new FP. Net recovery
is only +1 TP. Nonexact→nonexact changed predictions: 52/29. Gretel's two changed
predictions per threshold remain nonexact. All other exact-transition counts
are zero.

### Exclusive prediction relations before → after

These count each prediction once, prioritizing exact relations. Exact boundary
relations need not equal strict TP when duplicate predictions exhaust GT capacity.

| Dataset / threshold | Exact | Prediction inside GT | GT inside prediction | Partial | No overlap |
|---|---|---|---|---|---|
| AI4Privacy .3 | 31076→31077 | 2207→2209 | 160→158 | 5→4 | 13864→13864 |
| AI4Privacy .7 | 29528→29529 | 1923→1924 | 134→132 | 2→2 | 10743→10743 |
| Gretel .3 | 203→203 | 1→1 | 0→0 | 1→1 | 1638→1638 |
| Gretel .7 | 188→188 | 0→0 | 0→0 | 1→1 | 1443→1443 |
| Argilla .3 | 234→234 | 1→1 | 1→1 | 0→0 | 340→340 |
| Argilla .7 | 221→221 | 0→0 | 0→0 | 0→0 | 272→272 |

### Critical truncation diagnostic

Nonexclusive pairs reproduce the original root-cause counts exactly.
The original cohort is defined before replacements; it is not the after-view's
newly short predictions. Anchors can participate in more than one pair, or be
exact to another GT span, so pair/cohort/exclusive counts differ.

| Dataset / threshold | GT contains prediction pairs before→after | GT inside prediction pairs before→after | Original cohort anchors | Converted exact | Remaining nonexact | Cohort FN recovered | Cohort newly FP |
|---|---|---|---:|---:|---:|---:|---:|
| AI4Privacy .3 | 2266→2268 | 170→168 | 2219 | 0 | 2219 | 0 | 0 |
| AI4Privacy .7 | 1975→1976 | 144→142 | 1932 | 0 | 1932 | 0 | 0 |
| Gretel .3 | 1→1 | 0→0 | 1 | 0 | 1 | 0 | 0 |
| Gretel .7 | 0→0 | 0→0 | 0 | 0 | 0 | 0 | 0 |
| Argilla .3 | 1→1 | 1→1 | 1 | 0 | 1 | 0 | 0 |
| Argilla .7 | 0→0 | 0→0 | 0 | 0 | 0 | 0 | 0 |

No original truncation-cohort anchor changed. The observed 2266/1975 target
truncation pairs were not repaired. Newly nonexact predictions outside this
original cohort account for the worsening after-view containment counts.
Nonexclusive partial pairs: AI4Privacy 8→7 / 5→5, Gretel 1→1 / 1→1,
Argilla 0→0 / 0→0.

### Overall full-strict P/R/F1

| Dataset / threshold | Baseline P/R/F1 | B P/R/F1 | ΔP/R/F1 |
|---|---|---|---|
| AI4Privacy .3 | .498400153/.415369382/.453112422 | .498403913/.415372515/.453115840 | +.000003760/+.000003134/+.000003418 |
| AI4Privacy .7 | .548429082/.397436138/.460880751 | .548433406/.397439272/.460884385 | +.000004324/+.000003134/+.000003634 |
| Gretel .3 | .640900893/.714800262/.675836448 | same | 0/0/0 |
| Gretel .7 | .664664531/.698802507/.681306152 | same | 0/0/0 |
| Argilla .3 | .621366107/.396650573/.484206649 | same | 0/0/0 |
| Argilla .7 | .670218295/.389106820/.492363498 | same | 0/0/0 |

### Overall gateway-supported-only strict P/R/F1

| Dataset / threshold | Baseline P/R/F1 | B P/R/F1 | ΔP/R/F1 |
|---|---|---|---|
| AI4Privacy .3 | .588772078/.863699309/.700216254 | .588777237/.863706876/.700222389 | +.000005158/+.000007567/+.000006135 |
| AI4Privacy .7 | .627473220/.828897684/.714256371 | .627478948/.828905251/.714262892 | +.000005728/+.000007567/+.000006521 |
| Gretel .3 | .600732601/.950193129/.736091972 | same | 0/0/0 |
| Gretel .7 | .622583036/.939316934/.748835136 | same | 0/0/0 |
| Argilla .3 | .668520235/.930352537/.777997483 | same | 0/0/0 |
| Argilla .7 | .711237079/.917024936/.801126761 | same | 0/0/0 |

### Integrity and interpretation

All six LOCATION and both overall baseline views equal the frozen metrics
exactly, including floating-point P/R/F1 values. All frozen boundary-pair
counts match. Runtime checks preserved original tuples, labels, scores, order,
multiplicity and non-LOCATION predictions/metrics. Input hashes remained
unchanged before/after scoring.

Restored three normalized files and decompressed six existing prediction files
with all nine byte-level SHA256 identities matching the frozen reference.
AI4Privacy retained the original three whole-example exclusions; no GT repair.
Audited 264 preexisting tracked files: no changes except intentional scope-memory
updates. Production, GLiNER, scorer, ontology, persisted archives and all frozen
experiments remain unchanged. No inference, regenerated predictions, dependency
installation or automatic commit.

Aggregate evidence is in
.local/outputs/location-anchored-refinement-experiment.json, with exact expanded
evaluation commands, unrounded metrics, input identities, evidence and audits.
.local/outputs/location-preimplementation.json retains the pre-edit fingerprint
snapshot and aggregate source-structure analysis. Neither contains source
snippets, entity values, offsets or example IDs.

**Observations:** Tiny AI4Privacy-only net gain; no target truncation recovery;
two lost TP per AI4Privacy threshold. Wider containment counts did not improve.

**Hypothesis, not proven:** Much of the observed undercoverage requires adding
words beyond an unambiguous lexical attachment. These results do not establish
that all shorter spans are semantically wrong or that no other source-only
strategy could ever work. They do establish that this fixed experiment does not
justify another heuristic under the user's stopping rule.

Keep as frozen offline evidence only after review; not a production improvement.
Leave all implementation changes uncommitted.