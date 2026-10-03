# Offline PHONE_NUMBER Variant B: parser-identity boundary refinement

Variant A is committed and frozen. B imports its candidate generator, validation,
region policy, deduplication and append-only merger unchanged. It only selects
boundaries for candidates already validated by A. No model is loaded, no
prediction artifact is written, and no production or scorer code is changed.

## Rules fixed before evaluation

1. Parse each A-validated source slice with the identical explicit-region policy.
   The initial runs have no default region, so international candidates require
   a leading `+`. National numbers require an explicitly configured region.
2. Run `PhoneNumberMatcher` with `Leniency.VALID` and `max_tries=100` on that
   slice, not the surrounding source text. Boundaries cannot expand outside A.
3. A matcher-provided subspan is eligible only if its complete parsed identity
   equals the original: country code, national number, extension, Italian
   leading-zero flag, and number of leading zeros.
4. All discarded edge characters must be ASCII punctuation or horizontal
   whitespace. `+` is explicitly excluded. No digit or letter can be discarded.
5. Choose the unique shortest eligible matcher span. Reject absent or ambiguous
   matches with aggregate reasons. Unchanged matches remain unchanged.
6. Translate the selected local offsets to original absolute character offsets,
   deduplicate exact spans, then use A's unchanged append-only merger.

The rules are generic and contain no GT, dataset, language, model-label or
assignment/reachability decisions. GT is passed only to scoring and diagnostics.
No GT-specific offset rule is permitted. In particular, a valid extension is
retained, not removed to match annotations. Numeric suffixes cannot be shortened
into a different number. Candidate-inside-GT cases are not expanded heuristically.
No regex splitting or arbitrary search over digit substrings is performed.

## Motivation observed before implementation

AI4Privacy has 18 containing-GT and 9 inside-GT validated candidates. Of its
added candidates, 18 containing-GT and 4 inside-GT cases become boundary FP at
each threshold. Fifteen containing-GT cases have unmatched closing parentheses
with punctuation-only excess. Three contain extra digits relative to GT, and
their parser matches preserve the full span. Gretel's two boundary FP have
unmatched closing parentheses with punctuation-only excess. Argilla has no
validated boundary mismatches.

These are diagnostic observations, not inference-time rules. Parser-based
punctuation refinement is defensible; removing semantic digits to match GT is not.

## Accounting and invariants

Baseline, A and B are scored independently using the unchanged strict scorer for
full and seven-type supported-only views. PHONE_NUMBER metrics are derived from
the same strict states. Every baseline and A metric and A candidate count must
reproduce the frozen saved report; parser version and region must also match.

Recovered FN are identified by exact span multiset identity relative to
baseline matches. A recoveries retained by B are the multiset intersection,
not just a comparison of aggregate TP. A-added FP remaining/disappeared and
new B FP are likewise tracked by exact identity after subtracting unchanged
baseline FP. Reports distinguish surviving A FP from all B-added FP.
Retention is undefined (`null`) when A recovered no FN.

Candidate transition counts classify every A-validated candidate, including
rejections. B's final boundary table counts deduplicated selected spans.
Categories are exclusive and use A's existing boundary priority. Candidate
totals are shared by thresholds; added predictions differ after duplicate checks.

Every existing native GLiNER prediction remains unchanged, including non-phone
and unmapped labels, scores, offsets, order and multiplicity. Only added phone
candidates differ. Input fingerprints are checked before and after evaluation;
the persisted benchmark and frozen A files are independently audited.

Reports contain aggregates only: no text, phone values, raw offsets or example
IDs. CLI reports must be new paths outside the repository. No second-pass rule
tuning based on benchmark output is allowed in this experiment.

## Limitations

Number validity does not prove assignment or semantic correctness. Matcher
metadata/formatting constraints can reject an A-valid candidate; such losses
are measured rather than hidden with a fallback. The candidate subset cannot
recover A-rejected national numbers or expand inside-GT spans. Ground-truth
formatting conventions may retain punctuation that a generic parser excludes,
so a tighter boundary is not guaranteed to improve strict scoring.

## Example invocation

From `privacy-gateway/`, without `--region`:

```sh
python scripts/gliner_phone_boundary_refinement.py \
  --normalized /tmp/ood-reconstruction/ai4privacy/ai4privacy_validation.jsonl \
  --predictions /tmp/ood-reconstruction/predictions/ai4privacy/gliner_ood_t0.3.jsonl \
  --predictions /tmp/ood-reconstruction/predictions/ai4privacy/gliner_ood_t0.7.jsonl \
  --variant-a-report ../.local/outputs/phone-hybrid-experiment.json \
  --report /tmp/phone-boundary-results/ai4privacy.json
```

Repeat with the corresponding Gretel and Argilla files. Temporary reconstructed
inputs must be hash-verified against the frozen report when recreated.

## Initial measured results

All six baseline and Variant A conditions reproduce the frozen report.
All A recoveries are retained: AI4Privacy 254/254 and 406/406; Gretel 8/8 and
21/21. Argilla has no A recoveries and remains unchanged.

| Dataset | Threshold | B vs A TP | B vs A FP | B vs A FN | A-added FP disappeared | New B FP identity |
|---|---:|---:|---:|---:|---:|---:|
| AI4Privacy | 0.3 | +2 | -15 | -2 | 16 | 1 |
| AI4Privacy | 0.7 | +4 | -16 | -4 | 17 | 1 |
| Gretel | 0.3 | 0 | -2 | 0 | 2 | 0 |
| Gretel | 0.7 | 0 | -2 | 0 | 2 | 0 |
| Argilla | 0.3 | 0 | 0 | 0 | 0 | 0 |
| Argilla | 0.7 | 0 | 0 | 0 | 0 | 0 |

AI4Privacy: 1,706 A-validated candidates yield 1,705 selected B spans. Sixteen
spans are refined, one no-phone-GT-overlap span is rejected, and 1,689 remain
unchanged. Exact boundaries increase 1,547 to 1,562; GT-inside-candidate cases
fall 18 to 3. Inside-GT cases remain 9. Added boundary-mismatch FP fall 22 to 7
at both thresholds. Gretel's two boundary mismatches disappear.

The new FP identity is the tighter punctuation boundary of an existing A-added
no-phone-GT-overlap FP at both thresholds. It replaces that old FP identity;
it is still counted as a new strict FP, not hidden as “zero new FP.” The separate
rejected no-overlap candidate was already an exact model phone prediction at
0.3, so its original model FP remains unchanged; at 0.7, rejecting it removes
an A-added FP. No model prediction is deleted in either condition.

Boundary refinement helped without losing A recoveries or using suppression.
This supports reviewing/freezing B as the bounded offline phone candidate,
then moving to a separately scoped detector experiment rather than adding
more phone heuristics. It does not justify production promotion.

Aggregate results, complete strict metrics/deltas, commands and provenance:
`.local/outputs/phone-boundary-refinement-experiment.json` and
`.local/outputs/phone-boundary-refinement-experiment.html` at the repository root.
Variant B remains uncommitted for review.