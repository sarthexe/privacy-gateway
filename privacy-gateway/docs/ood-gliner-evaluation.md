# GLiNER-PII OOD evaluation

This is a separate reproduction experiment, not a claim to reproduce NVIDIA's
published numbers. The model card reports strict F1 `0.70` for “Argilla PII” and
`0.64` for AI4Privacy at threshold `0.3`, but identifies only the Argilla
organization—not an exact Argilla dataset/revision—and does not identify exact
public dataset revisions. This repository evaluates the explicit
`argilla/textcat-tokencat-pii-per-domain` revision listed below.

The metric of record remains strict, order-independent multiset matching of
gateway entity type and half-open character offsets. Native-label metrics are
reported only for source labels with an unambiguous mapping into GLiNER's pinned
Nemotron prompt vocabulary. All other source labels remain `UNMAPPED`, are kept
in full-dataset ground-truth support, and are reported separately. They are
excluded from the gateway-supported-only evaluation described below.

## Pinned public sources

| Dataset | Repository revision | Annotation contract | License |
| --- | --- | --- | --- |
| AI4Privacy | `c8c77895a005822682b66ab547fc0422579bc1d3` | `source_text`, `privacy_mask` start/end/value/label | repository `license.md` |
| Gretel | `e06eb1499ca8d54470f085021cd8e54f9efac7fd` | Parquet `text`, `entities` value/types; offsets must be uniquely reconstructed | Apache-2.0 |
| Argilla | `683954a23877151a1d05e7515d5e6cf87203ffd2` | Hugging Face `source-text`, `pii.suggestion` start/end/label | not specified in card |

Each normalization writes external JSONL records with `example_index`, source
`text`, offset-only `entities`, offset-only `native_entities`, and per-record
dataset/revision/split/language metadata. Its sidecar has repository, revision,
split, row count, normalized SHA-256, normalization schema, and ontology hash.
No reports, logs, or prediction artifacts contain source text or entity values.

## Normalize (later)

```bash
python scripts/prepare_ood_pii.py --dataset ai4privacy --split validation --output-dir /path/to/ood/ai4privacy
python scripts/prepare_ood_pii.py --dataset gretel --split test --output-dir /path/to/ood/gretel
python scripts/prepare_ood_pii.py --dataset argilla --split train --output-dir /path/to/ood/argilla
```

The forthcoming GPU condition runner must use the pinned
`nvidia/gliner-PII` revision `bd23e8ef4425fd04e34c5204ab49ffaa706eae79`, fp32,
batch size 1, and `max_width=24`, separately at thresholds `0.3` and `0.7`.
Those are fixed evaluation conditions—not OOD tuning. The deterministic-email
hybrid is intentionally not invoked by this OOD normalization path.

## GPU evaluation (later)

```bash
python scripts/gliner_ood_evaluate.py --normalized /path/to/ood/ai4privacy/ai4privacy_validation.jsonl --output-dir /path/to/ood/ai4privacy/gliner --device cuda
```

This writes separate prediction artifacts and aggregate reports for the fixed
`0.3` and `0.7` conditions. It must not be used to select a threshold.

## Offline error analysis and evaluation views

Reuse a persisted prediction artifact, once per dataset/threshold pair:

```bash
python scripts/analyze_gliner_ood_errors.py \
  --normalized /path/to/ood/ai4privacy/ai4privacy_validation.jsonl \
  --predictions /path/to/ood/ai4privacy/gliner/gliner_ood_t0.3.jsonl \
  --output /path/to/ood/ai4privacy/gliner/errors_t0.3.json
```

This command never loads a model or runs inference. Inputs are read-only; the
report must be a new file outside the repository. Existing reports and prediction
artifacts are not overwritten. Dataset hash, pinned model identity, record order,
and record count are validated before writing any report.

Each aggregate-only JSON report adds three clearly separated sections:

- **`full_strict`**: strict performance over all normalized dataset annotations,
  including unsupported mapped types and `UNMAPPED`. This preserves the original
  full-dataset TP/FP/FN and uses the same predictions after the existing gateway
  mapping (which already removes UNMAPPED predictions). It contains `precision`,
  `recall`, `f1`, `tp`, `fp`, `fn`, `gt_entity_count`, and `prediction_count`.
- **`gateway_supported_only_strict`**: strict performance over gateway-supported
  annotations and predictions only, with the same metric/count fields. Supported
  evaluation types come from `SUPPORTED_TYPES`: PERSON, EMAIL_ADDRESS,
  PHONE_NUMBER, CREDIT_CARD, IP_ADDRESS, DATE_TIME, and LOCATION. DATE and TIME
  remain mapped to DATE_TIME by the existing adapter before filtering. Unsupported
  mapped types and UNMAPPED GT cannot contribute FN here. Unsupported prediction
  types are also excluded. A prediction of a supported type remains in the scoring
  population even when it overlaps excluded GT, and is an FP unless it exactly
  matches supported GT; this view does not mask spans or discard entire records.
- **`ontology_coverage`**: ground-truth coverage, independent of predictions:

  | Field | Definition |
  | --- | --- |
  | `gt_entity_count` | Total GT entities, including duplicates |
  | `gateway_supported_gt_count` | GT entities with a supported evaluation type after DATE/TIME mapping |
  | `gateway_supported_gt_pct` | `100 * gateway_supported_gt_count / gt_entity_count` (0–100) |
  | `unmapped_gt_count` | GT entities whose normalized gateway type is UNMAPPED |
  | `unsupported_but_mapped_gt_count` | GT entities with a non-UNMAPPED type outside SUPPORTED_TYPES |
  | `gt_by_adapter_category` | Counts from the existing adapter category function on the original normalized type, preserving DATE/TIME adapter categories |

The three GT populations are disjoint and sum to `gt_entity_count`. Coverage
percentages always use total GT as the denominator. With zero GT, the percentage
is `0.0`. Precision, recall, and F1 use the existing helper's zero-denominator
behavior (`0.0`); supported predictions still count as FP when supported GT is
empty.

Both views reuse `score_batch()` and strict, order-independent **multiset**
matching of exact evaluation type plus half-open `[start,end)` offsets. Filtering
changes only the supported-only population, not the shared scorer or annotations.
Prediction count means the number of scored entities, not prediction JSONL rows;
`record_count` remains the number of dataset records.

The existing `overall` TP/FP/FN object is unchanged for backward compatibility.
Existing per-type metrics, FN/FP failure groups, nonexclusive structural boundary
patterns, likely type-confusion pairs, `unmapped_ground_truth`, and removed
UNMAPPED prediction counts remain **full-dataset diagnostics**, not supported-only
diagnostics. They are not removed or relabeled. Reports never include source
text, entity values, snippets, offsets, or untrusted source metadata.

The full view measures strict performance on all annotations and is affected by
ontology coverage. The supported-only view measures strict performance within
the gateway's declared scope. Coverage describes how much of the dataset falls
inside that scope. These are separate results, not a threshold-selection rule.
