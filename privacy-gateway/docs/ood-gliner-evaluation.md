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
in gateway support, and are reported separately.

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
