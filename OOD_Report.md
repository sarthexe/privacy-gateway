# GLiNER-PII Out-of-Distribution Evaluation Report

## 1. Executive Summary

This evaluation measures how well the pinned `nvidia/gliner-PII` model generalizes beyond the Nemotron-PII distribution used for the initial benchmark.

The evaluation uses the same strict exact-span methodology throughout:

> A prediction is correct only when **entity type, start offset, and end offset all match the ground truth exactly**.

The detector was evaluated without the email hybrid or any other post-processing.

### Fixed evaluation configuration

- **Model:** `nvidia/gliner-PII`
- **Model revision:** `bd23e8ef4425fd04e34c5204ab49ffaa706eae79`
- **max_width:** `24`
- **max_len:** `384`
- **dtype:** FP32
- **batch size:** `1`
- **thresholds:** `0.3`, `0.7`
- **OOD threshold tuning:** none
- **Email hybrid:** not applied

### Overall gateway-ontology results

| Dataset | Split | Rows | F1 @ 0.3 | F1 @ 0.7 |
|---|---|---:|---:|---:|
| **AI4Privacy** | validation | 47,725 | 0.4565 | **0.4639** |
| **Gretel** | test | 5,000 | 0.6758 | **0.6813** |
| **Argilla** | train | 2,096 | 0.4838 | **0.4925** |

For comparison, the earlier Nemotron `max_width=24` benchmark achieved:

```text
F1 = 0.8221
```

The results therefore show a substantial distribution-dependent performance gap.

---

# 2. Objective

The initial Nemotron benchmark answered:

> How well does GLiNER-PII perform on the Nemotron distribution?

That alone is insufficient for a privacy gateway.

The OOD evaluation asks:

> Does the same frozen detector configuration generalize to PII examples originating from different datasets and annotation distributions?

This matters because production traffic may differ significantly from the distribution on which a detector was evaluated or trained.

The OOD datasets were therefore evaluated **without using their results to tune the detector**.

---

# 3. Evaluation Protocol

## 3.1 Model

The exact pinned checkpoint was:

```text
nvidia/gliner-PII
revision = bd23e8ef4425fd04e34c5204ab49ffaa706eae79
```

Effective configuration:

```text
max_width = 24
max_len = 384
FP32
batch_size = 1
```

The model was run on a Kaggle Tesla T4 with:

```text
CUDA = True
Torch = 2.10.0+cu128
```

---

# 4. Threshold Selection

Two thresholds were evaluated:

```text
0.3
0.7
```

The `0.7` operating point was previously selected using a fixed 2,000-record subset derived from the **Nemotron train split**.

The OOD datasets were **not** used to select the threshold.

This preserves the distinction between configuration/development data and external evaluation data.

Conceptually:

```text
Nemotron train-derived subset
        ↓
choose threshold
        ↓
freeze configuration
        ↓
OOD evaluation
```

The OOD results are therefore measurements, not tuning targets.

---

# 5. Dataset Preparation

## 5.1 AI4Privacy

**Repository**

```text
ai4privacy/pii-masking-300k
```

**Revision**

```text
c8c77895a005822682b66ab547fc0422579bc1d3
```

**Split**

```text
validation
```

### Dataset statistics

| Metric | Value |
|---|---:|
| Source rows | 47,728 |
| Retained rows | 47,725 |
| Skipped rows | 3 |
| Valid normalized entities | 319,128 |

**Normalized SHA-256**

```text
575c7105a3ad5d27cbea7a467bebc360b08edfcdebadd71a45d5962a12472fca
```

### Annotation handling

Some AI4Privacy records contained cases where:

```text
value != source_text[start:end]
```

while the explicit offsets were otherwise valid.

The adapter therefore treats:

```text
label + start + end
```

as authoritative and does not reject an otherwise valid span because of the inconsistent `value` field.

Three records contained genuinely invalid offsets outside the source text. These were excluded as complete examples rather than clipping or repairing individual spans.

Final benchmark:

```text
47,728 source rows
        ↓
47,725 valid rows
3 invalid rows excluded
```

---

## 5.2 Gretel

**Repository**

```text
gretelai/gretel-pii-masking-en-v1
```

**Revision**

```text
e06eb1499ca8d54470f085021cd8e54f9efac7fd
```

**Split**

```text
test
```

### Dataset statistics

| Metric | Value |
|---|---:|
| Rows | 5,000 |

**Normalized SHA-256**

```text
2b628ffc09effc56475300038c648252a5faf9eb4f8877c9221ef31e5ca727f9
```

The dataset required normalization because entity annotations were exposed in serialized form.

---

## 5.3 Argilla

**Repository**

```text
argilla/textcat-tokencat-pii-per-domain
```

**Revision**

```text
683954a23877151a1d05e7515d5e6cf87203ffd2
```

**Split**

```text
train
```

Important:

> Argilla's `train` split was **not used to train GLiNER**. It was treated as an external OOD evaluation set.

### Dataset statistics

| Metric | Value |
|---|---:|
| Rows | 2,096 |
| Entities | 6,628 |
| Mapped entities | 3,104 |
| UNMAPPED entities | 3,524 |
| Skipped rows | 0 |

**Normalized SHA-256**

```text
551e98472d8c0df3e5ec1590f0b62960d3be667266eadf65518715fd534ee6cf
```

---

# 6. Scoring Method

The benchmark uses strict exact matching.

A prediction is a true positive only when:

```text
entity type == ground truth type
AND
start offset == ground truth start
AND
end offset == ground truth end
```

Example:

```text
Ground truth: EMAIL [100,120)
Prediction:   EMAIL [100,120)
```

This is a true positive.

But:

```text
Ground truth: EMAIL [100,120)
Prediction:   EMAIL [100,119)
```

is not an exact match.

Likewise:

```text
Ground truth: EMAIL [100,120)
Prediction:   PERSON [100,120)
```

is incorrect because the entity type differs.

This strictness is intentional because the downstream gateway needs the **exact sensitive text span** to tokenize it correctly.

---

# 7. AI4Privacy Results

## 7.1 Overall results

### Threshold 0.3

```text
Precision = 0.5035
Recall    = 0.4175
F1        = 0.4565
TP        = 133,227
FP        = 131,397
FN        = 185,901
```

### Threshold 0.7

```text
Precision = 0.5530
Recall    = 0.3995
F1        = 0.4639
TP        = 127,507
FP        = 103,075
FN        = 191,621
```

Increasing the threshold:

```text
0.3 → 0.7
```

resulted in:

```text
Precision: 50.35% → 55.30%
Recall:    41.75% → 39.95%
F1:        0.4565 → 0.4639
```

---

## 7.2 AI4Privacy entity-level results

At threshold `0.7`:

| Entity | Precision | Recall | F1 |
|---|---:|---:|---:|
| ADDRESS | 0.2092 | 0.4795 | **0.2913** |
| DATE_TIME | 0.6098 | 0.8951 | **0.7254** |
| EMAIL_ADDRESS | 0.9141 | 0.8948 | **0.9043** |
| LOCATION | 0.6974 | 0.8431 | **0.7634** |
| PERSON | 0.6956 | 0.7915 | **0.7405** |
| PHONE_NUMBER | 0.8584 | 0.6771 | **0.7571** |
| USERNAME | 0.7243 | 0.6723 | **0.6973** |

### Main observation

`ADDRESS` is a significant weakness:

```text
Precision = 0.2092
Recall    = 0.4795
F1        = 0.2913
```

The detector finds a substantial portion of addresses, but many predicted address spans do not exactly match the ground-truth spans.

---

# 8. AI4Privacy Native Label-Space Results

AI4Privacy contains labels that do not currently map cleanly into the gateway ontology.

At threshold `0.7`:

```text
Precision = 0.4059
Recall    = 0.7403
F1        = 0.5243
TP        = 119,412
FP        = 174,791
FN        = 41,901
```

AI4Privacy contains:

```text
Total entities = 319,128
UNMAPPED       = 157,815
```

Therefore approximately half of its annotations are currently outside the comparable gateway label space.

This creates a distinction between:

```text
Gateway ontology F1 = 0.4639
Native label-space F1 = 0.5243
```

The lower gateway score therefore reflects both detector behavior and limitations of the current common ontology.

---

# 9. Gretel Results

## 9.1 Overall results

### Threshold 0.3

```text
Precision = 0.6409
Recall    = 0.7148
F1        = 0.6758
TP        = 15,281
FP        = 8,562
FN        = 6,097
```

### Threshold 0.7

```text
Precision = 0.6647
Recall    = 0.6988
F1        = 0.6813
TP        = 14,939
FP        = 7,537
FN        = 6,439
```

Threshold `0.7` again increased precision while slightly decreasing recall.

---

## 9.2 Gretel entity-level results

At threshold `0.7`:

| Entity | Precision | Recall | F1 |
|---|---:|---:|---:|
| ADDRESS | 0.2355 | 0.4286 | **0.3039** |
| BANK_ACCOUNT | 0.8386 | 0.8744 | **0.8561** |
| CREDIT_CARD | 0.9333 | 0.8235 | **0.8750** |
| DATE_TIME | 0.7593 | 0.9590 | **0.8475** |
| EMAIL_ADDRESS | 0.9782 | 0.9828 | **0.9805** |
| ID_CARD | 0.8437 | 0.6884 | **0.7582** |
| IP_ADDRESS | 0.9582 | 0.9532 | **0.9557** |
| LOCATION | 0.1152 | 0.8319 | **0.2024** |
| ORGANIZATION | 0.4420 | 0.4324 | **0.4372** |
| PERSON | 0.4319 | 0.9785 | **0.5993** |
| PHONE_NUMBER | 0.9605 | 0.8064 | **0.8767** |
| SOCIAL_IDENTIFIER | 0.9712 | 0.7263 | **0.8311** |
| USERNAME | 0.8391 | 0.9260 | **0.8805** |

### Main observation

Structured entities perform well:

```text
EMAIL          0.9805
IP_ADDRESS     0.9557
PHONE_NUMBER   0.8767
CREDIT_CARD    0.8750
BANK_ACCOUNT   0.8561
DATE_TIME      0.8475
USERNAME       0.8805
```

Contextual entities are substantially noisier:

```text
ADDRESS        0.3039
LOCATION       0.2024
ORGANIZATION   0.4372
PERSON         0.5993
```

The LOCATION result is especially notable:

```text
Precision = 0.1152
Recall    = 0.8319
```

The detector identifies many actual locations but generates a large number of incorrect location predictions.

---

# 10. Argilla Results

## 10.1 Overall results

### Threshold 0.3

```text
Precision = 0.6209
Recall    = 0.3963
F1        = 0.4838
TP        = 2,627
FP        = 1,604
FN        = 4,001
```

### Threshold 0.7

```text
Precision = 0.6707
Recall    = 0.3891
F1        = 0.4925
TP        = 2,579
FP        = 1,266
FN        = 4,049
```

Again, increasing the threshold increased precision while reducing recall.

---

## 10.2 Argilla entity-level results

At threshold `0.7`:

| Entity | Precision | Recall | F1 |
|---|---:|---:|---:|
| ADDRESS | 0.3258 | 0.3583 | **0.3413** |
| BANK_ACCOUNT | 0.5104 | 0.6721 | **0.5802** |
| CREDIT_CARD | 0.6184 | 0.5222 | **0.5663** |
| DATE_TIME | 0.9318 | 0.9535 | **0.9425** |
| EMAIL_ADDRESS | 0.9365 | 1.0000 | **0.9672** |
| IP_ADDRESS | 0.6815 | 0.9828 | **0.8049** |
| LOCATION | 0.4483 | 0.8371 | **0.5839** |
| ORGANIZATION | 0.5619 | 0.4504 | **0.5000** |
| PERSON | 0.7109 | 0.9751 | **0.8223** |
| PHONE_NUMBER | 0.7597 | 0.8596 | **0.8066** |
| SOCIAL_IDENTIFIER | 0.8462 | 0.4342 | **0.5739** |
| USERNAME | 0.8382 | 0.9797 | **0.9034** |

### Main observation

The strongest categories were:

```text
EMAIL_ADDRESS
DATE_TIME
USERNAME
PERSON
PHONE_NUMBER
IP_ADDRESS
```

while `ADDRESS` remained comparatively weak.

---

# 11. Cross-Dataset Comparison

## 11.1 Overall F1

At threshold `0.7`:

| Dataset | F1 |
|---|---:|
| **Nemotron** | **0.8221** |
| **Gretel** | **0.6813** |
| **Argilla** | **0.4925** |
| **AI4Privacy** | **0.4639** |

Relative to the Nemotron benchmark:

```text
Gretel      -0.1408 F1
Argilla     -0.3296 F1
AI4Privacy  -0.3582 F1
```

This demonstrates a substantial distribution-dependent generalization gap.

---

# 12. Recurring Strengths

Several entity categories remain strong across multiple OOD datasets.

## Email

Approximate F1 at threshold `0.7`:

| Dataset | Email F1 |
|---|---:|
| AI4Privacy | 0.9043 |
| Gretel | 0.9805 |
| Argilla | 0.9672 |

Email is the strongest recurring entity category across the OOD evaluation.

## Date/time

| Dataset | Date/time F1 |
|---|---:|
| AI4Privacy | 0.7254 |
| Gretel | 0.8475 |
| Argilla | 0.9425 |

## Phone

| Dataset | Phone F1 |
|---|---:|
| AI4Privacy | 0.7571 |
| Gretel | 0.8767 |
| Argilla | 0.8066 |

## Username

| Dataset | Username F1 |
|---|---:|
| AI4Privacy | 0.6973 |
| Gretel | 0.8805 |
| Argilla | 0.9034 |

These results indicate stronger generalization for PII categories with relatively distinctive patterns or semantic cues.

---

# 13. Recurring Weaknesses

## Address

Address is the clearest repeated weakness.

At threshold `0.7`:

| Dataset | Address F1 |
|---|---:|
| AI4Privacy | 0.2913 |
| Gretel | 0.3039 |
| Argilla | 0.3413 |

The low precision across all three datasets indicates that address detection is not simply a recall problem.

Many predictions are being made, but their exact boundaries do not align with the annotations.

## Location

Location performance varies more strongly by dataset, but precision problems recur.

Gretel:

```text
Precision = 0.1152
Recall    = 0.8319
F1        = 0.2024
```

Argilla:

```text
Precision = 0.4483
Recall    = 0.8371
F1        = 0.5839
```

This suggests substantial contextual ambiguity and/or exact-boundary differences.

## Organization

Organization is also relatively weak:

```text
Gretel  F1 = 0.4372
Argilla F1 = 0.5000
```

## Person

Person performance is dataset-dependent:

```text
AI4Privacy  F1 = 0.7405
Gretel      F1 = 0.5993
Argilla     F1 = 0.8223
```

Recall remains high in several datasets, while precision varies considerably.

---

# 14. Threshold Analysis

Across all three OOD datasets, threshold `0.7` produced a slightly higher overall F1 than `0.3`.

```text
AI4Privacy
0.4565 → 0.4639

Gretel
0.6758 → 0.6813

Argilla
0.4838 → 0.4925
```

The expected precision/recall tradeoff is consistent:

```text
threshold ↑
    ↓
fewer predictions
    ↓
FP ↓
precision ↑
recall ↓
```

However, the improvement from `0.3` to `0.7` is relatively small at the aggregate level.

This indicates that threshold selection alone does not address the primary OOD performance gap.

---

# 15. Interpretation

The benchmark provides several observations.

## 15.1 Nemotron performance does not represent all distributions

The frozen detector achieved:

```text
F1 = 0.8221
```

on the Nemotron width-24 benchmark, but lower scores on all three OOD datasets.

Therefore performance is distribution-dependent.

## 15.2 GLiNER still generalizes meaningfully

Gretel achieved:

```text
F1 = 0.6813
```

at threshold `0.7`, with several individual entity classes above `0.85` F1.

Therefore OOD degradation does not mean that GLiNER fails completely outside Nemotron.

## 15.3 Performance is highly entity-dependent

Structured entities such as:

```text
EMAIL
IP_ADDRESS
PHONE_NUMBER
DATE/TIME
```

are generally stronger.

Contextual entities such as:

```text
ADDRESS
LOCATION
ORGANIZATION
```

show more variable performance and more false positives.

## 15.4 Exact-span matching is important

Some semantically plausible predictions receive no credit when their boundaries differ from the annotation.

This is particularly relevant for contextual PII where entity boundaries are ambiguous.

## 15.5 Ontology compatibility affects aggregate results

AI4Privacy contains a large number of labels that are currently `UNMAPPED`.

Its gateway score must therefore be interpreted alongside its native-label score.

---

# 16. Limitations

This benchmark does not establish universal real-world performance.

## Dataset limitations

The datasets are public annotated/synthetic datasets and may not represent production traffic.

## Annotation differences

Different datasets use different annotation conventions and span boundaries.

Under exact-span scoring, these differences can materially affect F1.

## Ontology mapping

The current gateway ontology does not cover every source-dataset label.

## Dataset size

The datasets differ considerably in size:

```text
AI4Privacy   47,725
Gretel        5,000
Argilla       2,096
```

Therefore the statistical reliability of aggregate metrics differs between datasets.

## Threshold coverage

Only thresholds `0.3` and `0.7` were evaluated in the OOD benchmark.

## No hybrid detector

The email hybrid was intentionally excluded.

These results therefore describe:

> **Raw GLiNER-PII with the frozen `max_width=24` configuration.**

They do not represent the eventual production detector.

---

# 17. Reproducibility

## Model

```text
nvidia/gliner-PII
revision = bd23e8ef4425fd04e34c5204ab49ffaa706eae79
```

## Inference configuration

```text
max_width = 24
max_len = 384
dtype = FP32
batch_size = 1
thresholds = [0.3, 0.7]
```

## Dataset revisions

```text
AI4Privacy
c8c77895a005822682b66ab547fc0422579bc1d3

Gretel
e06eb1499ca8d54470f085021cd8e54f9efac7fd

Argilla
683954a23877151a1d05e7515d5e6cf87203ffd2
```

## Normalized dataset hashes

```text
AI4Privacy
575c7105a3ad5d27cbea7a467bebc360b08edfcdebadd71a45d5962a12472fca

Gretel
2b628ffc09effc56475300038c648252a5faf9eb4f8877c9221ef31e5ca727f9

Argilla
551e98472d8c0df3e5ec1590f0b62960d3be667266eadf65518715fd534ee6cf
```

---

# 18. Notebook / Pipeline Issues Resolved

During OOD evaluation, one infrastructure issue was identified.

AI4Privacy contains U+2028 Unicode line-separator characters inside source text.

The original runner used:

```python
normalized.read_text(encoding="utf-8").splitlines()
```

`str.splitlines()` treats U+2028 as a line boundary.

This caused valid JSONL records to be split internally, producing:

```text
JSONDecodeError:
Unterminated string
```

The normalized JSONL itself was valid.

The loader was corrected to use true JSONL record iteration:

```python
with normalized.open("r", encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle if line.strip()]
```

This ensures that only actual newline-delimited records are parsed as separate examples.

---

# 19. Phase 1.5 Status

```text
Presidio baseline                    ✅
GLiNER max_width=12                 ✅
GLiNER max_width=24                 ✅
Threshold selection                 ✅
Email hybrid experiment             ✅

AI4Privacy normalization             ✅
Gretel normalization                 ✅
Argilla normalization                ✅

AI4Privacy OOD baseline             ✅
Gretel OOD baseline                 ✅
Argilla OOD baseline                ✅

Cross-dataset OOD comparison        ✅
Error analysis                      → NEXT
Targeted detector improvements      → AFTER ERROR ANALYSIS
Re-evaluation                      → AFTER IMPROVEMENTS
```

---

# 20. Next Step

The next step is **aggregate error analysis**, not immediate model modification.

The evaluation has identified recurring areas requiring investigation:

```text
ADDRESS
LOCATION precision
ORGANIZATION
PERSON span precision
```

The next analysis should separate:

```text
False negatives
├── span mismatch
├── detector miss
├── unsupported / unmapped label
└── other

False positives
├── span mismatch
├── detector false positive
└── other
```

The goal is to identify recurring patterns before adding any new deterministic detectors or post-processing logic.

The experimental loop then becomes:

```text
OOD baseline
    ↓
error analysis
    ↓
targeted detector improvement
    ↓
re-run identical benchmark
    ↓
measure improvement
    ↓
freeze detector
```

---

# 21. Final Conclusion

The OOD evaluation shows that **raw GLiNER-PII with `max_width=24` generalizes beyond Nemotron, but performance varies substantially by dataset and entity type**.

The strongest recurring categories are:

```text
EMAIL
IP_ADDRESS
PHONE_NUMBER
DATE/TIME
USERNAME
```

The most consistent weakness is:

```text
ADDRESS
```

with additional precision issues for:

```text
LOCATION
ORGANIZATION
PERSON
```

The overall OOD F1 at threshold `0.7` was:

```text
AI4Privacy   0.4639
Gretel       0.6813
Argilla      0.4925
```

compared with:

```text
Nemotron     0.8221
```

This demonstrates a substantial distribution-dependent generalization gap and provides a concrete basis for the next stage of Phase 1.5: **error analysis and targeted detector improvements**.