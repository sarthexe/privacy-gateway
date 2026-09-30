# Presidio baseline: Nemotron-PII test split

## Dataset and methodology

- Dataset: `nvidia/Nemotron-PII`, normalized `test.jsonl`; this report covers the complete test split.
- Matching: exact entity type and half-open `[start, end)` offsets.
- Scoring: order-independent multiset match of `(type, start, end)`; each prediction can match at most one identical ground-truth entity.
- Harness: `nervaluate` strict strategy must reproduce the exact-match count on every checkpointed batch (batch size 1000).
- No source text, entity values, or raw annotations are persisted in this report.

## Overall metrics

| Precision | Recall | F1 | Support | Predictions | Exact matches |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.4613 | 0.3139 | 0.3736 | 850340 | 578530 | 266880 |

Examples evaluated: 100000; examples containing PII: 99999.

## Per-entity metrics

| Type | Precision | Recall | F1 | Support | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ADDRESS | 0.0000 | 0.0000 | 0.0000 | 23158 | 0 | 23158 |
| BANK_ACCOUNT | 0.0000 | 0.0000 | 0.0000 | 30606 | 0 | 30606 |
| CREDIT_CARD | 0.7720 | 0.0858 | 0.1545 | 17710 | 449 | 16190 |
| DATE_TIME | 0.4266 | 0.8285 | 0.5632 | 129673 | 144405 | 22234 |
| EMAIL_ADDRESS | 0.9965 | 0.9913 | 0.9939 | 53930 | 187 | 469 |
| ID_CARD | 0.0000 | 0.0000 | 0.0000 | 32227 | 0 | 32227 |
| IP_ADDRESS | 0.7327 | 0.9679 | 0.8340 | 9217 | 3255 | 296 |
| LOCATION | 0.6592 | 0.6188 | 0.6384 | 76264 | 24398 | 29071 |
| ORGANIZATION | 0.0000 | 0.0000 | 0.0000 | 54837 | 0 | 54837 |
| PERSON | 0.1544 | 0.1377 | 0.1456 | 143639 | 108366 | 123855 |
| PHONE_NUMBER | 0.4829 | 0.9410 | 0.6382 | 30353 | 30590 | 1791 |
| SOCIAL_IDENTIFIER | 0.0000 | 0.0000 | 0.0000 | 29020 | 0 | 29020 |
| UNMAPPED | 0.0000 | 0.0000 | 0.0000 | 203906 | 0 | 203906 |
| USERNAME | 0.0000 | 0.0000 | 0.0000 | 15800 | 0 | 15800 |

## Error analysis

Top false-negative categories: `{"span_mismatch": 96267, "supported_direct": 97639, "unmapped_source_label": 203906, "unsupported_entity": 185648}`.

Top false-positive categories: `{"detector_recognizer": 255062, "span_mismatch": 56588}`.

## Unsupported and unmapped labels

`UNMAPPED` and entity types not implemented by Phase 1 are retained in scoring as ground truth; they cannot produce a Presidio exact match. DATE and TIME are evaluated through the documented adapter mapping to Phase 1 `DATE_TIME`.

## Harness note

nervaluate 1.2's strict matcher is greedy in prediction order: when Presidio emits nested same-type spans (common for `DATE_TIME`), an overlapping prediction listed before the exact one consumes the true entity as *incorrect* and the exact prediction is scored *spurious*. Exact matching is order-independent, so the cross-check lists exact matches first. In detector order nervaluate counts 266496 exact matches versus 266880.

## Run

- Detection time: 477.4 s across 2 session(s); throughput 209.5 examples/s.
- Detector: Presidio Analyzer 2.2.364, spaCy 3.8.16, models `{'en': 'en_core_web_sm==3.8.0'}`.
- Dataset SHA-256: `52c89435c888740d2be5bab6c0221761ae7b5b178f0c007342d14a2449ad99d4`.

## Limitations

This is a zero-tuning baseline. Exact spans penalize boundary differences, and Presidio's built-in recognizers do not cover many Nemotron PII/PHI categories.

## Reproduce

```bash
python scripts/evaluate_presidio.py --data-dir /path/to/nemotron-normalized --workers 4 --resume
```
