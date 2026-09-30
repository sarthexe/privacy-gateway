# PII detector comparison: Presidio vs NVIDIA GLiNER-PII

Both detectors were scored on the **same** normalized Nemotron-PII test file (SHA-256 `52c89435c888740d2be5bab6c0221761ae7b5b178f0c007342d14a2449ad99d4`, 100,000 examples, 850,340 ground-truth entities), with the same ontology mapping, the same strict rule (type, start and end must all match; each ground-truth entity matches at most one prediction) and the same `nervaluate` strict cross-check. Neither model was trained or tuned on the test split. Sources: `presidio_baseline.json`, `gliner_baseline.json`.

This comparison deliberately names no overall winner. A single aggregate hides that the two detectors cover different entity types, cost very different amounts to run, and fail in different ways.

## Overall (gateway ontology)

| Detector | Precision | Recall | F1 | TP | FP | FN | Predictions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Presidio (Phase 1) | 0.4613 | 0.3139 | 0.3736 | 266,880 | 311,650 | 583,460 | 578,530 |
| GLiNER-PII (t=0.7) | 0.9444 | 0.7255 | 0.8206 | 616,883 | 36,350 | 233,457 | 653,233 |

`UNMAPPED` ground truth (203,906 entities, 24.0% of support) counts as a false negative for both detectors because the gateway ontology has no type for it. Recall over the 646,434 entities that have a gateway type: Presidio 0.4128, GLiNER-PII 0.9543 (reported for context only; not a substitute for the headline metric).

## Per entity type

| Type | Support | Presidio P | Presidio R | Presidio F1 | GLiNER P | GLiNER R | GLiNER F1 | F1 Δ (GLiNER − Presidio) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ADDRESS | 23,158 | 0.0000 | 0.0000 | 0.0000 | 0.9828 | 0.9792 | 0.9810 | +0.9810 |
| BANK_ACCOUNT | 30,606 | 0.0000 | 0.0000 | 0.0000 | 0.9864 | 0.9858 | 0.9861 | +0.9861 |
| CREDIT_CARD | 17,710 | 0.7720 | 0.0858 | 0.1545 | 0.9748 | 0.9909 | 0.9828 | +0.8283 |
| DATE_TIME | 129,673 | 0.4266 | 0.8285 | 0.5632 | 0.9283 | 0.9505 | 0.9393 | +0.3761 |
| EMAIL_ADDRESS | 53,930 | 0.9965 | 0.9913 | 0.9939 | 0.9967 | 0.7685 | 0.8678 | -0.1261 |
| ID_CARD | 32,227 | 0.0000 | 0.0000 | 0.0000 | 0.9793 | 0.9782 | 0.9788 | +0.9788 |
| IP_ADDRESS | 9,217 | 0.7327 | 0.9679 | 0.8340 | 0.9935 | 0.7256 | 0.8387 | +0.0047 |
| LOCATION | 76,264 | 0.6592 | 0.6188 | 0.6384 | 0.9645 | 0.9713 | 0.9679 | +0.3295 |
| ORGANIZATION | 54,837 | 0.0000 | 0.0000 | 0.0000 | 0.9375 | 0.9752 | 0.9560 | +0.9560 |
| PERSON | 143,639 | 0.1544 | 0.1377 | 0.1456 | 0.8932 | 0.9935 | 0.9407 | +0.7951 |
| PHONE_NUMBER | 30,353 | 0.4829 | 0.9410 | 0.6382 | 0.9952 | 0.9829 | 0.9890 | +0.3508 |
| SOCIAL_IDENTIFIER | 29,020 | 0.0000 | 0.0000 | 0.0000 | 0.9790 | 0.9861 | 0.9825 | +0.9825 |
| UNMAPPED | 203,906 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
| USERNAME | 15,800 | 0.0000 | 0.0000 | 0.0000 | 0.9610 | 0.9408 | 0.9508 | +0.9508 |

- GLiNER-PII materially higher F1 (Δ ≥ 0.05): ADDRESS, BANK_ACCOUNT, CREDIT_CARD, DATE_TIME, ID_CARD, LOCATION, ORGANIZATION, PERSON, PHONE_NUMBER, SOCIAL_IDENTIFIER, USERNAME.
- Presidio materially higher F1 (Δ ≤ −0.05): EMAIL_ADDRESS.
- Within ±0.05: IP_ADDRESS.

## Coverage

| | Presidio | GLiNER-PII |
| --- | --- | --- |
| Gateway types it can emit | 7: CREDIT_CARD, DATE_TIME, EMAIL_ADDRESS, IP_ADDRESS, LOCATION, PERSON, PHONE_NUMBER | 13: ADDRESS, BANK_ACCOUNT, CREDIT_CARD, DATE_TIME, EMAIL_ADDRESS, ID_CARD, IP_ADDRESS, LOCATION, ORGANIZATION, PERSON, PHONE_NUMBER, SOCIAL_IDENTIFIER, USERNAME |
| Native label space | Phase 1 `EntityType` (7 types) | 55 Nemotron-PII labels; 55 returned |
| Predictions dropped by ontology filtering | n/a (Phase 1 types only) | 216,063 (170,025 were exact native matches) |
| `UNMAPPED` ground truth detected exactly (native label) | 0 (no such labels) | 170,025 of 203,906 |
| Native-label strict F1 (all labels, pre-ontology) | n/a | 0.9118 |

Presidio's zero scores for ADDRESS, BANK_ACCOUNT, ID_CARD, ORGANIZATION, SOCIAL_IDENTIFIER and USERNAME are ontology/recognizer gaps (Phase 1 never emits those types), not measured detection failures. GLiNER-PII's `UNMAPPED` losses are the gateway ontology's gap: the model often detects those spans, but they have no gateway type to be scored as.

## Runtime and model size

| | Presidio | GLiNER-PII |
| --- | --- | --- |
| Model | rule/regex recognizers + spaCy `en_core_web_sm` 3.8.0 NER | `nvidia/gliner-PII` @ `bd23e8ef4425`, microsoft/deberta-v3-large |
| Size | ~15 MB spaCy model + ~2.4 MB presidio-analyzer package; no GPU | 445,463,040 parameters; 1.78 GB fp32 checkpoint; run as `fp32` |
| Hardware | CPU, 8 worker processes | NVIDIA GeForce RTX 2050 (4.0 GiB), 1 process |
| Detection time (100k) | 477.4 s | 29,468.2 s (+ 25.9 s model load) |
| Throughput | 209.5 examples/s | 3.39 examples/s |

Both runs used the same laptop (12th Gen Intel Core i5-12450H). GLiNER-PII needs a GPU to be practical: on this machine's CPU it measured 0.19 examples/s in the pilot.

## Failure modes

Presidio-harness categories (identical definitions for both detectors):

| False-negative group | Presidio | GLiNER-PII |
| --- | ---: | ---: |
| span_mismatch | 96,267 | 2,314 |
| supported_direct | 97,639 | 23,295 |
| unmapped_source_label | 203,906 | 203,906 |
| unsupported_entity | 185,648 | 3,942 |

| False-positive group | Presidio | GLiNER-PII |
| --- | ---: | ---: |
| detector_recognizer | 255,062 | 34,018 |
| span_mismatch | 56,588 | 2,332 |

Group meanings: `span_mismatch` = same type, overlapping but different offsets; `unmapped_source_label` = UNMAPPED ground truth; `unsupported_entity` = a gateway type Phase 1 does not implement (a Phase 1 label, independent of detector); `supported_direct` = other misses of Phase 1 types; `detector_recognizer` = predictions that do not overlap a same-type entity.

GLiNER-PII finer taxonomy (see `gliner_baseline.md`): missed_entity 11,529, overlapping_prediction 13,803, unmapped_ground_truth_label 203,906, unmapped_prediction_label 963, wrong_boundaries 2,314, wrong_entity_type 942 (false negatives); false_positive 13,132, overlapping_prediction 19,509, wrong_boundaries 2,332, wrong_entity_type 1,377 (false positives).

## Implications for a future hybrid detector

- Route or union by entity type rather than choosing one model; the per-type table shows where each is materially stronger.
- EMAIL_ADDRESS: Presidio F1 0.9939 (recall 0.9913) vs GLiNER-PII F1 0.8678 (recall 0.7685, precision 0.9967). Native labels GLiNER predicted over its misses: `first_name` 9,076, `last_name` 6,034, `company_name` 971.
- IP_ADDRESS: Presidio F1 0.8340 (recall 0.9679) vs GLiNER-PII F1 0.8387 (recall 0.7256, precision 0.9935). Native labels GLiNER predicted over its misses: `customer_id` 1, `ipv4` 1, `ipv6` 1.
- `email`: spans are almost always short enough for GLiNER (2 too wide), yet 12,482 of 53,927 are missed; the overlap labels above show flat-NER decoding keeping higher-scoring sub-spans inside the value instead. A pattern recognizer run before the model would avoid this.
- `ipv6`: 2,521 of 3,139 ground-truth spans are wider than GLiNER's 12-word `max_width` and all 2,521 are missed, while only 3 of 618 reachable spans are missed: a hard architectural limit that a pattern recognizer does not have.
- `url`: 17,741 of 37,847 ground-truth spans are wider than GLiNER's 12-word `max_width` and all 17,741 are missed, while only 175 of 20,106 reachable spans are missed: a hard architectural limit that a pattern recognizer does not have.
- `http_cookie`: 3,193 of 4,892 ground-truth spans are wider than GLiNER's 12-word `max_width` and all 3,193 are missed, while only 116 of 1,690 reachable spans are missed: a hard architectural limit that a pattern recognizer does not have.
- PERSON: GLiNER-PII recall 0.9935, precision 0.8932; 16,323 of its 17,061 false positives overlap a ground-truth entity of another type (e.g. name parts inside email addresses).
- Types Phase 1 Presidio cannot emit at all (ADDRESS, BANK_ACCOUNT, ID_CARD, ORGANIZATION, SOCIAL_IDENTIFIER, USERNAME) are where a learned detector adds coverage rather than accuracy.
- `UNMAPPED` is an ontology gap: GLiNER-PII exactly matched 170,025 of 203,906 such entities under their native labels; extending the gateway ontology, not changing models, is what would let a detector be credited for them.
- Caveat: Nemotron-PII is GLiNER-PII's training distribution (train split; the test split is held out and shares no uid/text with train), so this benchmark favours GLiNER relative to real gateway traffic. A hybrid decision needs an out-of-distribution evaluation as well.
