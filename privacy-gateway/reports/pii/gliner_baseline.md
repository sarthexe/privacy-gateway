# NVIDIA GLiNER-PII baseline: Nemotron-PII test split

Pretrained `nvidia/gliner-PII`, evaluated without any training or fine-tuning, on the same normalized test file, ground truth, ontology and strict matching as `presidio_baseline.md`. No source text, entity values, or model inputs are stored.

## Model

- Model: `nvidia/gliner-PII` at revision `bd23e8ef4425fd04e34c5204ab49ffaa706eae79` (gliner 0.2.29, torch 2.11.0+cu128, transformers 5.16.1).
- Encoder `microsoft/deberta-v3-large`, span mode `markerV0`, 445,463,040 parameters as counted by torch (the model card states 5.7 × 10^8); inference weights `fp32`.
- Input limit `max_len=384` words; longest predictable span `max_width=12` words; `whitespace` splitter.

## Dataset and methodology

- Dataset: `nvidia/Nemotron-PII`, normalized `test.jsonl`; this report covers the complete test split. SHA-256 `52c89435c888740d2be5bab6c0221761ae7b5b178f0c007342d14a2449ad99d4` (identical to the Presidio baseline).
- Matching: a prediction is a true positive only if entity type, start and end all equal a ground-truth entity (half-open offsets); multiset matching, so each ground-truth entity matches at most one prediction. No overlap or token credit.
- Scoring reuses the Presidio harness (`score_batch`): the `nervaluate` strict cross-check must reproduce the exact-match count on every 1,000-example batch.
- Ground truth: normalized Nemotron labels via `configs/pii_ontology.yaml`, with DATE and TIME scored as `DATE_TIME`, exactly as for Presidio. `UNMAPPED` ground truth stays in the denominator.
- Predictions: every GLiNER label maps through `configs/gliner_label_map.yaml` (table below). Predictions whose label maps to `UNMAPPED` are removed before gateway scoring and counted under Coverage; they are not silently discarded.

## Inference configuration

- Prompt: all 55 native Nemotron-PII labels in one pass (the model's training vocabulary).
- `threshold=0.7`, `flat_ner=True`, `multi_label=False`, batch size 1, device `cuda`.
- Texts over 384 words (the library would silently truncate them) are split into word windows overlapping by 24 words; each window keeps predictions starting in its own centre region, so no predictable entity is cut. Shorter texts are one unchanged call.
- Chunked examples: 11,510; model calls (chunks): 112,182.

## Threshold selection (development subset only)

- Source: 2,000 records of Nemotron-PII **train** (`train.jsonl`, SHA-256 `0e10ceb2d312b2d7…`), chosen by ranking indices by SHA-256 of `"privacy-gateway/gliner-dev/v1:<index>"`. The test split was not read during selection.
- Rule (fixed before scoring): maximize development-subset gateway-ontology micro-F1 (strict exact span/type); ties within 0.0001 go to the lower threshold.
- One dev inference pass at the lowest candidate; higher thresholds are the same predictions filtered by `score > t`, which equals a direct run because the flat-NER decoder is greedy in descending score order.
- Verified against a direct run at t=0.5: 300/300 examples identical (0 / 0 predictions differ).
- Caveat: The development subset is drawn from the Nemotron-PII train split, which the model card reports as GLiNER-PII's training data; dev scores are in-sample and likely optimistic. The test split was not read.
- NVIDIA's model card reports evaluating at t=0.3.

| t | Dev precision | Dev recall | Dev F1 | Dev native F1 |  |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.2 | 0.9287 | 0.7203 | 0.8114 | 0.8718 |  |
| 0.3 | 0.9361 | 0.7200 | 0.8139 | 0.8847 |  |
| 0.4 | 0.9409 | 0.7197 | 0.8156 | 0.8933 |  |
| 0.5 | 0.9435 | 0.7194 | 0.8164 | 0.8996 |  |
| 0.6 | 0.9470 | 0.7190 | 0.8174 | 0.9051 |  |
| 0.7 | 0.9496 | 0.7186 | 0.8181 | 0.9100 | **selected** |

Selected threshold: **0.7**. The final test split was then run once at this threshold. The dev curve is nearly flat and still rising at the grid's upper edge; because dev records are in-sample (the model is very confident on them), this procedure likely favours a higher threshold than a held-out dev set would.

### Inference configuration pilot (development records only)

Chosen: `cuda`, `fp32`, batch size 1, 1 process. fp32/batch 1 is the library's default precision and exactly the unbatched predict_entities path; fp32 batching gave no speedup on the 4 GB GPU. fp16/batch 4 was 2.2x faster but changed predictions on 69/2,000 dev texts and raised dev gateway F1 by ~0.003 (+33 EMAIL_ADDRESS exact matches), a systematic shift rather than noise, so it was not used. Decided on dev data only.

| Device:dtype:batch | Examples/s | GPU peak MiB | Identical texts |
| --- | ---: | ---: | ---: |
| `cpu:fp32:4` | 0.186 | n/a | 8/8 |
| `cuda:bf16:8` | 6.14 | 2476 | 62/64 |
| `cuda:fp16:1` | 5.633 | 2006 | 63/64 |
| `cuda:fp16:4` | 6.607 | 2006 | 64/64 |
| `cuda:fp16:8` | 6.176 | 2474 | 64/64 |
| `cuda:fp32:1` | 2.973 | 2218 | 64/64 |
| `cuda:fp32:4` | 2.743 | 3430 | 64/64 |

fp16 sensitivity on the same 2,000 dev records (not used for the test run):

| t | fp32 dev F1 | fp16 dev F1 |
| --- | ---: | ---: |
| 0.2 | 0.8114 | 0.8141 |
| 0.3 | 0.8139 | 0.8167 |
| 0.4 | 0.8156 | 0.8184 |
| 0.5 | 0.8164 | 0.8191 |
| 0.6 | 0.8174 | 0.8202 |
| 0.7 | 0.8181 | 0.8209 |

## Overall metrics (gateway ontology)

| Precision | Recall | F1 | TP | FP | FN | Support | Predictions |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.9444 | 0.7255 | 0.8206 | 616883 | 36350 | 233457 | 850340 | 653233 |

Examples evaluated: 100,000; examples containing PII: 99,999; ground-truth entities: 850,340; predictions: 653,233; exact matches: 616,883 (nervaluate strict cross-check: 616,883).

## Per-entity metrics (gateway ontology)

| Type | Precision | Recall | F1 | TP | FP | FN | Support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ADDRESS | 0.9828 | 0.9792 | 0.9810 | 22676 | 396 | 482 | 23158 |
| BANK_ACCOUNT | 0.9864 | 0.9858 | 0.9861 | 30170 | 416 | 436 | 30606 |
| CREDIT_CARD | 0.9748 | 0.9909 | 0.9828 | 17549 | 453 | 161 | 17710 |
| DATE_TIME | 0.9283 | 0.9505 | 0.9393 | 123254 | 9517 | 6419 | 129673 |
| EMAIL_ADDRESS | 0.9967 | 0.7685 | 0.8678 | 41445 | 138 | 12485 | 53930 |
| ID_CARD | 0.9793 | 0.9782 | 0.9788 | 31526 | 666 | 701 | 32227 |
| IP_ADDRESS | 0.9935 | 0.7256 | 0.8387 | 6688 | 44 | 2529 | 9217 |
| LOCATION | 0.9645 | 0.9713 | 0.9679 | 74077 | 2730 | 2187 | 76264 |
| ORGANIZATION | 0.9375 | 0.9752 | 0.9560 | 53476 | 3565 | 1361 | 54837 |
| PERSON | 0.8932 | 0.9935 | 0.9407 | 142706 | 17061 | 933 | 143639 |
| PHONE_NUMBER | 0.9952 | 0.9829 | 0.9890 | 29833 | 145 | 520 | 30353 |
| SOCIAL_IDENTIFIER | 0.9790 | 0.9861 | 0.9825 | 28618 | 615 | 402 | 29020 |
| UNMAPPED | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 203906 | 203906 |
| USERNAME | 0.9610 | 0.9408 | 0.9508 | 14865 | 604 | 935 | 15800 |

## Coverage before ontology filtering

The gateway view above cannot credit a detection whose label the gateway ontology does not have. This section scores the model in its **native** label space (GLiNER label vs raw Nemotron label, same exact span rule) to separate detection failure from ontology mismatch. These numbers are not comparable to the gateway metrics and are not mixed into them.

- Prompt labels: 55; labels the model actually returned: 55; never returned: none.
- Predictions before ontology filtering: 869,296; removed by ontology filtering (label maps to UNMAPPED): 216,063, of which 170,025 exactly match a native ground-truth entity (correct detections the ontology drops).
- Gateway-mapped labels returned: 32; unmapped labels returned: 23.

Native-label exact match (all 55 labels):

| Precision | Recall | F1 | TP | FP | FN |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.9018 | 0.9219 | 0.9118 | 783948 | 85348 | 66392 |

`UNMAPPED` ground truth (203,906 entities) as seen by the model:

| Outcome | Entities | Share |
| --- | ---: | ---: |
| exact native match | 170025 | 83.4% |
| exact span other native label | 1314 | 0.6% |
| no prediction | 29411 | 14.4% |
| overlapping prediction | 3156 | 1.5% |

Per native label (gateway type in brackets):

| Native label | Gateway | Precision | Recall | F1 | TP | FP | FN | Support |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| account_number | BANK_ACCOUNT | 0.9834 | 0.9823 | 0.9829 | 16398 | 277 | 295 | 16693 |
| age | UNMAPPED | 0.7839 | 0.8969 | 0.8366 | 6428 | 1772 | 739 | 7167 |
| api_key | UNMAPPED | 0.9867 | 0.9891 | 0.9879 | 4616 | 62 | 51 | 4667 |
| bank_routing_number | BANK_ACCOUNT | 0.9955 | 0.9904 | 0.9930 | 8274 | 37 | 80 | 8354 |
| biometric_identifier | UNMAPPED | 0.9966 | 0.9895 | 0.9931 | 11260 | 38 | 119 | 11379 |
| blood_type | UNMAPPED | 0.9724 | 0.9904 | 0.9813 | 5486 | 156 | 53 | 5539 |
| certificate_license_number | UNMAPPED | 0.9460 | 0.9694 | 0.9576 | 2910 | 166 | 92 | 3002 |
| city | LOCATION | 0.9239 | 0.9624 | 0.9428 | 17658 | 1455 | 689 | 18347 |
| company_name | ORGANIZATION | 0.9375 | 0.9752 | 0.9560 | 53476 | 3565 | 1361 | 54837 |
| coordinate | LOCATION | 0.9917 | 0.9961 | 0.9939 | 7629 | 64 | 30 | 7659 |
| country | LOCATION | 0.9525 | 0.9829 | 0.9675 | 23074 | 1151 | 401 | 23475 |
| county | LOCATION | 0.9509 | 0.9109 | 0.9305 | 7711 | 398 | 754 | 8465 |
| credit_debit_card | CREDIT_CARD | 0.9827 | 0.9924 | 0.9875 | 12769 | 225 | 98 | 12867 |
| customer_id | ID_CARD | 0.9778 | 0.9801 | 0.9789 | 20094 | 457 | 408 | 20502 |
| cvv | CREDIT_CARD | 0.9513 | 0.9837 | 0.9672 | 4764 | 244 | 79 | 4843 |
| date | DATE | 0.9308 | 0.9593 | 0.9448 | 70860 | 5266 | 3007 | 73867 |
| date_of_birth | DATE | 0.9864 | 0.9981 | 0.9922 | 18045 | 249 | 34 | 18079 |
| date_time | DATE | 0.9774 | 0.9564 | 0.9668 | 12645 | 292 | 576 | 13221 |
| device_identifier | UNMAPPED | 0.9438 | 0.9378 | 0.9408 | 2351 | 140 | 156 | 2507 |
| education_level | UNMAPPED | 0.8637 | 0.8828 | 0.8732 | 8279 | 1306 | 1099 | 9378 |
| email | EMAIL_ADDRESS | 0.9967 | 0.7685 | 0.8678 | 41445 | 138 | 12485 | 53930 |
| employee_id | ID_CARD | 0.9900 | 0.9729 | 0.9814 | 8637 | 87 | 241 | 8878 |
| employment_status | UNMAPPED | 0.9067 | 0.9884 | 0.9458 | 10890 | 1121 | 128 | 11018 |
| fax_number | PHONE_NUMBER | 0.9425 | 0.8876 | 0.9142 | 5701 | 348 | 722 | 6423 |
| first_name | PERSON | 0.8898 | 0.9918 | 0.9380 | 83353 | 10323 | 690 | 84043 |
| gender | UNMAPPED | 0.9353 | 0.9864 | 0.9601 | 7541 | 522 | 104 | 7645 |
| health_plan_beneficiary_number | SOCIAL_IDENTIFIER | 0.9907 | 0.9846 | 0.9876 | 10395 | 98 | 163 | 10558 |
| http_cookie | UNMAPPED | 0.9104 | 0.3217 | 0.4755 | 1574 | 155 | 3318 | 4892 |
| ipv4 | IP_ADDRESS | 0.9957 | 0.9988 | 0.9973 | 6071 | 26 | 7 | 6078 |
| ipv6 | IP_ADDRESS | 0.9685 | 0.1959 | 0.3259 | 615 | 20 | 2524 | 3139 |
| language | UNMAPPED | 0.9426 | 0.9150 | 0.9286 | 7916 | 482 | 735 | 8651 |
| last_name | PERSON | 0.8943 | 0.9917 | 0.9405 | 59102 | 6989 | 494 | 59596 |
| license_plate | UNMAPPED | 0.9813 | 0.9929 | 0.9871 | 4049 | 77 | 29 | 4078 |
| mac_address | UNMAPPED | 0.9983 | 0.9944 | 0.9963 | 4226 | 7 | 24 | 4250 |
| medical_record_number | SOCIAL_IDENTIFIER | 0.9936 | 0.9886 | 0.9911 | 10971 | 71 | 127 | 11098 |
| national_id | ID_CARD | 0.9469 | 0.9701 | 0.9584 | 2762 | 155 | 85 | 2847 |
| occupation | UNMAPPED | 0.4469 | 0.8023 | 0.5740 | 29594 | 36629 | 7294 | 36888 |
| password | UNMAPPED | 0.9810 | 0.9686 | 0.9748 | 6766 | 131 | 219 | 6985 |
| phone_number | PHONE_NUMBER | 0.9701 | 0.9700 | 0.9701 | 23213 | 716 | 717 | 23930 |
| pin | UNMAPPED | 0.9655 | 0.9507 | 0.9581 | 6138 | 219 | 318 | 6456 |
| political_view | UNMAPPED | 0.8775 | 0.9308 | 0.9034 | 5959 | 832 | 443 | 6402 |
| postcode | ADDRESS | 0.9683 | 0.9873 | 0.9777 | 6199 | 203 | 80 | 6279 |
| race_ethnicity | UNMAPPED | 0.9466 | 0.9549 | 0.9507 | 8831 | 498 | 417 | 9248 |
| religious_belief | UNMAPPED | 0.9500 | 0.9389 | 0.9445 | 5875 | 309 | 382 | 6257 |
| sexuality | UNMAPPED | 0.9502 | 0.9680 | 0.9590 | 3206 | 168 | 106 | 3312 |
| ssn | SOCIAL_IDENTIFIER | 0.9296 | 0.9797 | 0.9540 | 5939 | 450 | 123 | 6062 |
| state | LOCATION | 0.9577 | 0.9236 | 0.9403 | 16919 | 748 | 1399 | 18318 |
| street_address | ADDRESS | 0.9882 | 0.9759 | 0.9820 | 16473 | 197 | 406 | 16879 |
| swift_bic | BANK_ACCOUNT | 0.9746 | 0.9818 | 0.9782 | 5458 | 142 | 101 | 5559 |
| tax_id | SOCIAL_IDENTIFIER | 0.9824 | 0.9877 | 0.9851 | 1286 | 23 | 16 | 1302 |
| time | TIME | 0.8311 | 0.8619 | 0.8462 | 21122 | 4292 | 3384 | 24506 |
| unique_id | UNMAPPED | 0.6700 | 0.9403 | 0.7825 | 1671 | 823 | 106 | 1777 |
| url | UNMAPPED | 0.9857 | 0.5266 | 0.6865 | 19931 | 290 | 17916 | 37847 |
| user_name | USERNAME | 0.9610 | 0.9408 | 0.9508 | 14865 | 604 | 935 | 15800 |
| vehicle_identifier | UNMAPPED | 0.9710 | 0.9928 | 0.9818 | 4528 | 135 | 33 | 4561 |

## Error analysis (gateway ontology)

Each unmatched ground-truth entity (false negative) and each unmatched prediction (false positive) gets exactly one category, checked in the order listed.

| False-negative category | Count | Share | Meaning |
| --- | ---: | ---: | --- |
| unmapped_ground_truth_label | 203906 | 87.3% | ground-truth label is UNMAPPED in the gateway ontology (never matchable in this view; see Coverage) |
| wrong_entity_type | 942 | 0.4% | a scored prediction has the exact span but a different type |
| wrong_boundaries | 2314 | 1.0% | a scored prediction of the same type overlaps with different offsets |
| overlapping_prediction | 13803 | 5.9% | only overlapped by scored predictions of other types and offsets |
| unmapped_prediction_label | 963 | 0.4% | only overlapped by predictions removed by ontology filtering (model label maps to UNMAPPED) |
| missed_entity | 11529 | 4.9% | no prediction of any label overlaps it |

| False-positive category | Count | Share | Meaning |
| --- | ---: | ---: | --- |
| wrong_entity_type | 1377 | 3.8% | exact span of a ground-truth entity of another type (incl. UNMAPPED) |
| wrong_boundaries | 2332 | 6.4% | overlaps a ground-truth entity of the same type, different offsets |
| overlapping_prediction | 19509 | 53.7% | overlaps only ground-truth entities of other types and offsets |
| false_positive | 13132 | 36.1% | overlaps no ground-truth entity |

- Overlapping false positives that are nested in / contain a ground-truth span: 23,210.
- Prediction pairs overlapping each other: 0 (flat-NER decoding forbids overlap within one model call, so only chunk-window seams could produce any). Duplicate predictions removed: 0.
- Structurally unreachable ground truth (no whole-word span of at most `max_width` words can equal it): misaligned_word_boundary 3,514, wider_than_max_width 23,620; of these still false negatives: misaligned_word_boundary 3,514, wider_than_max_width 23,620.

Structural reachability by native label (labels with any unreachable ground truth; native exact-match misses in brackets):

| Native label | Reachable (missed) | Wider than max_width (missed) | Misaligned boundary (missed) |
| --- | ---: | ---: | ---: |
| `account_number` | 16,687 (289) | 3 (3) | 3 (3) |
| `age` | 6,536 (108) | 0 (0) | 631 (631) |
| `biometric_identifier` | 11,378 (118) | 0 (0) | 1 (1) |
| `certificate_license_number` | 3,001 (91) | 0 (0) | 1 (1) |
| `city` | 18,324 (666) | 0 (0) | 23 (23) |
| `company_name` | 54,831 (1,355) | 0 (0) | 6 (6) |
| `coordinate` | 7,641 (12) | 18 (18) | 0 (0) |
| `country` | 23,316 (242) | 0 (0) | 159 (159) |
| `county` | 8,357 (646) | 0 (0) | 108 (108) |
| `customer_id` | 20,467 (373) | 0 (0) | 35 (35) |
| `cvv` | 4,841 (77) | 0 (0) | 2 (2) |
| `date` | 73,027 (2,167) | 0 (0) | 840 (840) |
| `date_of_birth` | 18,078 (33) | 0 (0) | 1 (1) |
| `date_time` | 13,156 (511) | 54 (54) | 11 (11) |
| `device_identifier` | 2,505 (154) | 0 (0) | 2 (2) |
| `education_level` | 9,292 (1,013) | 0 (0) | 86 (86) |
| `email` | 53,927 (12,482) | 2 (2) | 1 (1) |
| `employee_id` | 8,876 (239) | 0 (0) | 2 (2) |
| `employment_status` | 10,941 (51) | 0 (0) | 77 (77) |
| `fax_number` | 6,422 (721) | 0 (0) | 1 (1) |
| `first_name` | 84,009 (656) | 0 (0) | 34 (34) |
| `gender` | 7,639 (98) | 0 (0) | 6 (6) |
| `http_cookie` | 1,690 (116) | 3,193 (3,193) | 9 (9) |
| `ipv6` | 618 (3) | 2,521 (2,521) | 0 (0) |
| `language` | 8,392 (476) | 0 (0) | 259 (259) |
| `last_name` | 59,553 (451) | 0 (0) | 43 (43) |
| `mac_address` | 4,247 (21) | 3 (3) | 0 (0) |
| `national_id` | 2,845 (83) | 1 (1) | 1 (1) |
| `occupation` | 36,042 (6,448) | 22 (22) | 824 (824) |
| `password` | 6,982 (216) | 1 (1) | 2 (2) |
| `pin` | 6,453 (315) | 0 (0) | 3 (3) |
| `political_view` | 6,332 (373) | 0 (0) | 70 (70) |
| `postcode` | 6,274 (75) | 0 (0) | 5 (5) |
| `race_ethnicity` | 9,206 (375) | 0 (0) | 42 (42) |
| `religious_belief` | 6,234 (359) | 0 (0) | 23 (23) |
| `sexuality` | 3,233 (27) | 0 (0) | 79 (79) |
| `ssn` | 6,060 (121) | 0 (0) | 2 (2) |
| `state` | 18,272 (1,353) | 0 (0) | 46 (46) |
| `street_address` | 16,790 (317) | 61 (61) | 28 (28) |
| `time` | 24,458 (3,336) | 0 (0) | 48 (48) |
| `url` | 20,106 (175) | 17,741 (17,741) | 0 (0) |

Native labels predicted over unmatched gateway entities (top 5 per type; shows what took the span instead, e.g. name parts inside an email address):

| Type | Overlapping native labels (count) |
| --- | --- |
| ADDRESS | `street_address` 143, `city` 78, `postcode` 44, `state` 24, `last_name` 21 |
| BANK_ACCOUNT | `unique_id` 106, `customer_id` 79, `credit_debit_card` 21, `health_plan_beneficiary_number` 12, `user_name` 7 |
| CREDIT_CARD | `account_number` 20, `pin` 20, `phone_number` 8, `bank_routing_number` 7, `password` 3 |
| DATE_TIME | `date` 568, `time` 517, `date_time` 183, `age` 18, `customer_id` 4 |
| EMAIL_ADDRESS | `first_name` 9,076, `last_name` 6,034, `company_name` 971, `user_name` 330, `email` 30 |
| ID_CARD | `unique_id` 242, `account_number` 68, `user_name` 31, `certificate_license_number` 15, `medical_record_number` 12 |
| IP_ADDRESS | `customer_id` 1, `political_view` 1, `ipv4` 1, `ipv6` 1 |
| LOCATION | `city` 319, `county` 202, `state` 149, `company_name` 107, `country` 82 |
| ORGANIZATION | `country` 527, `company_name` 204, `city` 48, `language` 18, `last_name` 14 |
| PERSON | `user_name` 113, `occupation` 76, `company_name` 66, `first_name` 48, `last_name` 47 |
| PHONE_NUMBER | `pin` 28, `account_number` 12, `medical_record_number` 9, `customer_id` 7, `credit_debit_card` 5 |
| SOCIAL_IDENTIFIER | `unique_id` 68, `customer_id` 53, `national_id` 31, `account_number` 17, `credit_debit_card` 15 |
| USERNAME | `first_name` 723, `last_name` 577, `email` 29, `employee_id` 16, `company_name` 12 |

False-negative categories by gateway type:

| Type | unmapped_ground_truth_label | wrong_entity_type | wrong_boundaries | overlapping_prediction | unmapped_prediction_label | missed_entity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ADDRESS | 0 | 35 | 152 | 44 | 14 | 237 |
| BANK_ACCOUNT | 0 | 131 | 3 | 4 | 133 | 165 |
| CREDIT_CARD | 0 | 43 | 3 | 0 | 29 | 86 |
| DATE_TIME | 0 | 6 | 1058 | 3 | 30 | 5322 |
| EMAIL_ADDRESS | 0 | 29 | 30 | 12211 | 7 | 208 |
| ID_CARD | 0 | 146 | 6 | 5 | 289 | 255 |
| IP_ADDRESS | 0 | 1 | 2 | 0 | 1 | 2525 |
| LOCATION | 0 | 11 | 755 | 127 | 160 | 1134 |
| ORGANIZATION | 0 | 28 | 204 | 585 | 43 | 501 |
| PERSON | 0 | 139 | 89 | 108 | 122 | 475 |
| PHONE_NUMBER | 0 | 48 | 2 | 0 | 40 | 430 |
| SOCIAL_IDENTIFIER | 0 | 141 | 1 | 2 | 89 | 169 |
| UNMAPPED | 203906 | 0 | 0 | 0 | 0 | 0 |
| USERNAME | 0 | 184 | 9 | 714 | 6 | 22 |

False-positive categories by gateway type:

| Type | wrong_entity_type | wrong_boundaries | overlapping_prediction | false_positive |
| --- | ---: | ---: | ---: | ---: |
| ADDRESS | 71 | 181 | 9 | 135 |
| BANK_ACCOUNT | 161 | 3 | 1 | 251 |
| CREDIT_CARD | 65 | 3 | 4 | 381 |
| DATE_TIME | 27 | 1074 | 566 | 7850 |
| EMAIL_ADDRESS | 9 | 30 | 48 | 51 |
| ID_CARD | 374 | 6 | 15 | 271 |
| IP_ADDRESS | 4 | 2 | 1 | 37 |
| LOCATION | 147 | 757 | 1006 | 820 |
| ORGANIZATION | 53 | 204 | 1197 | 2111 |
| PERSON | 179 | 59 | 16323 | 500 |
| PHONE_NUMBER | 22 | 2 | 1 | 120 |
| SOCIAL_IDENTIFIER | 113 | 1 | 3 | 498 |
| USERNAME | 152 | 10 | 335 | 107 |

Offset-only samples (first per category; uid is Nemotron's):

| Category | uid | Index | Type | Start | End |
| --- | --- | ---: | --- | ---: | ---: |
| fn_missed_entity | 3e3f8e5a1e064d5ba44271375c59f1cb | 4 | DATE_TIME | 250 | 255 |
| fn_missed_entity | db462f03d2a54386bb11cb452ed35283 | 10 | BANK_ACCOUNT | 484 | 493 |
| fn_overlapping_prediction | 9179275a56154824a36257eb5181b88c | 47 | EMAIL_ADDRESS | 142 | 167 |
| fn_overlapping_prediction | ea0aa01e00d24b08962d253e50b0b336 | 58 | EMAIL_ADDRESS | 89 | 112 |
| fn_unmapped_ground_truth_label | bde4585a9dea42cdaf8a881ef3e69167 | 2 | UNMAPPED | 149 | 154 |
| fn_unmapped_ground_truth_label | bde4585a9dea42cdaf8a881ef3e69167 | 2 | UNMAPPED | 384 | 389 |
| fn_unmapped_prediction_label | 17276297f35343ea86cf6d98cc0d8cd8 | 30 | PERSON | 67 | 89 |
| fn_unmapped_prediction_label | 5977ed9892694dc08d9d37678b3fc168 | 183 | ID_CARD | 170 | 179 |
| fn_wrong_boundaries | 1dc91a0f6765487aa2bafcf06c243c74 | 95 | ORGANIZATION | 15 | 25 |
| fn_wrong_boundaries | 7c90b4cb2cdd443ea964ab3e16706fdd | 115 | DATE_TIME | 79 | 101 |
| fn_wrong_entity_type | eb3a14e152d540f6874267b13a673e80 | 173 | USERNAME | 131 | 143 |
| fn_wrong_entity_type | 019f4234839449eca75827c91e7d2c6f | 479 | SOCIAL_IDENTIFIER | 135 | 146 |
| fp_false_positive | 0e0b2612b96746d188056d4139a9f32a | 49 | DATE_TIME | 174 | 190 |
| fp_false_positive | 0e0b2612b96746d188056d4139a9f32a | 49 | DATE_TIME | 195 | 208 |
| fp_overlapping_prediction | 9179275a56154824a36257eb5181b88c | 47 | PERSON | 142 | 147 |
| fp_overlapping_prediction | 9179275a56154824a36257eb5181b88c | 47 | PERSON | 148 | 157 |
| fp_wrong_boundaries | 1dc91a0f6765487aa2bafcf06c243c74 | 95 | ORGANIZATION | 15 | 22 |
| fp_wrong_boundaries | 7c90b4cb2cdd443ea964ab3e16706fdd | 115 | DATE_TIME | 97 | 101 |
| fp_wrong_entity_type | 89fb5fd372d6472c874c9dd190532483 | 86 | ID_CARD | 93 | 97 |
| fp_wrong_entity_type | eb3a14e152d540f6874267b13a673e80 | 173 | PERSON | 131 | 143 |

## Structure

| Slice | Examples | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| format=structured | 49422 | 0.9380 | 0.7264 | 0.8188 |
| format=unstructured | 50578 | 0.9526 | 0.7243 | 0.8229 |
| chunked | 11510 | 0.9279 | 0.6750 | 0.7815 |
| single_window | 88490 | 0.9477 | 0.7364 | 0.8288 |

## Runtime

- Hardware: NVIDIA GeForce RTX 2050 (4.0 GiB, CUDA 12.8, driver 610.78); CPU 12th Gen Intel(R) Core(TM) i5-12450H (8 cores / 12 threads), 15.7 GiB RAM; Windows-11-10.0.26200-SP0, Python 3.14.7.
- Workers: one process, one model instance; batching inside `GLiNER.inference` (length-sorted batches).
- Model load time (sum over sessions): 25.9 s; inference time: 29,468.2 s; total runtime: 29,500.2 s; throughput 3.39 examples/s over 100,000 examples (112,182 model calls).

| Session | From | Examples | Chunks | Batch | Load s | Inference s | Ex/s | ms/example | GPU peak MiB | GPU util % | CPU % | RSS MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 1000 | 1102 | 1 | 15.28 | 323.282 | 3.09 | 323.28 | 2440 | 88.8 | 94.8 | 1979.9 |
| 2 | 1000 | 99000 | 111080 | 1 | 10.61 | 29144.955 | 3.4 | 294.39 | 3518 | 97.1 | 91.2 | 2015.6 |

## GLiNER label → gateway ontology mapping

GLiNER-PII's labels are the native Nemotron-PII labels, so each maps to the same gateway type the ground truth uses (`configs/pii_ontology.yaml`); the loader rejects any undocumented divergence. DATE/TIME targets are scored as `DATE_TIME` on both sides.

| GLiNER label | Gateway type | Basis | Gateway scoring | Predictions |
| --- | --- | --- | --- | ---: |
| `account_number` | BANK_ACCOUNT | same as ground-truth ontology | scored | 16675 |
| `age` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 8200 |
| `api_key` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 4678 |
| `bank_routing_number` | BANK_ACCOUNT | same as ground-truth ontology | scored | 8311 |
| `biometric_identifier` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 11298 |
| `blood_type` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 5642 |
| `certificate_license_number` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 3076 |
| `city` | LOCATION | same as ground-truth ontology | scored | 19113 |
| `company_name` | ORGANIZATION | same as ground-truth ontology | scored | 57041 |
| `coordinate` | LOCATION | same as ground-truth ontology | scored | 7693 |
| `country` | LOCATION | same as ground-truth ontology | scored | 24225 |
| `county` | LOCATION | same as ground-truth ontology | scored | 8109 |
| `credit_debit_card` | CREDIT_CARD | same as ground-truth ontology | scored | 12994 |
| `customer_id` | ID_CARD | same as ground-truth ontology | scored | 20551 |
| `cvv` | CREDIT_CARD | same as ground-truth ontology | scored | 5008 |
| `date` | DATE | same as ground-truth ontology | scored | 76126 |
| `date_of_birth` | DATE | same as ground-truth ontology | scored | 18294 |
| `date_time` | DATE | same as ground-truth ontology | scored | 12937 |
| `device_identifier` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 2491 |
| `education_level` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 9585 |
| `email` | EMAIL_ADDRESS | same as ground-truth ontology | scored | 41583 |
| `employee_id` | ID_CARD | same as ground-truth ontology | scored | 8724 |
| `employment_status` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 12011 |
| `fax_number` | PHONE_NUMBER | same as ground-truth ontology | scored | 6049 |
| `first_name` | PERSON | same as ground-truth ontology | scored | 93676 |
| `gender` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 8063 |
| `health_plan_beneficiary_number` | SOCIAL_IDENTIFIER | same as ground-truth ontology | scored | 10493 |
| `http_cookie` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 1729 |
| `ipv4` | IP_ADDRESS | same as ground-truth ontology | scored | 6097 |
| `ipv6` | IP_ADDRESS | same as ground-truth ontology | scored | 635 |
| `language` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 8398 |
| `last_name` | PERSON | same as ground-truth ontology | scored | 66091 |
| `license_plate` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 4126 |
| `mac_address` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 4233 |
| `medical_record_number` | SOCIAL_IDENTIFIER | same as ground-truth ontology | scored | 11042 |
| `national_id` | ID_CARD | same as ground-truth ontology | scored | 2917 |
| `occupation` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 66223 |
| `password` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 6897 |
| `phone_number` | PHONE_NUMBER | same as ground-truth ontology | scored | 23929 |
| `pin` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 6357 |
| `political_view` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 6791 |
| `postcode` | ADDRESS | same as ground-truth ontology | scored | 6402 |
| `race_ethnicity` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 9329 |
| `religious_belief` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 6184 |
| `sexuality` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 3374 |
| `ssn` | SOCIAL_IDENTIFIER | same as ground-truth ontology | scored | 6389 |
| `state` | LOCATION | same as ground-truth ontology | scored | 17667 |
| `street_address` | ADDRESS | same as ground-truth ontology | scored | 16670 |
| `swift_bic` | BANK_ACCOUNT | same as ground-truth ontology | scored | 5600 |
| `tax_id` | SOCIAL_IDENTIFIER | same as ground-truth ontology | scored | 1309 |
| `time` | TIME | same as ground-truth ontology | scored | 25414 |
| `unique_id` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 2494 |
| `url` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 20221 |
| `user_name` | USERNAME | same as ground-truth ontology | scored | 15469 |
| `vehicle_identifier` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 4663 |

## Limitations

- Nemotron-PII is GLiNER-PII's training source (train split). The test split is held out and shares no uid or text with train, but it is in-distribution for GLiNER and out-of-distribution for Presidio; do not read these numbers as performance on real gateway traffic.
- The threshold was selected on a train-derived subset (in-sample for the model).
- `UNMAPPED` ground truth counts against both detectors; the Coverage section separates that ontology limit from detection failure.
- Exact spans penalize boundary conventions (e.g. whether a title or trailing punctuation is included).

## Reproduce

```bash
python scripts/gliner_predict.py --data-dir /path/to/nemotron-normalized --source test --threshold 0.7 --batch-size 1 --dtype fp32 --output /path/to/nemotron-normalized/gliner/test.jsonl --resume
python scripts/evaluate_gliner.py --data-dir /path/to/nemotron-normalized --predictions /path/to/nemotron-normalized/gliner/test.jsonl --selection /path/to/nemotron-normalized/gliner/threshold_selection.json
```
