# NVIDIA GLiNER-PII baseline: Nemotron-PII test split

Pretrained `nvidia/gliner-PII`, evaluated without any training or fine-tuning, on the same normalized test file, ground truth, ontology and strict matching as `presidio_baseline.md`. No source text, entity values, or model inputs are stored.

## Model

- Model: `nvidia/gliner-PII` at revision `bd23e8ef4425fd04e34c5204ab49ffaa706eae79` (gliner 0.2.29, torch 2.11.0+cu128, transformers 5.16.1).
- Encoder `microsoft/deberta-v3-large`, span mode `markerV0`, 445,463,040 parameters as counted by torch (the model card states 5.7 × 10^8); inference weights `fp32`.
- Input limit `max_len=384` words; longest predictable span `max_width=24` words; `whitespace` splitter.

## Dataset and methodology

- Dataset: `nvidia/Nemotron-PII`, normalized `test.jsonl`; this report covers the complete test split. SHA-256 `52c89435c888740d2be5bab6c0221761ae7b5b178f0c007342d14a2449ad99d4` (identical to the Presidio baseline).
- Matching: a prediction is a true positive only if entity type, start and end all equal a ground-truth entity (half-open offsets); multiset matching, so each ground-truth entity matches at most one prediction. No overlap or token credit.
- Scoring reuses the Presidio harness (`score_batch`): the `nervaluate` strict cross-check must reproduce the exact-match count on every 1,000-example batch.
- Ground truth: normalized Nemotron labels via `configs/pii_ontology.yaml`, with DATE and TIME scored as `DATE_TIME`, exactly as for Presidio. `UNMAPPED` ground truth stays in the denominator.
- Predictions: every GLiNER label maps through `configs/gliner_label_map.yaml` (table below). Predictions whose label maps to `UNMAPPED` are removed before gateway scoring and counted under Coverage; they are not silently discarded.

## Inference configuration

- Prompt: all 55 native Nemotron-PII labels in one pass (the model's training vocabulary).
- `threshold=0.7`, `flat_ner=True`, `multi_label=False`, batch size 1, device `cuda`.
- Texts over 384 words (the library would silently truncate them) are split into word windows overlapping by 48 words; each window keeps predictions starting in its own centre region, so no predictable entity is cut. Shorter texts are one unchanged call.
- Chunked examples: 11,510; model calls (chunks): 112,344.

## Threshold selection

Not recorded for this run.

## Overall metrics (gateway ontology)

| Precision | Recall | F1 | TP | FP | FN | Support | Predictions |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.9446 | 0.7278 | 0.8221 | 618862 | 36276 | 231478 | 850340 | 655138 |

Examples evaluated: 100,000; examples containing PII: 99,999; ground-truth entities: 850,340; predictions: 655,138; exact matches: 618,862 (nervaluate strict cross-check: 618,862).

## Per-entity metrics (gateway ontology)

| Type | Precision | Recall | F1 | TP | FP | FN | Support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ADDRESS | 0.9829 | 0.9814 | 0.9821 | 22727 | 396 | 431 | 23158 |
| BANK_ACCOUNT | 0.9864 | 0.9858 | 0.9861 | 30170 | 415 | 436 | 30606 |
| CREDIT_CARD | 0.9749 | 0.9909 | 0.9828 | 17548 | 452 | 162 | 17710 |
| DATE_TIME | 0.9296 | 0.9491 | 0.9392 | 123069 | 9320 | 6604 | 129673 |
| EMAIL_ADDRESS | 0.9967 | 0.7682 | 0.8676 | 41427 | 138 | 12503 | 53930 |
| ID_CARD | 0.9794 | 0.9779 | 0.9787 | 31515 | 663 | 712 | 32227 |
| IP_ADDRESS | 0.9922 | 0.9976 | 0.9949 | 9195 | 72 | 22 | 9217 |
| LOCATION | 0.9642 | 0.9712 | 0.9677 | 74065 | 2747 | 2199 | 76264 |
| ORGANIZATION | 0.9377 | 0.9744 | 0.9557 | 53431 | 3549 | 1406 | 54837 |
| PERSON | 0.8925 | 0.9916 | 0.9395 | 142436 | 17153 | 1203 | 143639 |
| PHONE_NUMBER | 0.9952 | 0.9828 | 0.9889 | 29831 | 145 | 522 | 30353 |
| SOCIAL_IDENTIFIER | 0.9790 | 0.9857 | 0.9823 | 28606 | 615 | 414 | 29020 |
| UNMAPPED | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 203906 | 203906 |
| USERNAME | 0.9605 | 0.9394 | 0.9498 | 14842 | 611 | 958 | 15800 |

## Coverage before ontology filtering

The gateway view above cannot credit a detection whose label the gateway ontology does not have. This section scores the model in its **native** label space (GLiNER label vs raw Nemotron label, same exact span rule) to separate detection failure from ontology mismatch. These numbers are not comparable to the gateway metrics and are not mixed into them.

- Prompt labels: 55; labels the model actually returned: 55; never returned: none.
- Predictions before ontology filtering: 890,370; removed by ontology filtering (label maps to UNMAPPED): 235,232, of which 189,138 exactly match a native ground-truth entity (correct detections the ontology drops).
- Gateway-mapped labels returned: 32; unmapped labels returned: 23.

Native-label exact match (all 55 labels):

| Precision | Recall | F1 | TP | FP | FN |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.9041 | 0.9467 | 0.9249 | 805027 | 85343 | 45313 |

`UNMAPPED` ground truth (203,906 entities) as seen by the model:

| Outcome | Entities | Share |
| --- | ---: | ---: |
| exact native match | 189138 | 92.8% |
| exact span other native label | 1311 | 0.6% |
| no prediction | 10530 | 5.2% |
| overlapping prediction | 2927 | 1.4% |

Per native label (gateway type in brackets):

| Native label | Gateway | Precision | Recall | F1 | TP | FP | FN | Support |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| account_number | BANK_ACCOUNT | 0.9834 | 0.9824 | 0.9829 | 16399 | 276 | 294 | 16693 |
| age | UNMAPPED | 0.7841 | 0.8963 | 0.8365 | 6424 | 1769 | 743 | 7167 |
| api_key | UNMAPPED | 0.9878 | 0.9886 | 0.9882 | 4614 | 57 | 53 | 4667 |
| bank_routing_number | BANK_ACCOUNT | 0.9955 | 0.9904 | 0.9930 | 8274 | 37 | 80 | 8354 |
| biometric_identifier | UNMAPPED | 0.9966 | 0.9892 | 0.9929 | 11256 | 38 | 123 | 11379 |
| blood_type | UNMAPPED | 0.9725 | 0.9903 | 0.9813 | 5485 | 155 | 54 | 5539 |
| certificate_license_number | UNMAPPED | 0.9466 | 0.9690 | 0.9577 | 2909 | 164 | 93 | 3002 |
| city | LOCATION | 0.9245 | 0.9621 | 0.9429 | 17651 | 1442 | 696 | 18347 |
| company_name | ORGANIZATION | 0.9377 | 0.9744 | 0.9557 | 53431 | 3549 | 1406 | 54837 |
| coordinate | LOCATION | 0.9829 | 0.9983 | 0.9905 | 7646 | 133 | 13 | 7659 |
| country | LOCATION | 0.9531 | 0.9823 | 0.9675 | 23060 | 1135 | 415 | 23475 |
| county | LOCATION | 0.9513 | 0.9108 | 0.9306 | 7710 | 395 | 755 | 8465 |
| credit_debit_card | CREDIT_CARD | 0.9827 | 0.9924 | 0.9875 | 12769 | 225 | 98 | 12867 |
| customer_id | ID_CARD | 0.9778 | 0.9799 | 0.9788 | 20089 | 457 | 413 | 20502 |
| cvv | CREDIT_CARD | 0.9515 | 0.9835 | 0.9672 | 4763 | 243 | 80 | 4843 |
| date | DATE | 0.9330 | 0.9570 | 0.9449 | 70691 | 5075 | 3176 | 73867 |
| date_of_birth | DATE | 0.9864 | 0.9978 | 0.9921 | 18039 | 248 | 40 | 18079 |
| date_time | DATE | 0.9770 | 0.9589 | 0.9679 | 12678 | 299 | 543 | 13221 |
| device_identifier | UNMAPPED | 0.9465 | 0.9378 | 0.9421 | 2351 | 133 | 156 | 2507 |
| education_level | UNMAPPED | 0.8642 | 0.8825 | 0.8732 | 8276 | 1301 | 1102 | 9378 |
| email | EMAIL_ADDRESS | 0.9967 | 0.7682 | 0.8676 | 41427 | 138 | 12503 | 53930 |
| employee_id | ID_CARD | 0.9904 | 0.9718 | 0.9810 | 8628 | 84 | 250 | 8878 |
| employment_status | UNMAPPED | 0.9068 | 0.9881 | 0.9457 | 10887 | 1119 | 131 | 11018 |
| fax_number | PHONE_NUMBER | 0.9425 | 0.8876 | 0.9142 | 5701 | 348 | 722 | 6423 |
| first_name | PERSON | 0.8892 | 0.9901 | 0.9369 | 83208 | 10367 | 835 | 84043 |
| gender | UNMAPPED | 0.9355 | 0.9860 | 0.9601 | 7538 | 520 | 107 | 7645 |
| health_plan_beneficiary_number | SOCIAL_IDENTIFIER | 0.9907 | 0.9841 | 0.9874 | 10390 | 98 | 168 | 10558 |
| http_cookie | UNMAPPED | 0.9650 | 0.8908 | 0.9264 | 4358 | 158 | 534 | 4892 |
| ipv4 | IP_ADDRESS | 0.9935 | 0.9988 | 0.9961 | 6071 | 40 | 7 | 6078 |
| ipv6 | IP_ADDRESS | 0.9848 | 0.9901 | 0.9875 | 3108 | 48 | 31 | 3139 |
| language | UNMAPPED | 0.9426 | 0.9147 | 0.9284 | 7913 | 482 | 738 | 8651 |
| last_name | PERSON | 0.8934 | 0.9896 | 0.9391 | 58978 | 7036 | 618 | 59596 |
| license_plate | UNMAPPED | 0.9813 | 0.9929 | 0.9871 | 4049 | 77 | 29 | 4078 |
| mac_address | UNMAPPED | 0.9983 | 0.9951 | 0.9967 | 4229 | 7 | 21 | 4250 |
| medical_record_number | SOCIAL_IDENTIFIER | 0.9936 | 0.9880 | 0.9908 | 10965 | 71 | 133 | 11098 |
| national_id | ID_CARD | 0.9462 | 0.9705 | 0.9582 | 2763 | 157 | 84 | 2847 |
| occupation | UNMAPPED | 0.4470 | 0.8009 | 0.5738 | 29543 | 36547 | 7345 | 36888 |
| password | UNMAPPED | 0.9810 | 0.9686 | 0.9748 | 6766 | 131 | 219 | 6985 |
| phone_number | PHONE_NUMBER | 0.9701 | 0.9700 | 0.9700 | 23211 | 716 | 719 | 23930 |
| pin | UNMAPPED | 0.9657 | 0.9507 | 0.9582 | 6138 | 218 | 318 | 6456 |
| political_view | UNMAPPED | 0.8773 | 0.9306 | 0.9032 | 5958 | 833 | 444 | 6402 |
| postcode | ADDRESS | 0.9684 | 0.9871 | 0.9777 | 6198 | 202 | 81 | 6279 |
| race_ethnicity | UNMAPPED | 0.9469 | 0.9546 | 0.9507 | 8828 | 495 | 420 | 9248 |
| religious_belief | UNMAPPED | 0.9500 | 0.9388 | 0.9444 | 5874 | 309 | 383 | 6257 |
| sexuality | UNMAPPED | 0.9502 | 0.9680 | 0.9590 | 3206 | 168 | 106 | 3312 |
| ssn | SOCIAL_IDENTIFIER | 0.9296 | 0.9797 | 0.9540 | 5939 | 450 | 123 | 6062 |
| state | LOCATION | 0.9588 | 0.9233 | 0.9407 | 16913 | 727 | 1405 | 18318 |
| street_address | ADDRESS | 0.9882 | 0.9790 | 0.9836 | 16525 | 198 | 354 | 16879 |
| swift_bic | BANK_ACCOUNT | 0.9746 | 0.9817 | 0.9781 | 5457 | 142 | 102 | 5559 |
| tax_id | SOCIAL_IDENTIFIER | 0.9824 | 0.9869 | 0.9847 | 1285 | 23 | 17 | 1302 |
| time | TIME | 0.8313 | 0.8602 | 0.8455 | 21080 | 4279 | 3426 | 24506 |
| unique_id | UNMAPPED | 0.6704 | 0.9398 | 0.7826 | 1670 | 821 | 107 | 1777 |
| url | UNMAPPED | 0.9876 | 0.9602 | 0.9737 | 36339 | 457 | 1508 | 37847 |
| user_name | USERNAME | 0.9605 | 0.9394 | 0.9498 | 14842 | 611 | 958 | 15800 |
| vehicle_identifier | UNMAPPED | 0.9710 | 0.9925 | 0.9817 | 4527 | 135 | 34 | 4561 |

## Error analysis (gateway ontology)

Each unmatched ground-truth entity (false negative) and each unmatched prediction (false positive) gets exactly one category, checked in the order listed.

| False-negative category | Count | Share | Meaning |
| --- | ---: | ---: | --- |
| unmapped_ground_truth_label | 203906 | 88.1% | ground-truth label is UNMAPPED in the gateway ontology (never matchable in this view; see Coverage) |
| wrong_entity_type | 927 | 0.4% | a scored prediction has the exact span but a different type |
| wrong_boundaries | 2846 | 1.2% | a scored prediction of the same type overlaps with different offsets |
| overlapping_prediction | 13922 | 6.0% | only overlapped by scored predictions of other types and offsets |
| unmapped_prediction_label | 977 | 0.4% | only overlapped by predictions removed by ontology filtering (model label maps to UNMAPPED) |
| missed_entity | 8900 | 3.8% | no prediction of any label overlaps it |

| False-positive category | Count | Share | Meaning |
| --- | ---: | ---: | --- |
| wrong_entity_type | 1361 | 3.8% | exact span of a ground-truth entity of another type (incl. UNMAPPED) |
| wrong_boundaries | 2568 | 7.1% | overlaps a ground-truth entity of the same type, different offsets |
| overlapping_prediction | 19147 | 52.8% | overlaps only ground-truth entities of other types and offsets |
| false_positive | 13200 | 36.4% | overlaps no ground-truth entity |

- Overlapping false positives that are nested in / contain a ground-truth span: 23,067.
- Prediction pairs overlapping each other: 1 (flat-NER decoding forbids overlap within one model call, so only chunk-window seams could produce any). Duplicate predictions removed: 0.
- Structurally unreachable ground truth (no whole-word span of at most `max_width` words can equal it): misaligned_word_boundary 3,514, wider_than_max_width 1,165; of these still false negatives: misaligned_word_boundary 3,514, wider_than_max_width 1,165.

Structural reachability by native label (labels with any unreachable ground truth; native exact-match misses in brackets):

| Native label | Reachable (missed) | Wider than max_width (missed) | Misaligned boundary (missed) |
| --- | ---: | ---: | ---: |
| `account_number` | 16,690 (291) | 0 (0) | 3 (3) |
| `age` | 6,536 (112) | 0 (0) | 631 (631) |
| `biometric_identifier` | 11,378 (122) | 0 (0) | 1 (1) |
| `certificate_license_number` | 3,001 (92) | 0 (0) | 1 (1) |
| `city` | 18,324 (673) | 0 (0) | 23 (23) |
| `company_name` | 54,831 (1,400) | 0 (0) | 6 (6) |
| `country` | 23,316 (256) | 0 (0) | 159 (159) |
| `county` | 8,357 (647) | 0 (0) | 108 (108) |
| `customer_id` | 20,467 (378) | 0 (0) | 35 (35) |
| `cvv` | 4,841 (78) | 0 (0) | 2 (2) |
| `date` | 73,027 (2,336) | 0 (0) | 840 (840) |
| `date_of_birth` | 18,078 (39) | 0 (0) | 1 (1) |
| `date_time` | 13,210 (532) | 0 (0) | 11 (11) |
| `device_identifier` | 2,505 (154) | 0 (0) | 2 (2) |
| `education_level` | 9,292 (1,016) | 0 (0) | 86 (86) |
| `email` | 53,929 (12,502) | 0 (0) | 1 (1) |
| `employee_id` | 8,876 (248) | 0 (0) | 2 (2) |
| `employment_status` | 10,941 (54) | 0 (0) | 77 (77) |
| `fax_number` | 6,422 (721) | 0 (0) | 1 (1) |
| `first_name` | 84,009 (801) | 0 (0) | 34 (34) |
| `gender` | 7,639 (101) | 0 (0) | 6 (6) |
| `http_cookie` | 4,555 (197) | 328 (328) | 9 (9) |
| `ipv6` | 3,137 (29) | 2 (2) | 0 (0) |
| `language` | 8,392 (479) | 0 (0) | 259 (259) |
| `last_name` | 59,553 (575) | 0 (0) | 43 (43) |
| `national_id` | 2,846 (83) | 0 (0) | 1 (1) |
| `occupation` | 36,064 (6,521) | 0 (0) | 824 (824) |
| `password` | 6,983 (217) | 0 (0) | 2 (2) |
| `pin` | 6,453 (315) | 0 (0) | 3 (3) |
| `political_view` | 6,332 (374) | 0 (0) | 70 (70) |
| `postcode` | 6,274 (76) | 0 (0) | 5 (5) |
| `race_ethnicity` | 9,206 (378) | 0 (0) | 42 (42) |
| `religious_belief` | 6,234 (360) | 0 (0) | 23 (23) |
| `sexuality` | 3,233 (27) | 0 (0) | 79 (79) |
| `ssn` | 6,060 (121) | 0 (0) | 2 (2) |
| `state` | 18,272 (1,359) | 0 (0) | 46 (46) |
| `street_address` | 16,851 (326) | 0 (0) | 28 (28) |
| `time` | 24,458 (3,378) | 0 (0) | 48 (48) |
| `url` | 37,012 (673) | 835 (835) | 0 (0) |

Native labels predicted over unmatched gateway entities (top 5 per type; shows what took the span instead, e.g. name parts inside an email address):

| Type | Overlapping native labels (count) |
| --- | --- |
| ADDRESS | `street_address` 145, `city` 80, `postcode` 44, `state` 24, `last_name` 21 |
| BANK_ACCOUNT | `unique_id` 106, `customer_id` 78, `credit_debit_card` 21, `health_plan_beneficiary_number` 12, `user_name` 7 |
| CREDIT_CARD | `account_number` 20, `pin` 20, `phone_number` 8, `bank_routing_number` 7, `password` 3 |
| DATE_TIME | `date` 714, `time` 546, `date_time` 195, `age` 18, `last_name` 13 |
| EMAIL_ADDRESS | `first_name` 9,079, `last_name` 6,039, `company_name` 974, `user_name` 331, `email` 30 |
| ID_CARD | `unique_id` 241, `account_number` 67, `user_name` 32, `certificate_license_number` 15, `medical_record_number` 12 |
| IP_ADDRESS | `ipv6` 4, `customer_id` 1, `political_view` 1, `ipv4` 1 |
| LOCATION | `city` 322, `county` 202, `state` 153, `company_name` 109, `country` 95 |
| ORGANIZATION | `country` 527, `company_name` 242, `city` 49, `last_name` 18, `language` 18 |
| PERSON | `last_name` 204, `first_name` 152, `user_name` 112, `occupation` 76, `company_name` 56 |
| PHONE_NUMBER | `pin` 28, `account_number` 12, `medical_record_number` 9, `customer_id` 7, `credit_debit_card` 5 |
| SOCIAL_IDENTIFIER | `unique_id` 68, `customer_id` 53, `national_id` 31, `account_number` 17, `credit_debit_card` 15 |
| USERNAME | `first_name` 726, `last_name` 577, `email` 29, `user_name` 26, `company_name` 12 |

False-negative categories by gateway type:

| Type | unmapped_ground_truth_label | wrong_entity_type | wrong_boundaries | overlapping_prediction | unmapped_prediction_label | missed_entity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ADDRESS | 0 | 37 | 154 | 47 | 14 | 179 |
| BANK_ACCOUNT | 0 | 130 | 3 | 6 | 133 | 164 |
| CREDIT_CARD | 0 | 43 | 3 | 1 | 29 | 86 |
| DATE_TIME | 0 | 6 | 1245 | 41 | 35 | 5277 |
| EMAIL_ADDRESS | 0 | 29 | 30 | 12230 | 9 | 205 |
| ID_CARD | 0 | 145 | 10 | 14 | 291 | 252 |
| IP_ADDRESS | 0 | 1 | 5 | 0 | 1 | 15 |
| LOCATION | 0 | 11 | 776 | 136 | 159 | 1117 |
| ORGANIZATION | 0 | 28 | 242 | 594 | 43 | 499 |
| PERSON | 0 | 128 | 350 | 112 | 127 | 486 |
| PHONE_NUMBER | 0 | 48 | 2 | 4 | 40 | 428 |
| SOCIAL_IDENTIFIER | 0 | 141 | 1 | 13 | 89 | 170 |
| UNMAPPED | 203906 | 0 | 0 | 0 | 0 | 0 |
| USERNAME | 0 | 180 | 25 | 724 | 7 | 22 |

False-positive categories by gateway type:

| Type | wrong_entity_type | wrong_boundaries | overlapping_prediction | false_positive |
| --- | ---: | ---: | ---: | ---: |
| ADDRESS | 71 | 182 | 8 | 135 |
| BANK_ACCOUNT | 160 | 3 | 1 | 251 |
| CREDIT_CARD | 65 | 3 | 4 | 380 |
| DATE_TIME | 27 | 1158 | 312 | 7823 |
| EMAIL_ADDRESS | 9 | 30 | 49 | 50 |
| ID_CARD | 369 | 8 | 15 | 271 |
| IP_ADDRESS | 4 | 5 | 0 | 63 |
| LOCATION | 149 | 763 | 950 | 885 |
| ORGANIZATION | 43 | 224 | 1164 | 2118 |
| PERSON | 179 | 171 | 16304 | 499 |
| PHONE_NUMBER | 22 | 2 | 1 | 120 |
| SOCIAL_IDENTIFIER | 113 | 1 | 3 | 498 |
| USERNAME | 150 | 18 | 336 | 107 |

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
| format=structured | 49422 | 0.9382 | 0.7283 | 0.8200 |
| format=unstructured | 50578 | 0.9529 | 0.7271 | 0.8249 |
| chunked | 11510 | 0.9283 | 0.6777 | 0.7835 |
| single_window | 88490 | 0.9479 | 0.7386 | 0.8303 |

## Runtime

- Hardware: Tesla T4 (14.56 GiB, CUDA 12.8, driver 580.82.07); CPU x86_64 (1 cores / 2 threads), 12.7 GiB RAM; Linux-6.6.122+-x86_64-with-glibc2.39, Python 3.13.15.
- Workers: one process, one model instance; batching inside `GLiNER.inference` (length-sorted batches).
- Model load time (sum over sessions): 32.1 s; inference time: 21,732.3 s; total runtime: 21,778.5 s; throughput 4.47 examples/s over 97,184 examples (109,192 model calls).

| Session | From | Examples | Chunks | Batch | Load s | Inference s | Ex/s | ms/example | GPU peak MiB | GPU util % | CPU % | RSS MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2816 | 19712 | 22089 | 1 | 14.81 | 3942.781 | 5.0 | 200.02 | 2870 | 91.2 | 98.8 | 2309.2 |
| 2 | 22528 | 77472 | 87103 | 1 | 17.27 | 17789.473 | 4.35 | 229.62 | 4768 | 92.6 | 98.9 | 2319.0 |

## GLiNER label → gateway ontology mapping

GLiNER-PII's labels are the native Nemotron-PII labels, so each maps to the same gateway type the ground truth uses (`configs/pii_ontology.yaml`); the loader rejects any undocumented divergence. DATE/TIME targets are scored as `DATE_TIME` on both sides.

| GLiNER label | Gateway type | Basis | Gateway scoring | Predictions |
| --- | --- | --- | --- | ---: |
| `account_number` | BANK_ACCOUNT | same as ground-truth ontology | scored | 16675 |
| `age` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 8193 |
| `api_key` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 4671 |
| `bank_routing_number` | BANK_ACCOUNT | same as ground-truth ontology | scored | 8311 |
| `biometric_identifier` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 11294 |
| `blood_type` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 5640 |
| `certificate_license_number` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 3073 |
| `city` | LOCATION | same as ground-truth ontology | scored | 19093 |
| `company_name` | ORGANIZATION | same as ground-truth ontology | scored | 56980 |
| `coordinate` | LOCATION | same as ground-truth ontology | scored | 7779 |
| `country` | LOCATION | same as ground-truth ontology | scored | 24195 |
| `county` | LOCATION | same as ground-truth ontology | scored | 8105 |
| `credit_debit_card` | CREDIT_CARD | same as ground-truth ontology | scored | 12994 |
| `customer_id` | ID_CARD | same as ground-truth ontology | scored | 20546 |
| `cvv` | CREDIT_CARD | same as ground-truth ontology | scored | 5006 |
| `date` | DATE | same as ground-truth ontology | scored | 75766 |
| `date_of_birth` | DATE | same as ground-truth ontology | scored | 18287 |
| `date_time` | DATE | same as ground-truth ontology | scored | 12977 |
| `device_identifier` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 2484 |
| `education_level` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 9577 |
| `email` | EMAIL_ADDRESS | same as ground-truth ontology | scored | 41565 |
| `employee_id` | ID_CARD | same as ground-truth ontology | scored | 8712 |
| `employment_status` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 12006 |
| `fax_number` | PHONE_NUMBER | same as ground-truth ontology | scored | 6049 |
| `first_name` | PERSON | same as ground-truth ontology | scored | 93575 |
| `gender` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 8058 |
| `health_plan_beneficiary_number` | SOCIAL_IDENTIFIER | same as ground-truth ontology | scored | 10488 |
| `http_cookie` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 4516 |
| `ipv4` | IP_ADDRESS | same as ground-truth ontology | scored | 6111 |
| `ipv6` | IP_ADDRESS | same as ground-truth ontology | scored | 3156 |
| `language` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 8395 |
| `last_name` | PERSON | same as ground-truth ontology | scored | 66014 |
| `license_plate` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 4126 |
| `mac_address` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 4236 |
| `medical_record_number` | SOCIAL_IDENTIFIER | same as ground-truth ontology | scored | 11036 |
| `national_id` | ID_CARD | same as ground-truth ontology | scored | 2920 |
| `occupation` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 66090 |
| `password` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 6897 |
| `phone_number` | PHONE_NUMBER | same as ground-truth ontology | scored | 23927 |
| `pin` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 6356 |
| `political_view` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 6791 |
| `postcode` | ADDRESS | same as ground-truth ontology | scored | 6400 |
| `race_ethnicity` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 9323 |
| `religious_belief` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 6183 |
| `sexuality` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 3374 |
| `ssn` | SOCIAL_IDENTIFIER | same as ground-truth ontology | scored | 6389 |
| `state` | LOCATION | same as ground-truth ontology | scored | 17640 |
| `street_address` | ADDRESS | same as ground-truth ontology | scored | 16723 |
| `swift_bic` | BANK_ACCOUNT | same as ground-truth ontology | scored | 5599 |
| `tax_id` | SOCIAL_IDENTIFIER | same as ground-truth ontology | scored | 1308 |
| `time` | TIME | same as ground-truth ontology | scored | 25359 |
| `unique_id` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 2491 |
| `url` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 36796 |
| `user_name` | USERNAME | same as ground-truth ontology | scored | 15453 |
| `vehicle_identifier` | UNMAPPED | same as ground-truth ontology | filtered (not scored) | 4662 |

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
