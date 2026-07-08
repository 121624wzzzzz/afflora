# Small-model mergeable AffLoRA math sweep results

Updated: 2026-07-06T17:30:34+08:00

Phase: `complete`

## Runtime status

| status | jobs |
| --- | ---: |
| complete | 36 |

## Per-run results

| config | seed | MATH clean | GSM8K |
| --- | ---: | ---: | ---: |
| qwen25_05b_hidden_hr4 | 42 | 21.4414% | 47.4602% |
| qwen25_05b_hidden_hr4 | 43 | 20.3604% | 46.6262% |
| qwen25_05b_hidden_hr4 | 44 | 21.0811% | 47.3086% |
| qwen25_05b_mergeable_ar1_s8_hr4 | 42 | 21.1211% | 48.8249% |
| qwen25_05b_mergeable_ar1_s8_hr4 | 43 | 21.0010% | 48.1425% |
| qwen25_05b_mergeable_ar1_s8_hr4 | 44 | 21.0210% | 46.4746% |
| qwen25_05b_mergeable_ar2_s8_hr4 | 42 | 21.4615% | 47.0053% |
| qwen25_05b_mergeable_ar2_s8_hr4 | 43 | 20.6406% | 46.9295% |
| qwen25_05b_mergeable_ar2_s8_hr4 | 44 | 21.0811% | 46.5504% |
| qwen25_05b_mergeable_ar4_s8_hr4 | 42 | 21.5415% | 48.3700% |
| qwen25_05b_mergeable_ar4_s8_hr4 | 43 | 20.7207% | 47.6118% |
| qwen25_05b_mergeable_ar4_s8_hr4 | 44 | 20.9009% | 46.0955% |
| qwen25_05b_mergeable_ar8_s8_hr4 | 42 | 21.3013% | 48.4458% |
| qwen25_05b_mergeable_ar8_s8_hr4 | 43 | 20.3203% | 47.0811% |
| qwen25_05b_mergeable_ar8_s8_hr4 | 44 | 20.7207% | 46.6262% |
| qwen25_05b_mergeable_ar16_s8_hr4 | 42 | 21.2613% | 47.6118% |
| qwen25_05b_mergeable_ar16_s8_hr4 | 43 | 20.8008% | 48.6732% |
| qwen25_05b_mergeable_ar16_s8_hr4 | 44 | 20.6206% | 45.6406% |
| qwen25_15b_hidden_hr4 | 42 | 35.5956% | 69.1433% |
| qwen25_15b_hidden_hr4 | 43 | 36.0561% | 70.3563% |
| qwen25_15b_hidden_hr4 | 44 | 36.1161% | 70.5838% |
| qwen25_15b_mergeable_ar1_s1_hr4 | 42 | 35.6557% | 69.3707% |
| qwen25_15b_mergeable_ar1_s1_hr4 | 43 | 35.7958% | 70.5080% |
| qwen25_15b_mergeable_ar1_s1_hr4 | 44 | 35.9760% | 70.1289% |
| qwen25_15b_mergeable_ar2_s1_hr4 | 42 | 35.7357% | 69.5982% |
| qwen25_15b_mergeable_ar2_s1_hr4 | 43 | 36.1762% | 69.9014% |
| qwen25_15b_mergeable_ar2_s1_hr4 | 44 | 36.5766% | 70.7354% |
| qwen25_15b_mergeable_ar4_s1_hr4 | 42 | 35.9960% | 69.4466% |
| qwen25_15b_mergeable_ar4_s1_hr4 | 43 | 36.0961% | 69.6740% |
| qwen25_15b_mergeable_ar4_s1_hr4 | 44 | 36.2563% | 70.0531% |
| qwen25_15b_mergeable_ar8_s1_hr4 | 42 | 35.7958% | 69.9014% |
| qwen25_15b_mergeable_ar8_s1_hr4 | 43 | 36.5165% | 70.6596% |
| qwen25_15b_mergeable_ar8_s1_hr4 | 44 | 36.2963% | 70.7354% |
| qwen25_15b_mergeable_ar16_s1_hr4 | 42 | 36.0961% | 69.5224% |
| qwen25_15b_mergeable_ar16_s1_hr4 | 43 | 36.3564% | 70.4321% |
| qwen25_15b_mergeable_ar16_s1_hr4 | 44 | 36.6366% | 70.2805% |

## Three-seed aggregates

| config | completed seeds | MATH mean ± sd | GSM8K mean ± sd |
| --- | ---: | ---: | ---: |
| qwen25_05b_hidden_hr4 | 3/3 | 20.9610% ± 0.5505 | 47.1317% ± 0.4442 |
| qwen25_05b_mergeable_ar1_s8_hr4 | 3/3 | 21.0477% ± 0.0644 | 47.8140% ± 1.2091 |
| qwen25_05b_mergeable_ar2_s8_hr4 | 3/3 | 21.0611% ± 0.4108 | 46.8284% ± 0.2437 |
| qwen25_05b_mergeable_ar4_s8_hr4 | 3/3 | 21.0544% ± 0.4314 | 47.3591% ± 1.1581 |
| qwen25_05b_mergeable_ar8_s8_hr4 | 3/3 | 20.7808% ± 0.4932 | 47.3844% ± 0.9469 |
| qwen25_05b_mergeable_ar16_s8_hr4 | 3/3 | 20.8942% ± 0.3304 | 47.3086% ± 1.5389 |
| qwen25_15b_hidden_hr4 | 3/3 | 35.9226% ± 0.2848 | 70.0278% ± 0.7744 |
| qwen25_15b_mergeable_ar1_s1_hr4 | 3/3 | 35.8091% ± 0.1606 | 70.0025% ± 0.5790 |
| qwen25_15b_mergeable_ar2_s1_hr4 | 3/3 | 36.1628% ± 0.4206 | 70.0783% ± 0.5889 |
| qwen25_15b_mergeable_ar4_s1_hr4 | 3/3 | 36.1161% ± 0.1313 | 69.7245% ± 0.3064 |
| qwen25_15b_mergeable_ar8_s1_hr4 | 3/3 | 36.2029% ± 0.3693 | 70.4321% ± 0.4612 |
| qwen25_15b_mergeable_ar16_s1_hr4 | 3/3 | 36.3630% ± 0.2703 | 70.0783% ± 0.4874 |
