# Small-model mergeable AffLoRA math sweep results

Updated: 2026-07-11T15:50:17+08:00

Phase: `complete`

## Runtime status

| status | jobs |
| --- | ---: |
| complete | 9 |

## Per-run results

| config | seed | MATH clean | GSM8K |
| --- | ---: | ---: | ---: |
| qwen25_15b_hidden_hr4 | 45 | 36.7568% | 69.6740% |
| qwen25_15b_mergeable_ar8_s1_hr4 | 45 | 36.0761% | 69.7498% |
| qwen25_15b_mergeable_ar16_s1_hr4 | 45 | 36.0160% | 70.2047% |
| qwen25_15b_hidden_hr4 | 46 | 36.0961% | 70.3563% |
| qwen25_15b_mergeable_ar8_s1_hr4 | 46 | 36.2963% | 69.9773% |
| qwen25_15b_mergeable_ar16_s1_hr4 | 46 | 36.3163% | 70.9629% |
| qwen25_15b_hidden_hr4 | 47 | 35.8759% | 69.7498% |
| qwen25_15b_mergeable_ar8_s1_hr4 | 47 | 36.4965% | 70.1289% |
| qwen25_15b_mergeable_ar16_s1_hr4 | 47 | 36.4164% | 70.4321% |

## Three-seed aggregates

| config | completed seeds | MATH mean ± sd | GSM8K mean ± sd |
| --- | ---: | ---: | ---: |
| qwen25_15b_hidden_hr4 | 3/3 | 36.2429% ± 0.4584 | 69.9267% ± 0.3740 |
| qwen25_15b_mergeable_ar8_s1_hr4 | 3/3 | 36.2896% ± 0.2103 | 69.9520% ± 0.1908 |
| qwen25_15b_mergeable_ar16_s1_hr4 | 3/3 | 36.2496% ± 0.2084 | 70.5332% ± 0.3891 |
