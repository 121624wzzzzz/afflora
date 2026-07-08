# Qwen2.5-0.5B hidden LoRA rank analysis

Updated: 2026-07-07T12:53:10+08:00

This is the first stage before checking mergeable AffLoRA on top of hidden LoRA.
MATH reports the clean-4,995 score.

## Aggregate accuracy

| hidden rank | completed | MATH mean ± sd | GSM8K mean ± sd |
| ---: | ---: | ---: | ---: |
| 1 | 3/3 | 21.3013% ± 0.3308 | 46.9548% ± 0.3891 |
| 2 | 3/3 | 21.1745% ± 0.4114 | 46.4241% ± 0.2663 |
| 4 | 3/3 | 21.0744% ± 0.3715 | 46.6262% ± 0.3474 |
| 8 | 3/3 | 21.3480% ± 0.0758 | 47.0811% ± 0.6949 |
| 16 | 3/3 | 20.9743% ± 0.3304 | 47.2075% ± 0.3816 |

## Ranking

| rank by MATH | hidden rank | MATH mean | GSM8K mean | mean(MATH,GSM8K) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 21.3480% | 47.0811% | 34.2146% |
| 2 | 1 | 21.3013% | 46.9548% | 34.1280% |
| 3 | 2 | 21.1745% | 46.4241% | 33.7993% |
| 4 | 4 | 21.0744% | 46.6262% | 33.8503% |
| 5 | 16 | 20.9743% | 47.2075% | 34.0909% |

## Suggested second-stage hidden ranks

Use these hidden ranks for the mergeable AffLoRA follow-up unless manual inspection overrides:

- hidden hr8
- hidden hr16

Rationale:

- best MATH: hr8
- best GSM8K: hr16
- best average of MATH and GSM8K: hr8
