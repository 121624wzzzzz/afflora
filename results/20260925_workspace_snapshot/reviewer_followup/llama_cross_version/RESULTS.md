# Llama-3.1/3.2 A-LoRA cross-version results

Updated: 2026-07-11T20:23:40+08:00

Phase: `complete`

## Runtime status

| status | jobs |
| --- | ---: |
| complete | 12 |

## Per-run results

| config | seed | MATH clean | GSM8K |
| --- | ---: | ---: | ---: |
| llama31_8b_hidden_hr4 | 42 | 26.6867% | 74.8294% |
| llama31_8b_lmhead_ar16_s1_hr4 | 42 | 26.0661% | 74.4503% |
| llama31_8b_hidden_hr4 | 43 | 26.3463% | 74.9810% |
| llama31_8b_lmhead_ar16_s1_hr4 | 43 | 26.2462% | 74.3745% |
| llama31_8b_hidden_hr4 | 44 | 25.5055% | 75.6634% |
| llama31_8b_lmhead_ar16_s1_hr4 | 44 | 25.6857% | 73.9196% |
| llama32_3b_hidden_hr4 | 42 | 13.8739% | 57.0887% |
| llama32_3b_mergeable_ar16_s1_hr4 | 42 | 14.0340% | 56.4064% |
| llama32_3b_hidden_hr4 | 43 | 14.2342% | 55.2691% |
| llama32_3b_mergeable_ar16_s1_hr4 | 43 | 13.8539% | 56.0273% |
| llama32_3b_hidden_hr4 | 44 | 13.3534% | 56.5580% |
| llama32_3b_mergeable_ar16_s1_hr4 | 44 | 13.8939% | 55.3450% |

## Three-seed aggregates

| config | completed seeds | MATH mean ± sd | GSM8K mean ± sd |
| --- | ---: | ---: | ---: |
| llama31_8b_hidden_hr4 | 3/3 | 26.1795% ± 0.6080 | 75.1579% ± 0.4442 |
| llama31_8b_lmhead_ar16_s1_hr4 | 3/3 | 25.9993% ± 0.2862 | 74.2482% ± 0.2870 |
| llama32_3b_hidden_hr4 | 3/3 | 13.8205% ± 0.4429 | 56.3053% ± 0.9357 |
| llama32_3b_mergeable_ar16_s1_hr4 | 3/3 | 13.9273% ± 0.0946 | 55.9262% ± 0.5379 |
