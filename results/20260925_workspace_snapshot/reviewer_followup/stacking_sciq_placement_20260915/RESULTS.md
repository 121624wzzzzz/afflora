# SciQ placement: prespecified confirmation

40 fresh runs, five new seeds, primary 998 unambiguous test questions. Six-comparison corrected intervals.

| Model | Arm | Accuracy % | Rotation average % | NLL |
|---|---|---:|---:|---:|
| qwen3_06b_chat | none | 87.6754 | 87.8607 | 0.329575 |
| qwen3_06b_chat | both | 87.7355 | 87.8758 | 0.330521 |
| qwen3_06b_chat | interior | 87.7956 | 87.9960 | 0.327294 |
| qwen3_06b_chat | hidden_budget | 87.7756 | 87.9309 | 0.328339 |
| qwen25_15b_chat | none | 92.6052 | 92.5852 | 0.197073 |
| qwen25_15b_chat | both | 92.7856 | 92.6052 | 0.198327 |
| qwen25_15b_chat | interior | 92.7455 | 92.5902 | 0.200551 |
| qwen25_15b_chat | hidden_budget | 92.7054 | 92.6603 | 0.196412 |

| Model | Contrast | Gain pp | Corrected 95% CI | Pass |
|---|---|---:|---|---|
| qwen3_06b_chat | both-minus-none | +0.0601 | [-0.5067, +0.6270] | False |
| qwen3_06b_chat | both-minus-hidden_budget | -0.0401 | [-0.9826, +0.9025] | False |
| qwen3_06b_chat | both-minus-interior | -0.0601 | [-1.0866, +0.9664] | False |
| qwen25_15b_chat | both-minus-none | +0.1804 | [-1.2811, +1.6418] | False |
| qwen25_15b_chat | both-minus-hidden_budget | +0.0802 | [-1.1243, +1.2846] | False |
| qwen25_15b_chat | both-minus-interior | +0.0401 | [-1.5617, +1.6419] | False |

Intervals across seeds condition on the fixed test set. Question bootstrap conditions on these fitted seeds. Secondary metrics cannot replace the primary decision. No open-ended generation claim.
