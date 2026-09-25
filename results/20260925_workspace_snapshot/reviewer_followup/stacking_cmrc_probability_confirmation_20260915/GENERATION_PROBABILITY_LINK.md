# Exploratory probability / generation decomposition

This analysis does not change the fixed four primary tests. Categories pool five seeds descriptively; no independence assumption or p-values.

| Model | Control | Category | Pairs | Fraction | Mean probability Δ pp | Contribution to overall Δ pp |
|---|---|---|---:|---:|---:|---:|
| qwen3_06b_chat | none | both_correct | 10127 | 62.92% | +0.440016 | +0.276859 |
| qwen3_06b_chat | none | both_wrong | 5544 | 34.45% | +0.101869 | +0.035089 |
| qwen3_06b_chat | none | corrected | 229 | 1.42% | +4.246116 | +0.060414 |
| qwen3_06b_chat | none | regressed | 195 | 1.21% | -3.771173 | -0.045690 |
| qwen3_06b_chat | hidden_budget | both_correct | 10127 | 62.92% | +0.423686 | +0.266584 |
| qwen3_06b_chat | hidden_budget | both_wrong | 5550 | 34.48% | +0.111111 | +0.038314 |
| qwen3_06b_chat | hidden_budget | corrected | 229 | 1.42% | +4.250328 | +0.060474 |
| qwen3_06b_chat | hidden_budget | regressed | 189 | 1.17% | -3.935966 | -0.046219 |
| qwen25_15b_chat | none | both_correct | 10494 | 65.20% | +0.790231 | +0.515234 |
| qwen25_15b_chat | none | both_wrong | 5104 | 31.71% | -0.061908 | -0.019632 |
| qwen25_15b_chat | none | corrected | 228 | 1.42% | +6.481735 | +0.091820 |
| qwen25_15b_chat | none | regressed | 269 | 1.67% | -4.385898 | -0.073303 |
| qwen25_15b_chat | hidden_budget | both_correct | 10497 | 65.22% | +0.828086 | +0.540070 |
| qwen25_15b_chat | hidden_budget | both_wrong | 5106 | 31.72% | -0.038702 | -0.012278 |
| qwen25_15b_chat | hidden_budget | corrected | 225 | 1.40% | +7.026179 | +0.098222 |
| qwen25_15b_chat | hidden_budget | regressed | 267 | 1.66% | -4.338948 | -0.071979 |
