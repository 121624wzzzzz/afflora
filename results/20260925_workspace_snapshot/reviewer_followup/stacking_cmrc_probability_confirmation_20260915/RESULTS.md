# CMRC five-new-seed confirmation

Completed 30 fresh training runs. Joint two-model confirmation: **True**.

Primary: mean unique complete-reference probability (percentage-point differences below). Fixed Bonferroni family=4; paired t df=4.

| Model | Control | Mean Δ pp | Corrected CI pp | Seed deltas pp | Confirmed |
|---|---|---:|---|---|---|
| qwen3_06b_chat | none | +0.326672 | [+0.006713, +0.646631] | +0.282085, +0.098582, +0.290008, +0.423120, +0.539566 | True |
| qwen3_06b_chat | hidden_budget | +0.319153 | [+0.008724, +0.629581] | +0.253121, +0.100230, +0.319829, +0.387693, +0.534891 | True |
| qwen25_15b_chat | none | +0.514119 | [+0.011013, +1.017224] | +0.133189, +0.813711, +0.693523, +0.459079, +0.471092 | True |
| qwen25_15b_chat | hidden_budget | +0.554035 | [+0.219164, +0.888906] | +0.262592, +0.703070, +0.639928, +0.628923, +0.535662 | True |

All endpoint scores and all supporting contrasts are retained in RESULTS.json and paired_effects.csv.

- Same previously examined public dev; independent new training seeds, not a new unseen test set.
- Only primary probability intervals adjust family=4. All supporting intervals are descriptive.
- FP32 inference differs from old BF16 inference; old and new scores are not pooled.
- Reference probability covers given unique strings and im_end, not all correct paraphrases or calibration.
- Five seed pairs imply t-distribution assumptions; cluster intervals condition on fitted models.
