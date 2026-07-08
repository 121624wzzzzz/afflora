# Statistical analysis

Updated: 2026-07-03T14:03:34+08:00

MATH reports the clean-4,995 score. Intervals below are paired across the three initialization/training seeds; with n=3 they are intentionally conservative.

## Aggregate accuracy

| config | completed | MATH mean ± sd | GSM8K mean ± sd |
| --- | ---: | ---: | ---: |
| hidden_hr4 | 3/3 | 50.1768% ± 0.4134 | 86.2269% ± 0.2189 |
| lmhead_ar1_s1_hr4 | 3/3 | 49.9433% ± 0.0644 | 86.0248% ± 0.2663 |
| lmhead_ar1_s2_hr4 | 3/3 | 50.0234% ± 0.1806 | 85.9237% ± 0.5790 |
| lmhead_ar1_s4_hr4 | 3/3 | 50.1301% ± 0.1602 | 85.8479% ± 0.3740 |
| lmhead_ar1_s8_hr4 | 3/3 | 49.9766% ± 0.4109 | 85.8226% ± 0.0758 |
| lmhead_ar1_s16_hr4 | 3/3 | 49.8765% ± 0.4105 | 85.7215% ± 0.3064 |
| hidden_hr8 | 1/3 | 49.8899% | 85.4435% |
| lmhead_ar1_s8_hr8 | 0/3 | - | - |

## Paired improvement over matching hidden-LoRA baseline

| treatment | paired seeds | Δ MATH | MATH positive seeds | Δ GSM8K | GSM8K positive seeds |
| --- | ---: | ---: | ---: | ---: | ---: |
| lmhead_ar1_s1_hr4 | 3/3 | -0.2336 ± 0.3873 pp; 95% CI [-1.1958, +0.7286] | 0/3 | -0.2022 ± 0.4632 pp; 95% CI [-1.3529, +0.9486] | 1/3 |
| lmhead_ar1_s2_hr4 | 3/3 | -0.1535 ± 0.2503 pp; 95% CI [-0.7753, +0.4683] | 1/3 | -0.3033 ± 0.3791 pp; 95% CI [-1.2449, +0.6384] | 1/3 |
| lmhead_ar1_s4_hr4 | 3/3 | -0.0467 ± 0.3012 pp; 95% CI [-0.7949, +0.7015] | 1/3 | -0.3791 ± 0.3033 pp; 95% CI [-1.1324, +0.3743] | 0/3 |
| lmhead_ar1_s8_hr4 | 3/3 | -0.2002 ± 0.0722 pp; 95% CI [-0.3795, -0.0209] | 0/3 | -0.4043 ± 0.1578 pp; 95% CI [-0.7964, -0.0123] | 0/3 |
| lmhead_ar1_s16_hr4 | 3/3 | -0.3003 ± 0.2556 pp; 95% CI [-0.9352, +0.3346] | 0/3 | -0.5054 ± 0.4570 pp; 95% CI [-1.6407, +0.6298] | 0/3 |
| lmhead_ar1_s8_hr8 | 0/3 | - | 0/0 | - | 0/0 |

## Scale sweep ranking

| rank by MATH | scale | MATH mean | GSM8K mean |
| ---: | ---: | ---: | ---: |
| 1 | 4 | 50.1301% | 85.8479% |
| 2 | 2 | 50.0234% | 85.9237% |
| 3 | 8 | 49.9766% | 85.8226% |
| 4 | 1 | 49.9433% | 86.0248% |
| 5 | 16 | 49.8765% | 85.7215% |
