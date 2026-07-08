# Reproducible AffLoRA sweep results

Updated: 2026-07-03T15:12:04+08:00

Phase: `complete`

## Runtime status

| status | jobs |
| --- | ---: |
| complete | 24 |

## Per-run results

| config | seed | MATH clean | GSM8K |
| --- | ---: | ---: | ---: |
| hidden_hr4 | 42 | 49.8899% | 86.3533% |
| hidden_hr4 | 43 | 50.6507% | 85.9742% |
| hidden_hr4 | 44 | 49.9900% | 86.3533% |
| lmhead_ar1_s1_hr4 | 42 | 49.8699% | 86.0500% |
| lmhead_ar1_s1_hr4 | 43 | 49.9700% | 86.2775% |
| lmhead_ar1_s1_hr4 | 44 | 49.9900% | 85.7468% |
| lmhead_ar1_s2_hr4 | 42 | 49.8498% | 86.4291% |
| lmhead_ar1_s2_hr4 | 43 | 50.2102% | 85.2919% |
| lmhead_ar1_s2_hr4 | 44 | 50.0100% | 86.0500% |
| lmhead_ar1_s4_hr4 | 42 | 50.1301% | 86.2775% |
| lmhead_ar1_s4_hr4 | 43 | 50.2903% | 85.5951% |
| lmhead_ar1_s4_hr4 | 44 | 49.9700% | 85.6710% |
| lmhead_ar1_s8_hr4 | 42 | 49.6296% | 85.8984% |
| lmhead_ar1_s8_hr4 | 43 | 50.4304% | 85.7468% |
| lmhead_ar1_s8_hr4 | 44 | 49.8699% | 85.8226% |
| lmhead_ar1_s16_hr4 | 42 | 49.8699% | 85.8984% |
| lmhead_ar1_s16_hr4 | 43 | 50.2903% | 85.8984% |
| lmhead_ar1_s16_hr4 | 44 | 49.4695% | 85.3677% |
| hidden_hr8 | 42 | 49.8899% | 85.4435% |
| lmhead_ar1_s8_hr8 | 42 | 50.3704% | 85.7468% |
| hidden_hr8 | 43 | 50.9910% | 86.3533% |
| lmhead_ar1_s8_hr8 | 43 | 50.4505% | 85.8984% |
| hidden_hr8 | 44 | 50.0100% | 86.2017% |
| lmhead_ar1_s8_hr8 | 44 | 50.7508% | 85.8226% |

## Three-seed aggregates

| config | completed seeds | MATH mean ± sd | GSM8K mean ± sd |
| --- | ---: | ---: | ---: |
| hidden_hr4 | 3/3 | 50.1768% ± 0.4134 | 86.2269% ± 0.2189 |
| lmhead_ar1_s1_hr4 | 3/3 | 49.9433% ± 0.0644 | 86.0248% ± 0.2663 |
| lmhead_ar1_s2_hr4 | 3/3 | 50.0234% ± 0.1806 | 85.9237% ± 0.5790 |
| lmhead_ar1_s4_hr4 | 3/3 | 50.1301% ± 0.1602 | 85.8479% ± 0.3740 |
| lmhead_ar1_s8_hr4 | 3/3 | 49.9766% ± 0.4109 | 85.8226% ± 0.0758 |
| lmhead_ar1_s16_hr4 | 3/3 | 49.8765% ± 0.4105 | 85.7215% ± 0.3064 |
| hidden_hr8 | 3/3 | 50.2970% ± 0.6040 | 85.9995% ± 0.4874 |
| lmhead_ar1_s8_hr8 | 3/3 | 50.5239% ± 0.2005 | 85.8226% ± 0.0758 |
