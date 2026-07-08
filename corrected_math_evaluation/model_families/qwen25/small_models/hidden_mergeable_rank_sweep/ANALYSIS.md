# Small-model mergeable AffLoRA math analysis

Updated: 2026-07-06T20:42:08+08:00

MATH reports the clean-4,995 score. Treatments use mergeable tied input/lm_head AffLoRA on top of hidden LoRA hr4. This sweep fixes scale per model and varies affine rank.

## Aggregate accuracy

| config | completed | MATH mean ± sd | GSM8K mean ± sd |
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

## Paired improvement over matching hidden-LoRA baseline

| treatment | paired seeds | Δ MATH | MATH positive seeds | Δ GSM8K | GSM8K positive seeds |
| --- | ---: | ---: | ---: | ---: | ---: |
| qwen25_05b_mergeable_ar1_s8_hr4 | 3/3 | +0.0868 ± 0.4970 pp; 95% CI [-1.1479, +1.3214] | 1/3 | +0.6823 ± 1.3153 pp; 95% CI [-2.5852, +3.9498] | 2/3 |
| qwen25_05b_mergeable_ar2_s8_hr4 | 3/3 | +0.1001 ± 0.1564 pp; 95% CI [-0.2883, +0.4885] | 2/3 | -0.3033 ± 0.5467 pp; 95% CI [-1.6614, +1.0548] | 1/3 |
| qwen25_05b_mergeable_ar4_s8_hr4 | 3/3 | +0.0934 ± 0.2703 pp; 95% CI [-0.5781, +0.7650] | 2/3 | +0.2274 ± 1.2481 pp; 95% CI [-2.8729, +3.3278] | 2/3 |
| qwen25_05b_mergeable_ar8_s8_hr4 | 3/3 | -0.1802 ± 0.1639 pp; 95% CI [-0.5873, +0.2269] | 0/3 | +0.2527 ± 0.8521 pp; 95% CI [-1.8641, +2.3696] | 2/3 |
| qwen25_05b_mergeable_ar16_s8_hr4 | 3/3 | -0.0667 ± 0.4610 pp; 95% CI [-1.2120, +1.0786] | 1/3 | +0.1769 ± 1.8576 pp; 95% CI [-4.4376, +4.7914] | 2/3 |
| qwen25_15b_mergeable_ar1_s1_hr4 | 3/3 | -0.1134 ± 0.1618 pp; 95% CI [-0.5154, +0.2885] | 1/3 | -0.0253 ± 0.3740 pp; 95% CI [-0.9543, +0.9038] | 2/3 |
| qwen25_15b_mergeable_ar2_s1_hr4 | 3/3 | +0.2402 ± 0.1910 pp; 95% CI [-0.2342, +0.7147] | 3/3 | +0.0505 ± 0.4632 pp; 95% CI [-1.1002, +1.2013] | 2/3 |
| qwen25_15b_mergeable_ar4_s1_hr4 | 3/3 | +0.1935 ± 0.1860 pp; 95% CI [-0.2686, +0.6556] | 3/3 | -0.3033 ± 0.5307 pp; 95% CI [-1.6216, +1.0151] | 1/3 |
| qwen25_15b_mergeable_ar8_s1_hr4 | 3/3 | +0.2803 ± 0.1564 pp; 95% CI [-0.1081, +0.6687] | 3/3 | +0.4043 ± 0.3156 pp; 95% CI [-0.3798, +1.1884] | 3/3 |
| qwen25_15b_mergeable_ar16_s1_hr4 | 3/3 | +0.4404 ± 0.1218 pp; 95% CI [+0.1379, +0.7430] | 3/3 | +0.0505 ± 0.3419 pp; 95% CI [-0.7987, +0.8998] | 2/3 |

## Ranking by model

### qwen25_05b

| rank by MATH | affine rank | scale | MATH mean | GSM8K mean | Δ MATH vs hidden |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 8 | 21.0611% | 46.8284% | +0.1001 pp |
| 2 | 4 | 8 | 21.0544% | 47.3591% | +0.0934 pp |
| 3 | 1 | 8 | 21.0477% | 47.8140% | +0.0868 pp |
| 4 | 16 | 8 | 20.8942% | 47.3086% | -0.0667 pp |
| 5 | 8 | 8 | 20.7808% | 47.3844% | -0.1802 pp |

### qwen25_15b

| rank by MATH | affine rank | scale | MATH mean | GSM8K mean | Δ MATH vs hidden |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 1 | 36.3630% | 70.0783% | +0.4404 pp |
| 2 | 8 | 1 | 36.2029% | 70.4321% | +0.2803 pp |
| 3 | 2 | 1 | 36.1628% | 70.0783% | +0.2402 pp |
| 4 | 4 | 1 | 36.1161% | 69.7245% | +0.1935 pp |
| 5 | 1 | 1 | 35.8091% | 70.0025% | -0.1134 pp |

