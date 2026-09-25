# Qwen 3B/4B selected-configuration multi-seed confirmation

Updated: 2026-07-14T08:07:19+08:00

Training: MetaMathQA-40K, one epoch, effective batch 16, hidden LoRA r4. A-LoRA uses rank 16, scale 1, lambda=100, and is jointly trained with hidden LoRA.
Seed 42 is the tau-selection seed; seeds 43--46 are holdout confirmation seeds. Thresholds were frozen across model sizes: input=0.0125, shared=0.025, output=0.00625. Shared is tied/mergeable; output-only has no beta and is mergeable.

## qwen25_3b

### input (tau=0.0125)

| seed | hidden MATH | treatment MATH | paired delta | hidden GSM8K | treatment GSM8K | paired delta | rho |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 (selection) | 41.9820% | 41.7818% | -0.2002 pp | 78.0895% | 79.1509% | +1.0614 pp | 0.01221454 |
| 43 (holdout) | 41.2412% | 41.5215% | +0.2803 pp | 78.3169% | 77.6346% | -0.6823 pp | 0.01231691 |
| 44 (holdout) | 41.2613% | 41.8018% | +0.5405 pp | 78.5444% | 78.2411% | -0.3033 pp | 0.01216439 |
| 45 (holdout) | 41.3213% | 41.8018% | +0.4805 pp | 77.3313% | 77.6346% | +0.3033 pp | 0.01207802 |
| 46 (holdout) | 41.5215% | 41.6216% | +0.1001 pp | 78.0895% | 77.5588% | -0.5307 pp | 0.01192960 |
| mean +/- std | 41.4655 +/- 0.3093 | 41.7057 +/- 0.1277 | 0.2402 +/- 0.3013 pp | 78.0743 +/- 0.4562 | 78.0440 +/- 0.6773 | -0.0303 +/- 0.7164 pp | - |

Paired delta 95% CI (n=5): MATH +0.2402 +/- 0.3741 pp; GSM8K -0.0303 +/- 0.8896 pp.
Holdout-only mean delta (seeds 43--46): MATH +0.3504 pp; GSM8K -0.3033 pp.

### shared (tau=0.025)

| seed | hidden MATH | treatment MATH | paired delta | hidden GSM8K | treatment GSM8K | paired delta | rho |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 (selection) | 41.9820% | 42.2823% | +0.3003 pp | 78.0895% | 78.1653% | +0.0758 pp | 0.02302829 |
| 43 (holdout) | 41.2412% | 41.2613% | +0.0200 pp | 78.3169% | 78.9992% | +0.6823 pp | 0.02343650 |
| 44 (holdout) | 41.2613% | 41.1411% | -0.1201 pp | 78.5444% | 77.8620% | -0.6823 pp | 0.02359568 |
| 45 (holdout) | 41.3213% | 41.7417% | +0.4204 pp | 77.3313% | 78.6202% | +1.2889 pp | 0.02382878 |
| 46 (holdout) | 41.5215% | 41.8418% | +0.3203 pp | 78.0895% | 77.5588% | -0.5307 pp | 0.02361118 |
| mean +/- std | 41.4655 +/- 0.3093 | 41.6537 +/- 0.4623 | 0.1882 +/- 0.2276 pp | 78.0743 +/- 0.4562 | 78.2411 +/- 0.5774 | 0.1668 +/- 0.8277 pp | - |

Paired delta 95% CI (n=5): MATH +0.1882 +/- 0.2827 pp; GSM8K +0.1668 +/- 1.0278 pp.
Holdout-only mean delta (seeds 43--46): MATH +0.1602 pp; GSM8K +0.1895 pp.

### output (tau=0.00625)

| seed | hidden MATH | treatment MATH | paired delta | hidden GSM8K | treatment GSM8K | paired delta | rho |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 (selection) | 41.9820% | 42.4224% | +0.4404 pp | 78.0895% | 78.6202% | +0.5307 pp | 0.00622224 |
| 43 (holdout) | 41.2412% | 41.3614% | +0.1201 pp | 78.3169% | 78.5444% | +0.2274 pp | 0.00625147 |
| 44 (holdout) | 41.2613% | 41.2613% | +0.0000 pp | 78.5444% | 78.3927% | -0.1516 pp | 0.00623386 |
| 45 (holdout) | 41.3213% | 41.5015% | +0.1802 pp | 77.3313% | 77.8620% | +0.5307 pp | 0.00618365 |
| 46 (holdout) | 41.5215% | 41.4014% | -0.1201 pp | 78.0895% | 78.5444% | +0.4549 pp | 0.00620375 |
| mean +/- std | 41.4655 +/- 0.3093 | 41.5896 +/- 0.4735 | 0.1241 +/- 0.2111 pp | 78.0743 +/- 0.4562 | 78.3927 +/- 0.3080 | 0.3184 +/- 0.2907 pp | - |

Paired delta 95% CI (n=5): MATH +0.1241 +/- 0.2621 pp; GSM8K +0.3184 +/- 0.3609 pp.
Holdout-only mean delta (seeds 43--46): MATH +0.0450 pp; GSM8K +0.2654 pp.

## qwen3_4b

### input (tau=0.0125)

| seed | hidden MATH | treatment MATH | paired delta | hidden GSM8K | treatment GSM8K | paired delta | rho |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 (selection) | 48.0280% | 48.3083% | +0.2803 pp | 84.3821% | 84.4579% | +0.0758 pp | 0.01249769 |
| 43 (holdout) | 48.4284% | 48.3884% | -0.0400 pp | 84.5337% | 83.8514% | -0.6823 pp | 0.01248917 |
| 44 (holdout) | 48.0480% | 48.7688% | +0.7207 pp | 84.6854% | 85.2161% | +0.5307 pp | 0.01238348 |
| 45 (holdout) | 48.4885% | 48.0280% | -0.4605 pp | 84.6854% | 84.3063% | -0.3791 pp | 0.01248401 |
| 46 (holdout) | 48.6086% | 48.6486% | +0.0400 pp | 84.6854% | 84.3821% | -0.3033 pp | 0.01245078 |
| mean +/- std | 48.3203 +/- 0.2658 | 48.4284 +/- 0.2918 | 0.1081 +/- 0.4344 pp | 84.5944 +/- 0.1356 | 84.4428 +/- 0.4925 | -0.1516 +/- 0.4674 pp | - |

Paired delta 95% CI (n=5): MATH +0.1081 +/- 0.5394 pp; GSM8K -0.1516 +/- 0.5803 pp.
Holdout-only mean delta (seeds 43--46): MATH +0.0651 pp; GSM8K -0.2085 pp.

### shared (tau=0.025)

| seed | hidden MATH | treatment MATH | paired delta | hidden GSM8K | treatment GSM8K | paired delta | rho |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 (selection) | 48.0280% | 48.1682% | +0.1401 pp | 84.3821% | 84.4579% | +0.0758 pp | 0.02497865 |
| 43 (holdout) | 48.4284% | 48.2683% | -0.1602 pp | 84.5337% | 84.0030% | -0.5307 pp | 0.02496846 |
| 44 (holdout) | 48.0480% | 48.7287% | +0.6807 pp | 84.6854% | 84.3821% | -0.3033 pp | 0.02471624 |
| 45 (holdout) | 48.4885% | 47.9079% | -0.5806 pp | 84.6854% | 84.1547% | -0.5307 pp | 0.02468764 |
| 46 (holdout) | 48.6086% | 47.9479% | -0.6607 pp | 84.6854% | 84.4579% | -0.2274 pp | 0.02487589 |
| mean +/- std | 48.3203 +/- 0.2658 | 48.2042 +/- 0.3294 | -0.1161 +/- 0.5511 pp | 84.5944 +/- 0.1356 | 84.2911 +/- 0.2034 | -0.3033 +/- 0.2514 pp | - |

Paired delta 95% CI (n=5): MATH -0.1161 +/- 0.6843 pp; GSM8K -0.3033 +/- 0.3122 pp.
Holdout-only mean delta (seeds 43--46): MATH -0.1802 pp; GSM8K -0.3980 pp.

### output (tau=0.00625)

| seed | hidden MATH | treatment MATH | paired delta | hidden GSM8K | treatment GSM8K | paired delta | rho |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 (selection) | 48.0280% | 48.3483% | +0.3203 pp | 84.3821% | 84.4579% | +0.0758 pp | 0.00623145 |
| 43 (holdout) | 48.4284% | 48.4885% | +0.0601 pp | 84.5337% | 84.0030% | -0.5307 pp | 0.00624013 |
| 44 (holdout) | 48.0480% | 48.7087% | +0.6607 pp | 84.6854% | 84.5337% | -0.1516 pp | 0.00624571 |
| 45 (holdout) | 48.4885% | 48.2883% | -0.2002 pp | 84.6854% | 84.0030% | -0.6823 pp | 0.00625109 |
| 46 (holdout) | 48.6086% | 49.1091% | +0.5005 pp | 84.6854% | 84.3821% | -0.3033 pp | 0.00624175 |
| mean +/- std | 48.3203 +/- 0.2658 | 48.5886 +/- 0.3329 | 0.2683 +/- 0.3440 pp | 84.5944 +/- 0.1356 | 84.2760 +/- 0.2549 | -0.3184 +/- 0.3004 pp | - |

Paired delta 95% CI (n=5): MATH +0.2683 +/- 0.4272 pp; GSM8K -0.3184 +/- 0.3730 pp.
Holdout-only mean delta (seeds 43--46): MATH +0.2553 pp; GSM8K -0.4170 pp.
