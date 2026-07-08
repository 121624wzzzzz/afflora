# Qwen3 targeted hidden+mergeable multi-seed analysis

Updated: 2026-07-08T10:35:15+08:00

Deltas are paired against the matching hidden-only baseline for the same model, hidden rank, and seed.

## qwen3_06b hidden hr8

### mergeable ar1_s8

| seed | hidden MATH | mergeable MATH | Δ MATH | hidden GSM8K | mergeable GSM8K | Δ GSM8K |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 31.8719% | 31.4314% | -0.4404 pp | 63.6088% | 64.8976% | +1.2889 pp |
| 43 | 32.0921% | 31.5315% | -0.5606 pp | 64.5944% | 64.5944% | +0.0000 pp |
| 44 | 31.2112% | 32.0120% | +0.8008 pp | 62.6990% | 62.6232% | -0.0758 pp |

- completed paired seeds: 3/3
- Δ MATH: -0.0667 ± 0.7537 pp; 95% CI [-1.9390, +1.8056]; positive seeds 1/3
- Δ GSM8K: +0.4043 ± 0.7669 pp; 95% CI [-1.5008, +2.3095]; positive seeds 1/3

### mergeable ar2_s8

| seed | hidden MATH | mergeable MATH | Δ MATH | hidden GSM8K | mergeable GSM8K | Δ GSM8K |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 31.8719% | 31.8919% | +0.0200 pp | 63.6088% | 64.2153% | +0.6065 pp |
| 43 | 32.0921% | 31.1512% | -0.9409 pp | 64.5944% | 65.2009% | +0.6065 pp |
| 44 | 31.2112% | 32.1722% | +0.9610 pp | 62.6990% | 63.3055% | +0.6065 pp |

- completed paired seeds: 3/3
- Δ MATH: +0.0133 ± 0.9510 pp; 95% CI [-2.3490, +2.3757]; positive seeds 2/3
- Δ GSM8K: +0.6065 ± 0.0000 pp; 95% CI [+0.6065, +0.6065]; positive seeds 3/3

## qwen3_17b hidden hr4

### mergeable ar2_s1

| seed | hidden MATH | mergeable MATH | Δ MATH | hidden GSM8K | mergeable GSM8K | Δ GSM8K |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 42.1421% | 41.5816% | -0.5606 pp | 75.1327% | 75.8150% | +0.6823 pp |
| 43 | 41.8218% | 41.7217% | -0.1001 pp | 74.4503% | 75.5118% | +1.0614 pp |
| 44 | 41.7417% | 42.3023% | +0.5606 pp | 75.7392% | 75.8150% | +0.0758 pp |

- completed paired seeds: 3/3
- Δ MATH: -0.0334 ± 0.5635 pp; 95% CI [-1.4333, +1.3665]; positive seeds 1/3
- Δ GSM8K: +0.6065 ± 0.4972 pp; 95% CI [-0.6285, +1.8415]; positive seeds 3/3

### mergeable ar8_s1

| seed | hidden MATH | mergeable MATH | Δ MATH | hidden GSM8K | mergeable GSM8K | Δ GSM8K |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 42.1421% | 42.0020% | -0.1401 pp | 75.1327% | 75.6634% | +0.5307 pp |
| 43 | 41.8218% | 41.9620% | +0.1401 pp | 74.4503% | 74.9052% | +0.4549 pp |
| 44 | 41.7417% | 42.0621% | +0.3203 pp | 75.7392% | 76.1941% | +0.4549 pp |

- completed paired seeds: 3/3
- Δ MATH: +0.1068 ± 0.2320 pp; 95% CI [-0.4696, +0.6832]; positive seeds 2/3
- Δ GSM8K: +0.4802 ± 0.0438 pp; 95% CI [+0.3714, +0.5889]; positive seeds 3/3

