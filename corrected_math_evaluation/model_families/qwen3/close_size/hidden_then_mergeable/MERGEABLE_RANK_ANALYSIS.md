# Qwen3 close-size hidden + mergeable AffLoRA rank analysis

Updated: 2026-07-08T00:56:29+08:00

This is a seed42 main-trend sweep. Deltas are paired against the matching hidden-only baseline.

## qwen3_06b

Baseline hidden hr8: MATH 31.8719% / GSM8K 63.6088%

| affine rank | MATH | Δ MATH | GSM8K | Δ GSM8K |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 31.4314% | -0.4404 pp | 64.8976% | +1.2889 pp |
| 2 | 31.8919% | +0.0200 pp | 64.2153% | +0.6065 pp |
| 4 | 31.2713% | -0.6006 pp | 63.1539% | -0.4549 pp |
| 8 | 31.4314% | -0.4404 pp | 63.9121% | +0.3033 pp |
| 16 | 31.6717% | -0.2002 pp | 64.3669% | +0.7582 pp |

### Ranking by Δ MATH

| rank by Δ MATH | affine rank | Δ MATH | Δ GSM8K |
| ---: | ---: | ---: | ---: |
| 1 | 2 | +0.0200 pp | +0.6065 pp |
| 2 | 16 | -0.2002 pp | +0.7582 pp |
| 3 | 1 | -0.4404 pp | +1.2889 pp |
| 4 | 8 | -0.4404 pp | +0.3033 pp |
| 5 | 4 | -0.6006 pp | -0.4549 pp |

## qwen3_17b

Baseline hidden hr4: MATH 42.1421% / GSM8K 75.1327%

| affine rank | MATH | Δ MATH | GSM8K | Δ GSM8K |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 41.8418% | -0.3003 pp | 75.1327% | +0.0000 pp |
| 2 | 41.5816% | -0.5606 pp | 75.8150% | +0.6823 pp |
| 4 | 41.8619% | -0.2803 pp | 75.5876% | +0.4549 pp |
| 8 | 42.0020% | -0.1401 pp | 75.6634% | +0.5307 pp |
| 16 | 42.1421% | +0.0000 pp | 74.9810% | -0.1516 pp |

### Ranking by Δ MATH

| rank by Δ MATH | affine rank | Δ MATH | Δ GSM8K |
| ---: | ---: | ---: | ---: |
| 1 | 16 | +0.0000 pp | -0.1516 pp |
| 2 | 8 | -0.1401 pp | +0.5307 pp |
| 3 | 4 | -0.2803 pp | +0.4549 pp |
| 4 | 1 | -0.3003 pp | +0.0000 pp |
| 5 | 2 | -0.5606 pp | +0.6823 pp |

