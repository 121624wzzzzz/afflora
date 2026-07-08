# Qwen3 close-size hidden + mergeable AffLoRA rank sweep

Updated: 2026-07-07T23:20:30+08:00

Phase: `complete`

## Runtime status

| status | jobs |
| --- | ---: |
| complete | 10 |

## Per-run results

| model | hidden rank | affine rank | seed | MATH | Δ MATH vs hidden | GSM8K | Δ GSM8K vs hidden |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3_06b | 8 | 1 | 42 | 31.4314% | -0.4404 pp | 64.8976% | +1.2889 pp |
| qwen3_06b | 8 | 2 | 42 | 31.8919% | +0.0200 pp | 64.2153% | +0.6065 pp |
| qwen3_06b | 8 | 4 | 42 | 31.2713% | -0.6006 pp | 63.1539% | -0.4549 pp |
| qwen3_06b | 8 | 8 | 42 | 31.4314% | -0.4404 pp | 63.9121% | +0.3033 pp |
| qwen3_06b | 8 | 16 | 42 | 31.6717% | -0.2002 pp | 64.3669% | +0.7582 pp |
| qwen3_17b | 4 | 1 | 42 | 41.8418% | -0.3003 pp | 75.1327% | +0.0000 pp |
| qwen3_17b | 4 | 2 | 42 | 41.5816% | -0.5606 pp | 75.8150% | +0.6823 pp |
| qwen3_17b | 4 | 4 | 42 | 41.8619% | -0.2803 pp | 75.5876% | +0.4549 pp |
| qwen3_17b | 4 | 8 | 42 | 42.0020% | -0.1401 pp | 75.6634% | +0.5307 pp |
| qwen3_17b | 4 | 16 | 42 | 42.1421% | +0.0000 pp | 74.9810% | -0.1516 pp |
