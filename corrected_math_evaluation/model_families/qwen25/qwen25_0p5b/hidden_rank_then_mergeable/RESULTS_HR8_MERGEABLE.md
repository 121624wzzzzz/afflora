# Qwen2.5-0.5B hidden hr8 + mergeable AffLoRA rank sweep

Updated: 2026-07-07T13:40:30+08:00

Phase: `complete`

## Runtime status

| status | jobs |
| --- | ---: |
| complete | 5 |

## Per-run results

| affine rank | seed | MATH clean | Δ MATH vs hr8 hidden | GSM8K | Δ GSM8K vs hr8 hidden |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 42 | 21.0010% | -0.2603 pp | 47.3844% | -0.4549 pp |
| 2 | 42 | 20.7608% | -0.5005 pp | 47.6118% | -0.2274 pp |
| 4 | 42 | 21.8418% | +0.5806 pp | 48.6732% | +0.8340 pp |
| 8 | 42 | 20.8008% | -0.4605 pp | 47.6876% | -0.1516 pp |
| 16 | 42 | 21.2813% | +0.0200 pp | 47.3086% | -0.5307 pp |
