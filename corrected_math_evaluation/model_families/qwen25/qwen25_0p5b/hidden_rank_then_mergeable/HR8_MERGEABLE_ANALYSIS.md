# Qwen2.5-0.5B hidden hr8 + mergeable AffLoRA rank analysis

Updated: 2026-07-07T14:10:48+08:00

This is a main-trend sweep before adding more seeds.
Baseline is the matching hidden-only `hr8` result from the hidden-rank stage.

| affine rank | seed | MATH | Δ MATH vs hr8 hidden | GSM8K | Δ GSM8K vs hr8 hidden |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 42 | 21.0010% | -0.2603 pp | 47.3844% | -0.4549 pp |
| 2 | 42 | 20.7608% | -0.5005 pp | 47.6118% | -0.2274 pp |
| 4 | 42 | 21.8418% | +0.5806 pp | 48.6732% | +0.8340 pp |
| 8 | 42 | 20.8008% | -0.4605 pp | 47.6876% | -0.1516 pp |
| 16 | 42 | 21.2813% | +0.0200 pp | 47.3086% | -0.5307 pp |

## Ranking by MATH delta

| rank | seed | Δ MATH | Δ GSM8K |
| ---: | ---: | ---: | ---: |
| 4 | 42 | +0.5806 pp | +0.8340 pp |
| 16 | 42 | +0.0200 pp | -0.5307 pp |
| 1 | 42 | -0.2603 pp | -0.4549 pp |
| 8 | 42 | -0.4605 pp | -0.1516 pp |
| 2 | 42 | -0.5005 pp | -0.2274 pp |
