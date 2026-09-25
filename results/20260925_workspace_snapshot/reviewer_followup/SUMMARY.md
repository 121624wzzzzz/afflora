# Reviewer follow-up experiment summary

Missing runs are shown as incomplete rather than omitted.

## Corrected SFT: Qwen3-0.6B extra seeds

| seed | hidden CE | +A-LoRA CE | ΔCE | hidden PPL | +A-LoRA PPL | item-bootstrap CI |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 42 | 1.215027 | 1.209675 | -0.005352 | 3.370385 | 3.352396 | [-0.005886, -0.004823] |
| 43 | 1.214631 | 1.209771 | -0.004860 | 3.369049 | 3.352716 | [-0.005359, -0.004342] |
| 44 | 1.214628 | 1.209182 | -0.005446 | 3.369040 | 3.350742 | [-0.005968, -0.004933] |

Seed-paired ΔCE: -0.005219 ± 0.000315; 95% CI [-0.006001, -0.004437]; positive improvements: 3/3.

## Corrected SFT: near-equal boundary budget

| method | completed | test CE mean | ΔCE vs hidden | trainable params |
| --- | ---: | ---: | --- | ---: |
| A-LoRA r48 | 3/3 | 1.122102 | -0.010496 ± 0.000127; 95% CI [-0.010811, -0.010181] | 9,528,832 |
| Vocab LoRA r1 | 3/3 | 1.019323 | -0.113275 ± 0.001081; 95% CI [-0.115960, -0.110590] | 9,539,328 |

## Qwen2.5-1.5B untouched-seed Math confirmation

| model/config | paired seeds | Δ MATH pp | MATH positive | Δ GSM8K pp | GSM8K positive |
| --- | ---: | --- | ---: | --- | ---: |
| qwen25_15b + ar8 s1 | 3 | +0.046713 ± 0.664090; 95% CI [-1.602976, +1.696403] | 2/3 | +0.025272 ± 0.381594; 95% CI [-0.922660, +0.973203] | 2/3 |
| qwen25_15b + ar16 s1 | 3 | +0.006673 ± 0.666800; 95% CI [-1.649749, +1.663096] | 2/3 | +0.606520 ± 0.075815; 95% CI [+0.418185, +0.794855] | 3/3 |

## Llama-3.1/3.2 cross-version contrast

| model/config | paired seeds | Δ MATH pp | MATH positive | Δ GSM8K pp | GSM8K positive |
| --- | ---: | --- | ---: | --- | ---: |
| Llama-3.1-8B output (R²=0.9976) | 3 | -0.180180 ± 0.406362; 95% CI [-1.189639, +0.829279] | 1/3 | -0.909780 ± 0.731133; 95% CI [-2.726016, +0.906456] | 0/3 |
| Llama-3.2-3B tied (R²=0.9795) | 3 | +0.106773 ± 0.462776; 95% CI [-1.042825, +1.256372] | 2/3 | -0.379075 ± 1.019987; 95% CI [-2.912862, +2.154712] | 1/3 |

