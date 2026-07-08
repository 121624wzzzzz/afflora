# Small-model affine-only AffLoRA math analysis

Updated: 2026-07-07T05:24:41+08:00

MATH reports the clean-4,995 score. This sweep has no hidden LoRA. It compares input-embedding-only AffLoRA (`emb`) and mergeable tied input/lm_head AffLoRA (`mergeable`) against the frozen base model.

## Aggregate accuracy

| config | completed | MATH mean ± sd | GSM8K mean ± sd |
| --- | ---: | ---: | ---: |
| qwen25_05b_base | 1/1 | 11.0911% | 21.9105% |
| qwen25_05b_emb_ar1_s8 | 1/1 | 20.9009% | 46.0955% |
| qwen25_05b_emb_ar2_s8 | 1/1 | 20.5806% | 45.6406% |
| qwen25_05b_emb_ar4_s8 | 1/1 | 21.2212% | 45.9439% |
| qwen25_05b_emb_ar8_s8 | 1/1 | 21.4014% | 46.9295% |
| qwen25_05b_emb_ar16_s8 | 1/1 | 20.8609% | 46.3230% |
| qwen25_05b_mergeable_ar1_s8 | 1/1 | 20.9610% | 46.6262% |
| qwen25_05b_mergeable_ar2_s8 | 1/1 | 20.9409% | 47.0811% |
| qwen25_05b_mergeable_ar4_s8 | 1/1 | 21.1411% | 45.5648% |
| qwen25_05b_mergeable_ar8_s8 | 1/1 | 20.9409% | 44.8067% |
| qwen25_05b_mergeable_ar16_s8 | 1/1 | 21.8018% | 46.6262% |
| qwen25_15b_base | 1/1 | 19.3594% | 43.5936% |
| qwen25_15b_emb_ar1_s1 | 1/1 | 36.4565% | 68.7642% |
| qwen25_15b_emb_ar2_s1 | 1/1 | 35.4555% | 70.3563% |
| qwen25_15b_emb_ar4_s1 | 1/1 | 35.8959% | 69.5982% |
| qwen25_15b_emb_ar8_s1 | 1/1 | 35.7758% | 69.2191% |
| qwen25_15b_emb_ar16_s1 | 1/1 | 36.2963% | 69.5982% |
| qwen25_15b_mergeable_ar1_s1 | 1/1 | 35.1952% | 69.9773% |
| qwen25_15b_mergeable_ar2_s1 | 1/1 | 36.3363% | 68.9917% |
| qwen25_15b_mergeable_ar4_s1 | 1/1 | 36.0761% | 69.0675% |
| qwen25_15b_mergeable_ar8_s1 | 1/1 | 35.7157% | 69.6740% |
| qwen25_15b_mergeable_ar16_s1 | 1/1 | 36.0961% | 69.4466% |

## Improvement over frozen base

| treatment | completed seeds | Δ MATH vs base | MATH positive seeds | Δ GSM8K vs base | GSM8K positive seeds |
| --- | ---: | ---: | ---: | ---: | ---: |
| qwen25_05b_emb_ar1_s8 | 1/1 | +9.8098 pp | 1/1 | +24.1850 pp | 1/1 |
| qwen25_05b_emb_ar2_s8 | 1/1 | +9.4895 pp | 1/1 | +23.7301 pp | 1/1 |
| qwen25_05b_emb_ar4_s8 | 1/1 | +10.1301 pp | 1/1 | +24.0334 pp | 1/1 |
| qwen25_05b_emb_ar8_s8 | 1/1 | +10.3103 pp | 1/1 | +25.0190 pp | 1/1 |
| qwen25_05b_emb_ar16_s8 | 1/1 | +9.7698 pp | 1/1 | +24.4124 pp | 1/1 |
| qwen25_05b_mergeable_ar1_s8 | 1/1 | +9.8699 pp | 1/1 | +24.7157 pp | 1/1 |
| qwen25_05b_mergeable_ar2_s8 | 1/1 | +9.8498 pp | 1/1 | +25.1706 pp | 1/1 |
| qwen25_05b_mergeable_ar4_s8 | 1/1 | +10.0501 pp | 1/1 | +23.6543 pp | 1/1 |
| qwen25_05b_mergeable_ar8_s8 | 1/1 | +9.8498 pp | 1/1 | +22.8961 pp | 1/1 |
| qwen25_05b_mergeable_ar16_s8 | 1/1 | +10.7107 pp | 1/1 | +24.7157 pp | 1/1 |
| qwen25_15b_emb_ar1_s1 | 1/1 | +17.0971 pp | 1/1 | +25.1706 pp | 1/1 |
| qwen25_15b_emb_ar2_s1 | 1/1 | +16.0961 pp | 1/1 | +26.7627 pp | 1/1 |
| qwen25_15b_emb_ar4_s1 | 1/1 | +16.5365 pp | 1/1 | +26.0045 pp | 1/1 |
| qwen25_15b_emb_ar8_s1 | 1/1 | +16.4164 pp | 1/1 | +25.6255 pp | 1/1 |
| qwen25_15b_emb_ar16_s1 | 1/1 | +16.9369 pp | 1/1 | +26.0045 pp | 1/1 |
| qwen25_15b_mergeable_ar1_s1 | 1/1 | +15.8358 pp | 1/1 | +26.3836 pp | 1/1 |
| qwen25_15b_mergeable_ar2_s1 | 1/1 | +16.9770 pp | 1/1 | +25.3980 pp | 1/1 |
| qwen25_15b_mergeable_ar4_s1 | 1/1 | +16.7167 pp | 1/1 | +25.4738 pp | 1/1 |
| qwen25_15b_mergeable_ar8_s1 | 1/1 | +16.3564 pp | 1/1 | +26.0804 pp | 1/1 |
| qwen25_15b_mergeable_ar16_s1 | 1/1 | +16.7367 pp | 1/1 | +25.8529 pp | 1/1 |

## Mergeable minus emb-only at same rank

| model | rank | scale | Δ MATH mergeable-emb | Δ GSM8K mergeable-emb |
| --- | ---: | ---: | ---: | ---: |
| qwen25_05b | 1 | 8 | +0.0601 pp | +0.5307 pp |
| qwen25_05b | 2 | 8 | +0.3604 pp | +1.4405 pp |
| qwen25_05b | 4 | 8 | -0.0801 pp | -0.3791 pp |
| qwen25_05b | 8 | 8 | -0.4605 pp | -2.1228 pp |
| qwen25_05b | 16 | 8 | +0.9409 pp | +0.3033 pp |
| qwen25_15b | 1 | 1 | -1.2613 pp | +1.2130 pp |
| qwen25_15b | 2 | 1 | +0.8809 pp | -1.3647 pp |
| qwen25_15b | 4 | 1 | +0.1802 pp | -0.5307 pp |
| qwen25_15b | 8 | 1 | -0.0601 pp | +0.4549 pp |
| qwen25_15b | 16 | 1 | -0.2002 pp | -0.1516 pp |

## Ranking by model and family

### qwen25_05b emb

| rank by MATH | affine rank | scale | MATH mean | GSM8K mean |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 8 | 21.4014% | 46.9295% |
| 2 | 4 | 8 | 21.2212% | 45.9439% |
| 3 | 1 | 8 | 20.9009% | 46.0955% |
| 4 | 16 | 8 | 20.8609% | 46.3230% |
| 5 | 2 | 8 | 20.5806% | 45.6406% |

### qwen25_05b mergeable

| rank by MATH | affine rank | scale | MATH mean | GSM8K mean |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 8 | 21.8018% | 46.6262% |
| 2 | 4 | 8 | 21.1411% | 45.5648% |
| 3 | 1 | 8 | 20.9610% | 46.6262% |
| 4 | 2 | 8 | 20.9409% | 47.0811% |
| 5 | 8 | 8 | 20.9409% | 44.8067% |

### qwen25_15b emb

| rank by MATH | affine rank | scale | MATH mean | GSM8K mean |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1 | 36.4565% | 68.7642% |
| 2 | 16 | 1 | 36.2963% | 69.5982% |
| 3 | 4 | 1 | 35.8959% | 69.5982% |
| 4 | 8 | 1 | 35.7758% | 69.2191% |
| 5 | 2 | 1 | 35.4555% | 70.3563% |

### qwen25_15b mergeable

| rank by MATH | affine rank | scale | MATH mean | GSM8K mean |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 1 | 36.3363% | 68.9917% |
| 2 | 16 | 1 | 36.0961% | 69.4466% |
| 3 | 4 | 1 | 36.0761% | 69.0675% |
| 4 | 8 | 1 | 35.7157% | 69.6740% |
| 5 | 1 | 1 | 35.1952% | 69.9773% |

