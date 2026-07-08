# Small-model affine-only AffLoRA math sweep results

Updated: 2026-07-07T02:46:42+08:00

Phase: `complete`

## Runtime status

| status | jobs |
| --- | ---: |
| complete | 22 |

## Per-run results

| config | seed | MATH clean | GSM8K |
| --- | ---: | ---: | ---: |
| qwen25_05b_base | None | 11.0911% | 21.9105% |
| qwen25_05b_emb_ar1_s8 | 42 | 20.9009% | 46.0955% |
| qwen25_05b_emb_ar2_s8 | 42 | 20.5806% | 45.6406% |
| qwen25_05b_emb_ar4_s8 | 42 | 21.2212% | 45.9439% |
| qwen25_05b_emb_ar8_s8 | 42 | 21.4014% | 46.9295% |
| qwen25_05b_emb_ar16_s8 | 42 | 20.8609% | 46.3230% |
| qwen25_05b_mergeable_ar1_s8 | 42 | 20.9610% | 46.6262% |
| qwen25_05b_mergeable_ar2_s8 | 42 | 20.9409% | 47.0811% |
| qwen25_05b_mergeable_ar4_s8 | 42 | 21.1411% | 45.5648% |
| qwen25_05b_mergeable_ar8_s8 | 42 | 20.9409% | 44.8067% |
| qwen25_05b_mergeable_ar16_s8 | 42 | 21.8018% | 46.6262% |
| qwen25_15b_base | None | 19.3594% | 43.5936% |
| qwen25_15b_emb_ar1_s1 | 42 | 36.4565% | 68.7642% |
| qwen25_15b_emb_ar2_s1 | 42 | 35.4555% | 70.3563% |
| qwen25_15b_emb_ar4_s1 | 42 | 35.8959% | 69.5982% |
| qwen25_15b_emb_ar8_s1 | 42 | 35.7758% | 69.2191% |
| qwen25_15b_emb_ar16_s1 | 42 | 36.2963% | 69.5982% |
| qwen25_15b_mergeable_ar1_s1 | 42 | 35.1952% | 69.9773% |
| qwen25_15b_mergeable_ar2_s1 | 42 | 36.3363% | 68.9917% |
| qwen25_15b_mergeable_ar4_s1 | 42 | 36.0761% | 69.0675% |
| qwen25_15b_mergeable_ar8_s1 | 42 | 35.7157% | 69.6740% |
| qwen25_15b_mergeable_ar16_s1 | 42 | 36.0961% | 69.4466% |

## Three-seed aggregates

| config | completed seeds | MATH mean ± sd | GSM8K mean ± sd |
| --- | ---: | ---: | ---: |
| qwen25_05b_base | 1/3 | 11.0911% | 21.9105% |
| qwen25_05b_emb_ar1_s8 | 1/3 | 20.9009% | 46.0955% |
| qwen25_05b_emb_ar2_s8 | 1/3 | 20.5806% | 45.6406% |
| qwen25_05b_emb_ar4_s8 | 1/3 | 21.2212% | 45.9439% |
| qwen25_05b_emb_ar8_s8 | 1/3 | 21.4014% | 46.9295% |
| qwen25_05b_emb_ar16_s8 | 1/3 | 20.8609% | 46.3230% |
| qwen25_05b_mergeable_ar1_s8 | 1/3 | 20.9610% | 46.6262% |
| qwen25_05b_mergeable_ar2_s8 | 1/3 | 20.9409% | 47.0811% |
| qwen25_05b_mergeable_ar4_s8 | 1/3 | 21.1411% | 45.5648% |
| qwen25_05b_mergeable_ar8_s8 | 1/3 | 20.9409% | 44.8067% |
| qwen25_05b_mergeable_ar16_s8 | 1/3 | 21.8018% | 46.6262% |
| qwen25_15b_base | 1/3 | 19.3594% | 43.5936% |
| qwen25_15b_emb_ar1_s1 | 1/3 | 36.4565% | 68.7642% |
| qwen25_15b_emb_ar2_s1 | 1/3 | 35.4555% | 70.3563% |
| qwen25_15b_emb_ar4_s1 | 1/3 | 35.8959% | 69.5982% |
| qwen25_15b_emb_ar8_s1 | 1/3 | 35.7758% | 69.2191% |
| qwen25_15b_emb_ar16_s1 | 1/3 | 36.2963% | 69.5982% |
| qwen25_15b_mergeable_ar1_s1 | 1/3 | 35.1952% | 69.9773% |
| qwen25_15b_mergeable_ar2_s1 | 1/3 | 36.3363% | 68.9917% |
| qwen25_15b_mergeable_ar4_s1 | 1/3 | 36.0761% | 69.0675% |
| qwen25_15b_mergeable_ar8_s1 | 1/3 | 35.7157% | 69.6740% |
| qwen25_15b_mergeable_ar16_s1 | 1/3 | 36.0961% | 69.4466% |
