# Output-only equal-budget final selection

Selection uses only the corrected 1,000-example dev split at seed 42. The search is frozen after round 2.

## Validation

- Runs validated: `18`
- Paired dev examples: `1000`
- Hidden LoRA initialization SHA256: `12de5ac2e5a9fa8032228014a6da12226c6132ea4b62799b4a5f8e1d26e9c469`
- Safetensor parameter audit: `passed`
- Optimizer-group audit: `passed`
- Search stopped after round 2: `true`

## Candidates

| Method | Scale | Boundary LR multiplier | Boundary LR | Dev CE | Selected |
|---|---:|---:|---:|---:|:---:|
| output-only A-LoRA r50 | 1 | 1 | 2.0e-04 | 1.110308589 |  |
| output-only A-LoRA r50 | 2 | 1 | 2.0e-04 | 1.108878686 |  |
| output-only A-LoRA r50 | 4 | 1 | 2.0e-04 | 1.106452684 |  |
| output-only A-LoRA r50 | 8 | 0.5 | 1.0e-04 | 1.104601140 |  |
| output-only A-LoRA r50 | 8 | 1 | 2.0e-04 | 1.103406532 |  |
| output-only A-LoRA r50 | 8 | 2 | 4.0e-04 | 1.102849070 |  |
| output-only A-LoRA r50 | 16 | 1 | 2.0e-04 | 1.102785799 | yes |
| output-only A-LoRA r50 | 16 | 2 | 4.0e-04 | 1.103521879 |  |
| output-only A-LoRA r50 | 32 | 1 | 2.0e-04 | 1.104271733 |  |
| output-only Vocab-LoRA r1 | 1 | 1 | 2.0e-04 | 1.006438024 |  |
| output-only Vocab-LoRA r1 | 2 | 1 | 2.0e-04 | 1.006106550 |  |
| output-only Vocab-LoRA r1 | 4 | 1 | 2.0e-04 | 1.005487838 |  |
| output-only Vocab-LoRA r1 | 8 | 0.5 | 1.0e-04 | 1.008175435 |  |
| output-only Vocab-LoRA r1 | 8 | 1 | 2.0e-04 | 1.004593819 |  |
| output-only Vocab-LoRA r1 | 8 | 2 | 4.0e-04 | 1.003894688 |  |
| output-only Vocab-LoRA r1 | 16 | 1 | 2.0e-04 | 1.002403993 |  |
| output-only Vocab-LoRA r1 | 16 | 2 | 4.0e-04 | 1.001470502 |  |
| output-only Vocab-LoRA r1 | 32 | 1 | 2.0e-04 | 0.999732684 | yes |

## Scale 32 versus scale 16 trend

This comparison holds the boundary-LR multiplier at 1. It is descriptive only; no further sweep is triggered.

| Method | Scale 16 CE | Scale 32 CE | ΔCE (32−16) | Direction |
|---|---:|---:|---:|---|
| output-only A-LoRA r50 | 1.102785799 | 1.104271733 | +0.001485934 | worsening |
| output-only Vocab-LoRA r1 | 1.002403993 | 0.999732684 | -0.002671309 | improving |

## Final selection

- output-only A-LoRA r50: `scale=16`, `boundary_lr_scale=1`, `dev_ce=1.102785799`, `run=qwen25_15b_out_aff_r50_s16_sd42`
- output-only Vocab-LoRA r1: `scale=32`, `boundary_lr_scale=1`, `dev_ce=0.999732684`, `run=qwen25_15b_out_vocab_r1_s32_sd42`

- Selected dev ΔCE (A-LoRA − Vocab-LoRA): `+0.103053115`
