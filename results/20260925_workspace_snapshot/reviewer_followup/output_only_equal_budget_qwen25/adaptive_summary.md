# Output-only equal-budget adaptive summary

Selection uses only the corrected 1,000-example dev split at seed 42.

## Validation

- Runs validated: `14`
- Paired dev examples: `1000`
- Hidden LoRA initialization SHA256: `12de5ac2e5a9fa8032228014a6da12226c6132ea4b62799b4a5f8e1d26e9c469`
- Safetensor parameter audit: `passed`
- Optimizer-group audit: `passed`

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
| output-only Vocab-LoRA r1 | 1 | 1 | 2.0e-04 | 1.006438024 |  |
| output-only Vocab-LoRA r1 | 2 | 1 | 2.0e-04 | 1.006106550 |  |
| output-only Vocab-LoRA r1 | 4 | 1 | 2.0e-04 | 1.005487838 |  |
| output-only Vocab-LoRA r1 | 8 | 0.5 | 1.0e-04 | 1.008175435 |  |
| output-only Vocab-LoRA r1 | 8 | 1 | 2.0e-04 | 1.004593819 |  |
| output-only Vocab-LoRA r1 | 8 | 2 | 4.0e-04 | 1.003894688 |  |
| output-only Vocab-LoRA r1 | 16 | 1 | 2.0e-04 | 1.002403993 | yes |

## Scale-boundary check

`needs_scale32_extension=true`

The check compares scale 16 against scale 8 at matched boundary-LR multiplier 1. The overall flag is true if either method still improves.

| Method | Scale 8 CE | Scale 16 CE | ΔCE (16−8) | Scale16 < Scale8 | Needs scale32 |
|---|---:|---:|---:|:---:|:---:|
| output-only A-LoRA r50 | 1.103406532 | 1.102785799 | -0.000620733 | yes | yes |
| output-only Vocab-LoRA r1 | 1.004593819 | 1.002403993 | -0.002189826 | yes | yes |

## Selected hyperparameters

- output-only A-LoRA r50: `scale=16`, `boundary_lr_scale=1`, `dev_ce=1.102785799`
- output-only Vocab-LoRA r1: `scale=16`, `boundary_lr_scale=1`, `dev_ce=1.002403993`

- Selected dev ΔCE (A-LoRA − Vocab-LoRA): `+0.100381806`
