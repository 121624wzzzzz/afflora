# Output-only equal-budget Qwen2.5 phase-1 summary

Selection used only the corrected 1,000-example dev split at seed 42. All artifact, configuration, parameter-budget, and paired-example checks passed.

| Method | Scale | Alpha | Rank | Trainable | Boundary | Dev CE | Selected |
|---|---:|---:|---:|---:|---:|---:|:---:|
| output-only A-LoRA r50 | 1 | 50 | 50 | 9,385,984 | 153,600 | 1.110308589 |  |
| output-only A-LoRA r50 | 2 | 100 | 50 | 9,385,984 | 153,600 | 1.108878686 |  |
| output-only A-LoRA r50 | 4 | 200 | 50 | 9,385,984 | 153,600 | 1.106452684 |  |
| output-only A-LoRA r50 | 8 | 400 | 50 | 9,385,984 | 153,600 | 1.103406532 | yes |
| output-only Vocab-LoRA r1 | 1 | 1 | 1 | 9,385,856 | 153,472 | 1.006438024 |  |
| output-only Vocab-LoRA r1 | 2 | 2 | 1 | 9,385,856 | 153,472 | 1.006106550 |  |
| output-only Vocab-LoRA r1 | 4 | 4 | 1 | 9,385,856 | 153,472 | 1.005487838 |  |
| output-only Vocab-LoRA r1 | 8 | 8 | 1 | 9,385,856 | 153,472 | 1.004593819 | yes |

## Frozen phase-2 scales

- A-LoRA r50 scale: `8`
- Vocab-LoRA r1 scale: `8`
- Selected dev ΔCE (A-LoRA − Vocab-LoRA): `+0.098812713`

Run phase 2 with:

```bash
bash reviewer_followup/run_output_only_equal_budget_qwen25_phase2.sh 8 8
```
