# AffLoRA rank-scale sweep

This experiment extends the lm-head AffLoRA sweep beyond rank 1.

Matrix:

- model/data/training/eval settings: same as previous repaired sweeps;
- hidden LoRA: rank 4, alpha 8;
- lm-head AffLoRA ranks: `2`, `4`, `8`, `16`;
- scales: `0.0625`, `0.125`, `0.25`, `0.5`;
- seeds: `42`, `43`, `44`;
- `scale = affine_alpha / affine_rank`, so training passes
  `affine_alpha = scale * affine_rank`;
- rank-1 small-scale results and hidden baseline are read from previous sweeps during
  analysis, not retrained here.

Eval tries batch `128`, then falls back to `96` and `64` if a shard fails.

Run now:

```bash
bash corrected_math_evaluation/model_families/qwen3/qwen3_8b/lmhead_afflora/rank_scale_sweep/launch.sh
```

Wait for the small-scale sweep to complete, then launch automatically:

```bash
setsid bash corrected_math_evaluation/model_families/qwen3/qwen3_8b/lmhead_afflora/rank_scale_sweep/wait_then_launch.sh \
  > corrected_math_evaluation/model_families/qwen3/qwen3_8b/lmhead_afflora/rank_scale_sweep/wait_then_launch.log 2>&1 < /dev/null &
```
