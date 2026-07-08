# AffLoRA small-scale sweep

This follow-up experiment tests smaller lm-head AffLoRA scales because the first sweep
covered mostly large coefficients.

Matrix:

- model/data/training/eval settings: same as `reproducible_afflora_sweep`;
- hidden LoRA: rank 4, alpha 8;
- lm-head AffLoRA: rank 1;
- new scales: `0.0625`, `0.125`, `0.25`, `0.5`;
- seeds: `42`, `43`, `44`;
- eval batch size: tries `128`, then falls back to `96` and `64` if a shard fails;
- hidden baseline and scale `1/2/4/8/16` are read from `reproducible_afflora_sweep`
  during analysis, not retrained here.

Run:

```bash
bash corrected_math_evaluation/model_families/qwen3/qwen3_8b/lmhead_afflora/rank1_small_scale_sweep/launch.sh
```

Or wait for the current large sweep to complete, then launch automatically:

```bash
nohup bash corrected_math_evaluation/model_families/qwen3/qwen3_8b/lmhead_afflora/rank1_small_scale_sweep/wait_then_launch.sh \
  > corrected_math_evaluation/model_families/qwen3/qwen3_8b/lmhead_afflora/rank1_small_scale_sweep/wait_then_launch.log 2>&1 &
```
