# Small-model mergeable AffLoRA math sweep

Purpose: test whether the repaired MetaMathQA math setup behaves better on smaller
models, while using a mergeable tied input/lm_head AffLoRA form for tied-embedding
models.

Matrix:

- models:
  - `qwen3_06b`: Qwen3-0.6B-Base
  - `qwen25_05b`: Qwen2.5-0.5B-Base
  - `qwen25_15b`: Qwen2.5-1.5B-Base
- seeds: `42`, `43`, `44`
- baseline: hidden LoRA hr4, alpha 8
- treatment: hidden LoRA hr4 + mergeable tied input/lm_head AffLoRA rank 16
- treatment scales: `0.25`, `1`, `4`, `8`
- training/eval data: same repaired MetaMathQA train, MATH full test, GSM8K full test

The treatment command uses:

```text
--variant affine_input_lm_head_plus_hidden_lora
--tie-affine-input-lm-head-adapters
--affine-lm-head-bias
```

so the shared affine adapter is compatible with tied-weight merge.

Eval tries batch `512`, then falls back to `256`, `128`, and `64`.

Run:

```bash
bash corrected_math_evaluation/cross_family/small_models/mergeable_math_scale_sweep/launch.sh
```
