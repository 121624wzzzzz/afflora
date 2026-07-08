# Small mergeable AffLoRA rank sweep

Purpose: keep hidden LoRA fixed and test whether small-model math results are sensitive to the AffLoRA rank.

Matrix:

- Models:
  - Qwen2.5-0.5B-Base
  - Qwen2.5-1.5B-Base
- Seeds: 42, 43, 44
- Baseline: hidden LoRA hr4
- Treatment: hidden LoRA hr4 + mergeable tied input/lm_head AffLoRA
- AffLoRA ranks: 1, 2, 4, 8, 16
- Fixed scales:
  - Qwen2.5-0.5B: scale=8
  - Qwen2.5-1.5B: scale=1
- Eval: full MATH and full GSM8K

Total jobs: 36 training jobs, followed by full MATH/GSM8K evaluation.

Run:

```bash
setsid bash corrected_math_evaluation/model_families/qwen25/small_models/hidden_mergeable_rank_sweep/launch.sh \
  > corrected_math_evaluation/model_families/qwen25/small_models/hidden_mergeable_rank_sweep/launch.log 2>&1 < /dev/null &
```

Analyze:

```bash
${PYTHON_BIN:-python} corrected_math_evaluation/model_families/qwen25/small_models/hidden_mergeable_rank_sweep/analyze_results.py
```
