# Small affine-only AffLoRA rank sweep

Single-seed first pass for small-model math experiments.

Scope:

- Models: Qwen2.5-0.5B-Base, Qwen2.5-1.5B-Base
- Seed: 42 only for the first pass
- Data: `data/metamathqa_40k/train.jsonl`
- Eval: full MATH test and full GSM8K test
- No hidden LoRA in any treatment
- Baseline: frozen base eval once per model
- Treatments:
  - `emb`: input-embedding-side AffLoRA only, no lm_head adapter
  - `mergeable`: tied input/lm_head AffLoRA, mergeable form
- Ranks: 1, 2, 4, 8, 16
- Scale:
  - Qwen2.5-0.5B: 8
  - Qwen2.5-1.5B: 1

Launch:

```bash
setsid bash corrected_math_evaluation/model_families/qwen25/small_models/affine_only_rank_sweep/launch.sh \
  > corrected_math_evaluation/model_families/qwen25/small_models/affine_only_rank_sweep/launch.log 2>&1 < /dev/null &
```

Analyze:

```bash
${PYTHON_BIN:-python} corrected_math_evaluation/model_families/qwen25/small_models/affine_only_rank_sweep/analyze_results.py
```
