# Post-wait decision for affine-only rank sweep

Updated: 2026-07-07 05:24 Asia/Shanghai

## Status

- Experiment directory: `corrected_math_evaluation/model_families/qwen25/small_models/affine_only_rank_sweep`
- State: complete
- Jobs: 22/22 complete
- Evaluation batch size: 512
- GPU state after completion: idle
- Analysis file: `ANALYSIS.md`

## Integrity check

The affine-only jobs were checked against their saved `run_args.json`,
`affine_vocab_config.json`, and `trainable_summary.json`.

Checked conditions:

- `emb` jobs use `variant=affine_input`.
- `mergeable` jobs use `variant=affine_input_lm_head`.
- No hidden LoRA is enabled in this sweep.
- `affine_rank` matches the job name.
- `affine_alpha = affine_rank * scale`.
- `mergeable` jobs have both input and lm_head AffLoRA enabled.
- `mergeable` jobs have tied input/lm_head adapters enabled.
- all jobs have non-zero trainable parameters.

Result: no configuration mismatch found.

## Main result

This single-seed sweep does not show a clean reason to immediately expand all
affine-only ranks to multi-seed.

The important pattern is:

- Affine-only training is real: both `emb` and `mergeable` strongly improve over
  frozen base.
- However, `mergeable` is not consistently better than `emb` at the same rank.
- Rank trend is not monotonic.
- MATH and GSM8K often prefer different ranks.

## Key numbers

### Qwen2.5-0.5B

Frozen base:

- MATH: 11.0911%
- GSM8K: 21.9105%

Best affine-only single-seed points:

- Best MATH: `mergeable_ar16_s8`, 21.8018% MATH / 46.6262% GSM8K
- Best GSM8K: `mergeable_ar2_s8`, 20.9409% MATH / 47.0811% GSM8K
- Best emb MATH: `emb_ar8_s8`, 21.4014% MATH / 46.9295% GSM8K

Compared with same-seed hidden hr4 baseline:

- hidden hr4 seed42: 21.4414% MATH / 47.4602% GSM8K
- affine-only can match or slightly exceed one metric, but not both robustly.

### Qwen2.5-1.5B

Frozen base:

- MATH: 19.3594%
- GSM8K: 43.5936%

Best affine-only single-seed points:

- Best MATH: `emb_ar1_s1`, 36.4565% MATH / 68.7642% GSM8K
- Best GSM8K: `emb_ar2_s1`, 35.4555% MATH / 70.3563% GSM8K
- Best mergeable MATH: `mergeable_ar2_s1`, 36.3363% MATH / 68.9917% GSM8K

Compared with same-seed hidden hr4 baseline:

- hidden hr4 seed42: 35.5956% MATH / 69.1433% GSM8K
- affine-only has some single-metric wins, but the preferred rank changes by
  metric.

## Decision

Do not launch a broad multi-seed affine-only rank sweep yet.

Reason: this result falls into the pre-defined "single-point / chaotic rank /
MATH-GSM split" case. The next step should be targeted follow-up, not blind
expansion.

Recommended next experiments:

1. For Qwen2.5-1.5B, run multi-seed only for the strongest small set:
   - `emb_ar1_s1`
   - `emb_ar2_s1`
   - `mergeable_ar2_s1`
2. For Qwen2.5-0.5B, run multi-seed only for:
   - `emb_ar8_s8`
   - `mergeable_ar2_s8`
   - `mergeable_ar16_s8`
3. Keep the previous hidden+mergeable result as the main evidence for
   interaction with hidden LoRA:
   - Qwen2.5-1.5B `hidden_hr4 + mergeable_ar16_s1` remains the cleanest positive
     point from the existing 3-seed sweep.

Interpretation:

Affine-side parameters alone are surprisingly strong on small models, but the
current evidence does not support the claim that the mergeable lm_head side is
generally better than input-only AffLoRA. The apparent benefit of
hidden+mergeable on 1.5B should be treated as an interaction effect and checked
with targeted follow-up, not generalized from this affine-only sweep.
