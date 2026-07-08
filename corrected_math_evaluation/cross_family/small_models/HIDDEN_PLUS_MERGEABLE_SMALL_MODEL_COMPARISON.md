# Hidden LoRA + mergeable AffLoRA small-model comparison

Updated: 2026-07-07

This note isolates the key question:

> On small models, does adding mergeable input/lm_head AffLoRA on top of hidden
> LoRA produce a measurable advantage over hidden LoRA alone?

Source experiment:

- `corrected_math_evaluation/model_families/qwen25/small_models/hidden_mergeable_rank_sweep`
- hidden LoRA rank: `hr4`
- mergeable AffLoRA: tied input/lm_head
- seeds: 42, 43, 44
- eval: MATH clean-4,995 and GSM8K

## Short answer

For Qwen2.5-0.5B, no robust advantage is visible.

For Qwen2.5-1.5B, yes, there is a small but meaningful MATH-side advantage.
The cleanest point is:

- `hidden_hr4 + mergeable_ar16_s1`
- Δ MATH vs hidden-only: `+0.4404 pp`
- 95% CI: `[+0.1379, +0.7430]`
- MATH positive seeds: `3/3`
- Δ GSM8K: `+0.0505 pp`, effectively flat

## Qwen2.5-0.5B

Hidden-only baseline:

- MATH: `20.9610% ± 0.5505`
- GSM8K: `47.1317% ± 0.4442`

Mergeable on top of hidden:

| config | MATH | Δ MATH vs hidden | MATH positive seeds | GSM8K | Δ GSM8K vs hidden |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ar1_s8_hr4` | 21.0477% | +0.0868 pp | 1/3 | 47.8140% | +0.6823 pp |
| `ar2_s8_hr4` | 21.0611% | +0.1001 pp | 2/3 | 46.8284% | -0.3033 pp |
| `ar4_s8_hr4` | 21.0544% | +0.0934 pp | 2/3 | 47.3591% | +0.2274 pp |
| `ar8_s8_hr4` | 20.7808% | -0.1802 pp | 0/3 | 47.3844% | +0.2527 pp |
| `ar16_s8_hr4` | 20.8942% | -0.0667 pp | 1/3 | 47.3086% | +0.1769 pp |

Interpretation:

- MATH deltas are around noise scale.
- Confidence intervals cross zero for all ranks.
- Positive-seed count is weak.
- This is not enough to claim hidden+mergeable helps 0.5B.

## Qwen2.5-1.5B

Hidden-only baseline:

- MATH: `35.9226% ± 0.2848`
- GSM8K: `70.0278% ± 0.7744`

Mergeable on top of hidden:

| config | MATH | Δ MATH vs hidden | MATH positive seeds | GSM8K | Δ GSM8K vs hidden |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ar1_s1_hr4` | 35.8091% | -0.1134 pp | 1/3 | 70.0025% | -0.0253 pp |
| `ar2_s1_hr4` | 36.1628% | +0.2402 pp | 3/3 | 70.0783% | +0.0505 pp |
| `ar4_s1_hr4` | 36.1161% | +0.1935 pp | 3/3 | 69.7245% | -0.3033 pp |
| `ar8_s1_hr4` | 36.2029% | +0.2803 pp | 3/3 | 70.4321% | +0.4043 pp |
| `ar16_s1_hr4` | 36.3630% | +0.4404 pp | 3/3 | 70.0783% | +0.0505 pp |

Interpretation:

- `ar2/ar4/ar8/ar16` all improve MATH on 3/3 seeds.
- `ar16` is the strongest and has a positive 95% CI for MATH.
- GSM8K does not degrade materially; `ar8` is the best balanced point for
  MATH+GSM8K.
- This supports a small hidden+mergeable advantage on 1.5B, mainly on MATH.

## Relation to affine-only result

The affine-only sweep showed that input-side AffLoRA alone is already strong on
small models, and mergeable-only is not consistently better than emb-only.

Therefore the current evidence should be phrased carefully:

- Not supported: "mergeable is generally better than input-only."
- Supported: "on Qwen2.5-1.5B, mergeable AffLoRA can add a small MATH gain when
  combined with hidden LoRA."

## Recommended next check

If this comparison is central, the next efficient run is not a full broad sweep.
Run a targeted replication:

1. Qwen2.5-1.5B:
   - hidden hr4
   - hidden hr4 + mergeable `ar8_s1`
   - hidden hr4 + mergeable `ar16_s1`
   - seeds 45, 46, 47
2. Optional Qwen2.5-0.5B sanity:
   - hidden hr4
   - hidden hr4 + mergeable `ar2_s8`
   - hidden hr4 + mergeable `ar4_s8`
   - seeds 45, 46, 47

The key confirmation criterion should be paired MATH delta vs hidden-only, not
absolute score.
