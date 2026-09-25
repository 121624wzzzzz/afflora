# Fixed-hidden FP32-boundary IFEval

IFEval was evaluation-only: the two endpoints and hidden seeds 42/43/44 were predeclared, and no IFEval result selected a configuration.

## Primary seed-level result

| mode | A-LoRA mean | Vocab-LoRA mean | A−V (pp) | paired-t 95% CI (pp) | directions (A/V/tie) |
|---|---:|---:|---:|---:|---:|
| strict | 23.660% | 21.627% | +2.033 | [-4.041, +8.108] | 2/1/0 |
| loose | 26.371% | 24.091% | +2.280 | [-5.531, +10.090] | 2/1/0 |

## Per-seed paired prompt results

| seed | mode | A-LoRA | Vocab-LoRA | A−V (pp) | prompt bootstrap 95% CI (pp) | discordant A/V | McNemar p |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | strict | 22.366% | 23.105% | -0.739 | [-3.882, +2.403] | 35/39 | 0.727547 |
| 42 | loose | 25.139% | 26.433% | -1.294 | [-4.621, +2.033] | 40/47 | 0.520292 |
| 43 | strict | 25.323% | 21.442% | +3.882 | [+0.739, +7.024] | 50/29 | 0.0238198 |
| 43 | loose | 27.172% | 23.660% | +3.512 | [+0.000, +7.024] | 57/38 | 0.0642128 |
| 44 | strict | 23.290% | 20.333% | +2.957 | [-0.370, +6.285] | 49/33 | 0.0970309 |
| 44 | loose | 26.802% | 22.181% | +4.621 | [+1.109, +8.133] | 59/34 | 0.0124006 |

## Interpretation boundary

- Prompt-level bootstrap intervals and McNemar tests describe paired prompt variation within one trained seed.
- The paired-t interval uses trained hidden-seed pairs as the inferential unit; with three pairs it has 2 degrees of freedom.
- Positive deltas favor A-LoRA. Strict and loose are reported separately.

## Validation

- Six checkpoint-bound 541-row response files validated.
- Six strict and six loose official score files validated.
- Canonical digest: `4d49ac039cbebdfc4beb3f9f30435c4fa25320caffaae441c6b9d868126744eb`.
- Protocol manifest: `/commondocument/wz/cross_encoder_workspace/im_exp/lora/reviewer_followup/fixed_hidden_boundary_fp32_qwen25/ifeval_main/protocol_manifest.json`.
