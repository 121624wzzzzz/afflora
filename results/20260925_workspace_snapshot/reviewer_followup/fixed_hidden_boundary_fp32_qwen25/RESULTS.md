# Fixed-hidden equal-raw-budget boundary control: final results

## Decision

Under corrected SFT on Qwen2.5-1.5B, the claim that output A-LoRA is
reliably better than a direct Vocab-LoRA at the same raw trainable-parameter
budget is **not supported**.

- Held-out corrected-SFT CE strongly and stably favors direct Vocab-LoRA.
- IFEval point estimates favor A-LoRA on average, but the result is
  seed-sensitive and the three-seed confidence intervals include zero.
- Therefore neither loss alone nor a single IFEval seed supports a general
  downstream superiority claim.

This is a mechanism control on previously inspected evaluation sets, not a
new virgin-benchmark claim.

## Controlled comparison

For each hidden seed 42, 43, and 44, both boundary treatments start from the
same completed hidden-only corrected-SFT checkpoint. The hidden LoRA and
backbone are frozen and tensor-identical within each pair. Hidden-LoRA dropout
is replaced by `Identity`.

The two boundary residuals are:

- A-LoRA: `W @ U @ D @ h`, rank 50, 153,600 FP32 trainable parameters;
- Vocab-LoRA: `B @ A @ h`, rank 1, 153,472 FP32 trainable parameters.

The raw budget difference is 128 parameters, or 0.0834%. Both treatments use
the same frozen native-precision base logits, a separately computed FP32
boundary residual, identical shuffled examples, one fixed epoch, no boundary
bias or dropout, no input adapter, no energy/KL/anchor penalty, no gradient
clipping, and TF32 disabled.

This matches raw parameter count, not intrinsic degrees of freedom or function
class. A-LoRA vocabulary directions must lie in the column space of the frozen
output matrix `W`; direct Vocab-LoRA can learn an arbitrary vocabulary-side
direction.

## Corrected-SFT test CE

Lower is better. The sign of `A−V` is A-LoRA CE minus Vocab-LoRA CE.

| hidden seed | hidden-only | A-LoRA | Vocab-LoRA | A−V |
|---:|---:|---:|---:|---:|
| 42 | 1.132984 | 1.126873 | 1.026656 | +0.100216 |
| 43 | 1.132497 | 1.126133 | 1.026240 | +0.099893 |
| 44 | 1.132312 | 1.126273 | 1.025984 | +0.100290 |
| mean | 1.132598 | 1.126426 | 1.026293 | +0.100133 |

- Seed-level paired-t 95% CI for A−V:
  `[+0.099609, +0.100658]`.
- Directions: Vocab-LoRA wins 3/3.
- Mean CE reduction from the same hidden-only starting point:
  A-LoRA `0.006172` (0.545%), Vocab-LoRA `0.106305` (9.386%).
- Per-seed 10,000-sample item bootstrap intervals for A−V are entirely
  positive.

The corrected dev result independently has the same magnitude:
mean A−V `+0.099564`, paired-t 95% CI
`[+0.098994, +0.100134]`.

Changing the scale on seed 42 does not explain the result. At common scale 16,
test A−V is `+0.100472`; at common scale 32 it is `+0.100850`.

## IFEval

The primary inferential unit is an independently trained hidden-seed pair.
Positive differences favor A-LoRA.

| seed | strict A | strict V | A−V (pp) | loose A | loose V | A−V (pp) |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 22.366% | 23.105% | -0.739 | 25.139% | 26.433% | -1.294 |
| 43 | 25.323% | 21.442% | +3.882 | 27.172% | 23.660% | +3.512 |
| 44 | 23.290% | 20.333% | +2.957 | 26.802% | 22.181% | +4.621 |
| mean | 23.660% | 21.627% | +2.033 | 26.371% | 24.091% | +2.280 |

- Strict paired-t 95% CI: `[-4.041, +8.108]` percentage points.
- Loose paired-t 95% CI: `[-5.531, +10.090]` percentage points.
- Directions: A-LoRA 2/3, Vocab-LoRA 1/3, with no ties in both modes.

The positive point estimates are compatible with a useful A-LoRA effect, but
three seed pairs do not establish it. In particular, seed 42 reverses the
direction, and the uncertainty intervals include both practically meaningful
negative and positive effects.

## Mechanistic interpretation

The fixed-hidden control removes hidden/boundary co-adaptation, hidden dropout,
gradient clipping, mixed-precision residual placement, and treatment-dependent
base-logit computation as explanations for the CE gap. The gap persists almost
unchanged, so the most plausible remaining explanation is the output
function-class geometry.

Direct rank-1 Vocab-LoRA can cheaply learn a domain-specific vocabulary vector
and modulate it with one scalar per token. That is well matched to reducing
average teacher-forced token loss through lexical, formatting, and domain-prior
corrections. A-LoRA has higher rank in hidden space, but every resulting
vocabulary direction is projected through the frozen `lm_head`; extra hidden
rank does not grant arbitrary vocabulary-side freedom.

This also explains why the large CE gain need not improve IFEval. CE averages
probability calibration over every supervised token, including changes to
non-argmax logits and stylistic token priors. IFEval measures whether a greedy
sequence satisfies sparse explicit constraints. A large gain in the former can
leave the latter unchanged, while A-LoRA's more structured residual can have a
different sequence-level bias despite worse CE.

## What is and is not established

Established for this protocol:

1. The earlier CE disadvantage is not explained by obvious numerical or
   joint-training confounds.
2. Equal raw boundary parameter count does not imply equal useful capacity.
3. Direct Vocab-LoRA is decisively better for corrected-SFT CE.
4. The CE advantage does not transfer to a detectable IFEval advantage.

Not established:

1. A-LoRA is generally worse on downstream behavior.
2. The positive IFEval mean is a stable A-LoRA improvement.
3. Energy-constrained or jointly trained A-LoRA has the same behavior as this
   boundary-only, no-energy mechanism control.
4. The result transfers to new tasks, model families, or genuinely unseen
   benchmarks.

## Artifact validation

- All 12 boundary CE reports and six hidden-only reports passed source-order,
  token-pairing, arithmetic, hash, precision, and frozen-parameter audits.
- All six 541-prompt IFEval response files passed checkpoint/protocol binding
  validation.
- Strict and loose scoring each contain 3,246 rows.
- Canonical IFEval prompt digest:
  `4d49ac039cbebdfc4beb3f9f30435c4fa25320caffaae441c6b9d868126744eb`.

Detailed machine-readable and tabular results are in `ce_summary.json`,
`ce_summary.md`, `ifeval_main_summary.json`, and `ifeval_main_summary.md`.
