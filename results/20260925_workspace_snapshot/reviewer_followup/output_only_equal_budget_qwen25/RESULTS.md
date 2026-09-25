# Strict output-only equal-budget A-LoRA versus Vocab-LoRA

## Bottom line

Under this protocol, the claim that output-only A-LoRA is stably better than
an equal-training-parameter direct Vocab-LoRA is **not supported**.

- Direct Vocab-LoRA is decisively better on corrected-SFT held-out CE.
- The two methods are statistically indistinguishable on official IFEval:
  the strict mean is nearly tied and the loose mean is nearly tied in the
  opposite direction.
- The IFEval differences vary substantially by training seed and constraint
  family. They do not provide a stable downstream win for either method.

This does not invalidate earlier parameter-efficiency comparisons against
ordinary hidden LoRA. It narrows the defensible claim: the present experiment
does not establish dominance over a parameter-matched direct vocabulary
adapter.

## Matched design

Both arms use Qwen2.5-1.5B Base, the same corrected-SFT records, the same
hidden-LoRA architecture, optimizer protocol, number of steps, and paired
hidden-LoRA initialization within each seed. Only the output-side
parameterization differs.

| Arm | Output update | Output trainable parameters | Total trainable parameters |
|---|---|---:|---:|
| output-only A-LoRA r50 | `W(h + UDh)` | 153,600 | 9,385,984 |
| output-only Vocab-LoRA r1 | `Wh + BAh` | 153,472 | 9,385,856 |

The output budgets differ by only 128 parameters (0.0834%), and total
trainable budgets differ by 0.00136%. Neither arm has an input-side adapter,
boundary bias, or boundary dropout.

A symmetric 18-run dev search selected:

- A-LoRA: scale 16, boundary-LR multiplier 1;
- Vocab-LoRA: scale 32, boundary-LR multiplier 1.

The selection and its validation are recorded in
[final_selection.md](final_selection.md).

## Corrected-SFT held-out CE

The confirmatory estimate uses untouched independently trained seeds 43--45.
Lower CE is better.

| Metric | A-LoRA | Vocab-LoRA | A-LoRA minus Vocab-LoRA |
|---|---:|---:|---:|
| mean held-out CE | 1.122153 | 1.018712 | +0.103441 |
| seed-level 95% CI for the paired difference |  |  | [+0.100349, +0.106534] |
| seed directions | 0/3 better | 3/3 better | 0 ties |

Every per-seed 10,000-sample paired-item bootstrap interval also favors
Vocab-LoRA. Seed 42, reported separately because it selected the
hyperparameters, gives essentially the same difference (+0.103303).
See [confirmation_summary.md](confirmation_summary.md).

## Official IFEval

All eight frozen checkpoints were evaluated on the same 541 prompts with the
native chat template, greedy decoding, and a 512-token generation cap. The
confirmatory seed-level estimate again uses seeds 43--45.

| Prompt-level metric | A-LoRA mean | Vocab-LoRA mean | A-LoRA minus Vocab-LoRA | seed-level 95% CI |
|---|---:|---:|---:|---:|
| strict | 21.935% | 21.811% | +0.123 pp | [-3.560, +3.806] pp |
| loose | 24.399% | 24.522% | -0.123 pp | [-2.652, +2.406] pp |

Strict directions are one A-LoRA win, one Vocab-LoRA win, and one tie. Loose
directions are two A-LoRA wins and one Vocab-LoRA win. No individual seed's
paired-prompt bootstrap interval or exact McNemar test is significant.

The four-seed descriptive means, including the dev-selection seed 42, favor
A-LoRA by +0.601 pp strict and +0.370 pp loose, but those estimates are not
confirmatory. Full per-seed contingencies and artifact hashes are in
[ifeval_summary.md](ifeval_summary.md).

## Behavioral diagnosis

The near-zero aggregate IFEval result hides mixed constraint behavior over
the independent seeds:

- A-LoRA is descriptively better on `combination` (+10.77 pp), `language`
  (+7.53 pp), and `punctuation` (+5.05 pp).
- Vocab-LoRA is descriptively better on `startend` (+12.44 pp),
  `length_constraints` (+3.50 pp), and `keywords` (+2.86 pp).
- Response-length behavior is seed dependent. Relative to Vocab-LoRA,
  A-LoRA's mean response is 180, 468, and 703 characters longer in seeds
  43, 44, and 45, respectively. The available artifacts do not contain
  finish reasons, so cap hits cannot be identified exactly.

This pattern is more consistent with a generation-calibration tradeoff than
with a uniform instruction-following advantage. Detailed paired diagnostics
are in [ifeval_diagnostics.md](ifeval_diagnostics.md).

## Interpretation

The result is compatible with the two parameterizations having different
inductive biases despite nearly identical parameter counts. Direct rank-one
Vocab-LoRA learns an arbitrary vocabulary-space direction. A-LoRA can learn
more hidden-space directions, but after multiplication by the frozen output
matrix its vocabulary patterns remain in that matrix's column space. A
single arbitrary vocabulary direction can efficiently fit token-frequency
or domain-prior corrections, which is a plausible reason for its large CE
advantage. The lack of a corresponding IFEval advantage shows that the CE
gain does not transfer uniformly to constrained generation.

This is a mechanism hypothesis, not a causal proof. The independent
implementation audit found no critical bug, but identified two relevant
limits:

1. hidden and boundary adapters were jointly trained under global gradient
   clipping, so this is a matched training-protocol comparison rather than a
   pure fixed-hidden expressivity test;
2. the BF16 computation paths of the two residual forms are not perfectly
   identical.

The full audit and lower-risk provenance notes are in [AUDIT.md](AUDIT.md).

## Most decisive remaining control

If a pure parameterization claim is required, start both arms from one
identical trained hidden-LoRA checkpoint, freeze the hidden adapter, and
train only the two boundary modules in FP32 with matched examples, steps,
optimizer, and trainable-parameter budget. This removes hidden-trajectory,
global-clipping, and mixed-precision-path explanations. Behavioral model
selection would additionally require a separate constraint-following dev
set; the official IFEval test must remain untouched.
