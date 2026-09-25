# Output-only AffLoRA energy sweep (exploratory)

## Design

Qwen3-0.6B corrected generic SFT, seed 42 only. All settings match the
formal bilateral AffLoRA run except the vocabulary topology is output-only:
`affine_lm_head_plus_hidden_lora`. The output affine map has rank 16 and alpha
128; hidden LoRA remains r8. The sweep compares no energy penalty with three
predeclared energy thresholds at `lambda=100`.

This is a selection experiment, not a final multi-seed claim.

## Important evaluation correction (2026-07-21)

The original formal hidden-LoRA response files are not compatible with this
sweep's 512-new-token decoding cap: when re-tokenized, nearly all of the old
hidden responses reach 1,024 tokens, whereas the output-only sweep reaches
512. Although the old pilot note stated 512, its saved response artifacts show
that a different cap was used. Therefore, the original hidden/bilateral IFEval
numbers in this document are retained as historical records only and must not
be contrasted with the output-only sweep.

We regenerated all three hidden-LoRA seeds from the checkpoints with the
current evaluator, native chat template, greedy decoding, and exactly 512 new
tokens. The work used eight non-overlapping GPU shards and every merged run was
validated to contain all 541 unique prompts. Corrected comparisons are in the
three-seed section below. In particular, the earlier statement that this
output-only method remains below hidden LoRA is withdrawn.

## Constraint check

| Energy tau | Final output rho | Final penalty |
|---:|---:|---:|
| 0.003125 | 0.003378 | 6.42e-06 |
| 0.006250 | 0.006467 | 4.70e-06 |
| 0.012500 | 0.012689 | 3.55e-06 |

All three penalties activate and keep rho close to their specified thresholds.

## IFEval

All 541 prompts were generated greedily with the native chat template. Each
configuration used two non-overlapping generation shards; the four
configurations were evaluated simultaneously across all eight GPUs. Shards
were validated and merged in original dataset order before official scoring.

| Configuration | Strict | Loose |
|---|---:|---:|
| Hidden LoRA, rechecked at 512 tokens | 0.216266 | 0.240296 |
| Bilateral AffLoRA, no energy formal seed 42 | 0.203327 | 0.223660 |
| Output-only, no energy | 0.216266 | 0.229205 |
| Output-only, tau=0.003125 | 0.214418 | 0.232902 |
| Output-only, tau=0.00625 | 0.208872 | 0.232902 |
| Output-only, tau=0.0125 | 0.210721 | 0.223660 |

For the output-only settings, the tightest threshold, tau=0.003125, is the
most balanced candidate: it is close to output-only unconstrained on strict
and is the best loose result. The historical bilateral value uses an unverified
different decoding cap and is not a valid comparison.

## Independent held-out test CE

The 1,000-example corrected-SFT test set was split into two non-overlapping
500-example shards per configuration and evaluated on eight GPUs at batch 32.
CE below is token-weighted across the 270,223 supervised tokens.

| Configuration | Test CE |
|---|---:|
| Bilateral AffLoRA, no energy formal seed 42 | 1.209675 |
| Output-only, no energy | 1.210739 |
| Output-only, tau=0.003125 | 1.212595 |
| Output-only, tau=0.00625 | 1.212224 |
| Output-only, tau=0.0125 | 1.211820 |

The behavior improvement is not predicted by CE: the output-only candidates
have slightly higher held-out CE but better strict IFEval. Do not select tau by
CE alone.

## Three-seed confirmation

The selected setting, `output-only + tau=0.003125 + lambda=100`, was retrained
on fresh paired seeds 43 and 44. Every run generated the full 541 IFEval
prompts greedily. Generation used two non-overlapping shards per run, with the
four seed-43/44 configurations occupying all eight GPUs; merged files were
checked for 541 unique, in-order rows before official scoring. The final output
rho values were 0.003405 (seed 43) and 0.003385 (seed 44), so the constraint
remained active at the selected threshold.

| Configuration | Strict, seeds 42/43/44 | Strict mean | Loose, seeds 42/43/44 | Loose mean |
|---|---:|---:|---:|---:|
| Hidden LoRA, rechecked at 512 tokens | 0.216266 / 0.205176 / 0.214418 | 0.211953 | 0.240296 / 0.223660 / 0.236599 | 0.233518 |
| Output-only, no energy | 0.216266 / 0.203327 / 0.192237 | 0.203943 | 0.229205 / 0.219963 / 0.219963 | 0.223044 |
| Output-only + energy | 0.214418 / 0.205176 / 0.216266 | 0.211953 | 0.232902 / 0.223660 / 0.236599 | 0.231054 |

Relative to output-only without energy, the constrained variant changes strict
IFEval by -0.001848, +0.001848, and +0.024030 across the three seeds: mean
+0.008010 (95% paired-t CI [-0.026758, +0.042778]). Its loose score improves
on all three seeds by +0.003697, +0.003697, and +0.016636: mean +0.008010
(95% paired-t CI [-0.010547, +0.026567]).

Against the corrected hidden baseline, output-only + energy is exactly tied on
mean strict IFEval (paired deltas -0.001848 / 0 / +0.001848; 95% paired-t CI
[-0.004592, +0.004592]) and is -0.002464 on mean loose IFEval (95% paired-t CI
[-0.013069, +0.008140]). Thus there is no evidence of either a downstream win
or a systematic deficit relative to hidden LoRA at the matched decoding cap.
The tight constraint does, however, recover the unconstrained output-only
degradation (+0.008010 mean in both strict and loose IFEval). The historical
bilateral-energy comparison must be regenerated at the same cap before it can
be interpreted.

## Recommended next experiment

First run a matched-cap bilateral recheck if bilateral versus output-only is
still a desired topology claim. The more targeted structural follow-up is a
tied-transpose input/output affine map: Qwen3 has tied input/lm-head weights,
while the formal bilateral adapter learned two independent maps. This test
separates useful vocabulary-space capacity from the loss of the base model's
input/output geometric tying. Retain held-out CE as a diagnostic, but use a
matched decoding protocol for IFEval selection.
