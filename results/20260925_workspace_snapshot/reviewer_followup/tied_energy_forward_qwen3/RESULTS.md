# Forward-consistent tied energy check: Qwen3-0.6B, seed 42

## Why this rerun was necessary

The original tied-transpose energy result was pathological after reload
(held-out CE 1.602367; IFEval strict 0.127542). An audit found that the
input/tied energy calculation used the output-codebook orientation `W U D`,
while `AffineEmbedding` actually applies the row-codebook update
`W D^T U^T + 1 beta^T`. The latter is also the codebook represented by the
exact tied-transpose LM head. The trainer was corrected and the formula was
checked against direct dense updates in a regression test.

This rerun changes only that energy geometry. It keeps Qwen3-0.6B, corrected
SFT, seed 42, tied affine rank 16/alpha 128, hidden LoRA rank 8/alpha 16,
effective batch 16, one epoch, tau=0.00625, lambda=100, optimizer, and native
chat templating fixed.

## Checkpoint integrity

The run saved every 250 steps. Every checkpoint was independently reloaded
and evaluated over the full 1,000-example dev set; its CE decreases smoothly:

| Checkpoint step | Reloaded dev CE |
|---:|---:|
| 250 | 1.283122 |
| 500 | 1.239941 |
| 750 | 1.215172 |
| 1000 | 1.200423 |
| 1250 | 1.193096 |
| 1424 | 1.192063 |

The in-training final dev loss was 1.199, consistent with reloaded CE 1.192
(the small difference is batch-mean versus token-weighted aggregation).
This is unlike the original tied-energy checkpoint, whose training log showed
about 1.076 but whose reloaded dev CE was 1.585. Thus the corrected path has
no observed training/save/reload integrity failure.

The final true input/tied energy was near the intended boundary: rho=0.006327
at step 1420, with a small active hinge penalty 5.91e-07.

## Final held-out and IFEval

Held-out CE uses all 1,000 test examples generated as eight non-overlapping
GPU shards. IFEval uses greedy decoding, Qwen native chat template, exactly
512 new tokens, all 541 prompts, eight generation shards, and the official
scorer.

| Configuration, seed 42 | Held-out CE | IFEval strict | IFEval loose |
|---|---:|---:|---:|
| Hidden LoRA | 1.215027 | 0.216266 | 0.240296 |
| Tied, no energy | 1.211280 | 0.207024 | 0.231054 |
| Original tied + energy (invalid geometry) | 1.602367 | 0.127542 | 0.131238 |
| **Tied + forward-consistent energy** | **1.210562** | **0.205176** | **0.229205** |

## Conclusion

The energy principle is supported by this repaired single-seed check: it
keeps the shared codebook update near tau, retains normal held-out CE, and
does not cause the earlier catastrophic IFEval failure. It is not evidence of
an IFEval improvement: the one-seed strict/loose values are slightly below
hidden LoRA and must not be interpreted as significant.

## Three-seed confirmation

Seeds 43 and 44 used the same corrected geometry and checkpointed training.
Their independently reloaded final dev CE values were 1.191985 and 1.190862,
again matching the respective training-time final losses (1.199 and 1.198).
Both then received full eight-GPU held-out and matched-512 IFEval.

| Configuration | Held-out CE, seeds 42/43/44 | Mean CE | Strict mean | Loose mean |
|---|---:|---:|---:|---:|
| Hidden LoRA | 1.215027 / 1.214631 / 1.214628 | 1.214762 | 0.211953 | 0.233518 |
| Tied, no energy | 1.211280 / 1.210755 / 1.210500 | 1.210845 | 0.214418 | 0.235367 |
| **Tied + forward-consistent energy** | **1.210562 / 1.210385 / 1.210137** | **1.210361** | **0.197166** | **0.215650** |

For the corrected energy condition, strict scores are 0.205176 / 0.188540 /
0.197782 and loose scores are 0.229205 / 0.205176 / 0.212569. Relative to
the paired hidden-LoRA runs, the deltas are -0.014788 strict (95% paired-t CI
[-0.022741, -0.006834]) and -0.017868 loose (CI [-0.033994, -0.001743]).

## Updated conclusion

The corrected energy implementation is mechanically valid: direct regression
tests match dense codebook updates, checkpointed dev CE is stable after reload,
and the former catastrophic CE/IFEval collapse was an implementation-path
artifact. Thus the energy principle should not be rejected.

At the tested tied setting, however, the constraint is still a real behavioral
trade-off: it improves held-out CE by 0.004401 relative to hidden LoRA, while
reducing IFEval strict and loose scores with three-seed confidence intervals
below zero. The next experiment is therefore a **small energy-strength sweep**
at the corrected geometry (lambda 0, 1, 10, 30, 100; fixed tau), using seed 42
with independent post-save dev/held-out CE. IFEval should be reserved for the
non-dominated candidates rather than assuming lambda=100 is the desired
operating point.
