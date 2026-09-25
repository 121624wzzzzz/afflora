# Strict tied-transpose AffLoRA without energy: Qwen3-0.6B

## Purpose

The tied-transpose plus energy run was catastrophically worse (held-out CE
1.596322; IFEval strict 0.135552). This follow-up holds the topology and all
other training choices fixed, but removes only the affine-energy penalty:
there is no `--affine-energy-tau` or `--affine-energy-lambda` argument.

The configuration is corrected SFT on Qwen3-0.6B base, seeds 42/43/44,
native chat template, assistant-content-plus-`im_end` loss, tied input/output
affine rank 16/alpha 128, shared input bias / exact tied output bias, hidden
LoRA rank 8/alpha 16, effective batch 16, one epoch, and the same optimizer.

An existing historical no-energy seed-42 checkpoint was first re-evaluated
over the full 1,000-example test set. Its CE reproduced exactly:

| Checkpoint | Held-out CE |
|---|---:|
| Historical seed 42 | 1.211244 |
| New reproducible seed 42 | 1.211280 |

This rules out adapter-loading or held-out-evaluator error as the explanation
for the tied-plus-energy collapse.

## Full held-out CE

Each seed was evaluated over all 1,000 test examples using eight
non-overlapping GPU shards.

| Configuration | CE, seeds 42/43/44 | Mean CE |
|---|---:|---:|
| Hidden LoRA | 1.215027 / 1.214631 / 1.214628 | 1.214762 |
| Independent bilateral + energy | 1.210067 / 1.209912 / 1.209627 | 1.209869 |
| Tied-transpose + energy | 1.602367 / 1.595354 / 1.591246 | 1.596322 |
| **Tied-transpose, no energy** | **1.211280 / 1.210755 / 1.210500** | **1.210845** |

Removing energy restores tied-transpose CE to the independent-bilateral
range and improves it by 0.003917 versus the hidden-LoRA mean.

## IFEval, matched 512-token cap

Each seed used greedy decoding, Qwen's native chat template, exactly 512 new
tokens, all 541 IFEval prompts, eight non-overlapping GPU generation shards,
and the official evaluator.

| Configuration | Strict, seeds 42/43/44 | Strict mean | Loose, seeds 42/43/44 | Loose mean |
|---|---:|---:|---:|---:|
| Hidden LoRA | 0.216266 / 0.205176 / 0.214418 | 0.211953 | 0.240296 / 0.223660 / 0.236599 | 0.233518 |
| Tied-transpose + energy | 0.127542 / 0.136784 / 0.142329 | 0.135552 | 0.131238 / 0.142329 / 0.146026 | 0.139864 |
| **Tied-transpose, no energy** | **0.207024 / 0.208872 / 0.227357** | **0.214418** | **0.231054 / 0.231054 / 0.243993** | **0.235367** |

Relative to hidden LoRA, the paired deltas are +0.002464 strict (95% paired-t
CI [-0.025213, +0.030142]) and +0.001848 loose (CI [-0.022011, +0.025708]).
They are statistically inconclusive but decisively exclude the large negative
effect of tied-plus-energy in this three-seed comparison.

## Provisional conclusion and next experiment

The failure is not caused by a missing tied bias, nor by tied input/output
geometry alone: the same tied model reproduces well without energy. However,
the comparison does **not yet prove** that the energy principle or even this
hyperparameter pair is at fault: subsequent audit found an input/tied energy
orientation error and a mismatch between training-time and saved-checkpoint
loss for the energy run. Therefore do not treat tied AffLoRA or energy
regularization as failed on the basis of this table.

The immediate next experiment is a forward-consistent, checkpointed rerun at
the same tau/lambda. Only after post-save dev CE agrees with the training loss
should we sweep the energy coefficient (0, 1, 10, 30, 100).
