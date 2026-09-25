# Matched tied-transpose AffLoRA + energy: Qwen3-0.6B

> **Diagnostic status (2026-07-22):** The numerical observations below are
> retained, but their former causal conclusion is withdrawn. The input/tied
> energy calculation used an update orientation inconsistent with the actual
> `AffineEmbedding` forward pass, and the training-time final dev loss does
> not match an independent evaluation of the saved checkpoint. A
> forward-consistent, checkpointed rerun is in progress before judging the
> energy constraint itself.

## Design

This is the requested paired structural test. It uses the same corrected-SFT
data, Qwen3-0.6B base model, seeds 42/43/44, hidden LoRA r8/alpha16, affine
rank 16/alpha128, optimizer, one epoch, and energy hyperparameters as the
existing independent bilateral energy control:

```text
--affine-energy-tau 0.00625 --affine-energy-lambda 100
```

The only structural intervention is `--tie-affine-input-lm-head-adapters` with
`--affine-lm-head-bias`: input embedding and LM head share one affine map, and
the LM head applies its transpose, preserving Qwen3's tied embedding/lm-head
geometry. This is compared with the matched-cap hidden-LoRA baseline and the
existing independent bilateral energy AffLoRA checkpoint.

## Constraint check

The shared input-codebook energy penalty was active at the final step:

| Seed | Final rho | Final penalty |
|---:|---:|---:|
| 42 | 0.006319 | 4.77e-07 |
| 43 | 0.006320 | 4.91e-07 |
| 44 | 0.006304 | 2.94e-07 |

## Independent held-out test CE

Tied checkpoints were evaluated over the full 1,000-example corrected-SFT
test set using eight non-overlapping GPU shards. Hidden and independent
bilateral values are the existing full-test reports on the same data.

| Configuration | CE, seeds 42/43/44 | Mean CE |
|---|---:|---:|
| Hidden LoRA | 1.215027 / 1.214631 / 1.214628 | 1.214762 |
| Independent bilateral + energy | 1.210067 / 1.209912 / 1.209627 | 1.209869 |
| Tied-transpose + energy | 1.602367 / 1.595354 / 1.591246 | 1.596322 |

The tied-transpose treatment is decisively worse in held-out CE; this is not a
small selection effect.

## IFEval, matched 512-token cap

Every row below uses greedy generation, native Qwen chat template, exactly 512
new tokens, all 541 prompts, and official IFEval scoring. For each treatment,
the 541 prompts were generated as eight non-overlapping GPU shards and merged
only after validating complete, unique coverage.

| Configuration | Strict, seeds 42/43/44 | Strict mean | Loose, seeds 42/43/44 | Loose mean |
|---|---:|---:|---:|---:|
| Hidden LoRA | 0.216266 / 0.205176 / 0.214418 | 0.211953 | 0.240296 / 0.223660 / 0.236599 | 0.233518 |
| Independent bilateral + energy | 0.214418 / 0.205176 / 0.221811 | 0.213802 | 0.234750 / 0.219963 / 0.238447 | 0.231053 |
| Tied-transpose + energy | 0.127542 / 0.136784 / 0.142329 | 0.135552 | 0.131238 / 0.142329 / 0.146026 | 0.139864 |

Relative to hidden LoRA, the independent bilateral energy condition is
inconclusive: +0.001848 strict (95% paired-t CI [-0.010299, +0.013996]) and
-0.002465 loose (CI [-0.012024, +0.007094]). In contrast, tied-transpose is
clearly worse: -0.076402 strict (CI [-0.103306, -0.049498]) and -0.093654
loose (CI [-0.128725, -0.058583]).

## Historical conclusion (superseded)

The hypothesis that the failure came mainly from breaking input/output tying is
rejected for this training setup. The tied update forces one low-rank direction
to simultaneously serve two different SFT gradient roles: modifying how every
token is read as input and modifying how every token is scored as output. That
constraint is much stronger than merely retaining the pretrained tying
inductive bias, and it causes severe interference here. The independent
bilateral energy control instead remains approximately tied with hidden LoRA
on matched-cap IFEval and improves held-out CE.

Do not use this historical conclusion to reject tied maps or energy
regularization. It was based on an implementation and checkpoint-integrity
path now under investigation.
