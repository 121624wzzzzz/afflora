# Energy-constrained AffLoRA on corrected SFT / Qwen3-0.6B

## Controlled intervention

This repeats the Qwen3-0.6B formal corrected-SFT AffLoRA runs for paired seeds
42, 43, and 44. All data, model, ranks, optimizer settings, seed handling, and
decoding settings match `corrected_sft_experiment/outputs/formal`. The only
changed training arguments are:

```text
--affine-energy-tau 0.00625
--affine-energy-lambda 100
```

The adapters remain the existing decoupled bilateral topology: independent
input and lm-head affine maps plus hidden LoRA r8; they are not tied.

## Matched decoding-cap correction (2026-07-21)

The initial IFEval table below used historical hidden-LoRA response artifacts
whose effective generation cap did not match the 512-token cap of later
experiments. Those baseline contrasts are therefore historical only. Hidden
LoRA and this independent bilateral energy checkpoint were both regenerated
and officially scored at exactly 512 new tokens, over all 541 prompts and
eight non-overlapping GPU shards per three-seed treatment.

| Configuration | Strict, seeds 42/43/44 | Strict mean | Loose, seeds 42/43/44 | Loose mean |
|---|---:|---:|---:|---:|
| Hidden LoRA, matched 512 cap | 0.216266 / 0.205176 / 0.214418 | 0.211953 | 0.240296 / 0.223660 / 0.236599 | 0.233518 |
| Independent bilateral + energy | 0.214418 / 0.205176 / 0.221811 | 0.213802 | 0.234750 / 0.219963 / 0.238447 | 0.231053 |

The matched comparison is inconclusive rather than negative: bilateral energy
minus hidden is +0.001848 strict (95% paired-t CI [-0.010299, +0.013996]) and
-0.002465 loose (CI [-0.012024, +0.007094]).

## Constraint check

At the end of training, the penalty is active and both relative update norms
are held close to the requested `tau=0.00625`.

| Seed | Input rho | Output rho | Final penalty |
|---:|---:|---:|---:|
| 42 | 0.006224 | 0.006459 | 4.39e-06 |
| 43 | 0.006253 | 0.006488 | 5.67e-06 |
| 44 | 0.006241 | 0.006456 | 4.22e-06 |

## Held-out CE

`delta = energy-constrained AffLoRA - unconstrained AffLoRA`; negative is
better.

| Seed | Energy CE | Unconstrained CE | Delta |
|---:|---:|---:|---:|
| 42 | 1.210067 | 1.209675 | +0.000392 |
| 43 | 1.209912 | 1.209771 | +0.000141 |
| 44 | 1.209627 | 1.209182 | +0.000445 |
| Mean | 1.209869 | 1.209543 | +0.000326 |

## IFEval (541 prompts, official scorer)

Responses were generated greedily using the native chat template. The 541
prompts per seed were partitioned into non-overlapping shards, generated across
all eight GPUs, validated for complete coverage and unique keys, then merged in
the original dataset order before official scoring.

| Mode | Hidden LoRA mean | Unconstrained AffLoRA mean | Energy AffLoRA mean | Energy - unconstrained | Energy - hidden |
|---|---:|---:|---:|---:|---:|
| strict | 0.217498 | 0.198398 | 0.209489 | +0.011091 | -0.008010 |
| loose | 0.237831 | 0.219347 | 0.224276 | +0.004929 | -0.013555 |

The energy regularizer partially recovers Qwen3 strict IFEval relative to the
unconstrained AffLoRA run, chiefly in seed 44, but it does not beat the hidden
LoRA baseline. With three seeds its paired 95% CIs remain wide:

- strict energy - unconstrained: +0.011091, CI [-0.033191, +0.055372]
- strict energy - hidden: -0.008010, CI [-0.024136, +0.008116]
- loose energy - unconstrained: +0.004929, CI [-0.032469, +0.042327]
- loose energy - hidden: -0.013555, CI [-0.023114, -0.003997]

## Artifacts

- `checkpoints/`: three trained energy-constrained adapters.
- `reports/`: independent held-out token CE reports.
- `ifeval_shards/`: non-overlapping generation shards.
- `ifeval_responses/`: validated 541-prompt merged response file per seed.
- `ifeval_scores/`: strict and loose official IFEval outputs.
