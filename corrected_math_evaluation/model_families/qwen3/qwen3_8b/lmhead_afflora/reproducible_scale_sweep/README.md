# Reproducible AffLoRA scale sweep

This experiment replaces the earlier nominal-single-seed sweep whose adapter
initialization happened before `Trainer` applied the requested seed.

The repaired trainer also resets the requested seed immediately before hidden
LoRA construction.  AffLoRA initialization therefore cannot advance the RNG
used by hidden LoRA: matching hidden-only and hidden+AffLoRA runs start from
identical hidden-LoRA tensors for the same seed.

## Questions

1. Does adding an lm-head AffLoRA adapter to hidden LoRA improve MATH and
   GSM8K consistently across seeds?
2. At fixed affine rank 1, how sensitive is the result to the coefficient
   `scale = affine_alpha / affine_rank`?
3. Does the hr4 observation reproduce at hidden rank 8?

## Matrix

- seeds: 42, 43, 44;
- hr4 control: hidden LoRA only, three seeds;
- hr4 treatment: hidden LoRA + lm-head AffLoRA rank 1, scale in
  `{1, 2, 4, 8, 16}`, three seeds each;
- hr8 control: hidden LoRA only, three seeds;
- hr8 replication: hidden LoRA + lm-head AffLoRA rank 1, scale 8, three seeds.

Total: 24 independently trained checkpoints.  Rank is held fixed during the
coefficient sweep, so rank capacity and update scale are not confounded.

All runs use the same 39,500 MetaMathQA training rows, one epoch, batch size
16, cosine LR 2e-4, 3% warmup, BF16 frozen base parameters and FP32 trainable
parameters/optimizer state.  The leaked 499-row internal eval split is not used.

Evaluation uses the corrected 8-shard evaluators on full MATH (5,000; clean
score excludes the five known train-equivalent rows) and full GSM8K (1,319),
greedy decoding, batch size 64 per GPU and 512 generated tokens.

## Runtime files

- `state.json`: authoritative job state and retry counts;
- `events.jsonl`: start/finish/failure events;
- `monitor.jsonl`: independent hourly process/GPU snapshots;
- `checkpoints/`: trained adapters;
- `outputs/` and `logs/`: evaluation artifacts and logs;
- `RESULTS.md`: continuously regenerated status and aggregate results.
- `ANALYSIS.md`: paired multi-seed deltas, confidence intervals and scale ranking;
- `manifest.json`: dataset/code hashes and package versions used by the run.

Start with `launch.sh`.  The launcher starts both the experiment orchestrator
and an independent hourly monitor.  If the orchestrator dies unexpectedly, the
monitor records an alert and terminates only processes whose command line is
scoped to this experiment directory, preventing orphaned GPU jobs.
