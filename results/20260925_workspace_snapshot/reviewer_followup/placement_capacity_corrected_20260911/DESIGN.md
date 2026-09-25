# P0: corrected-SFT placement × hidden capacity

This experiment tests the paper's placement reversal under the corrected
multi-turn SFT pipeline. The protocol is fixed before inspecting these runs.

## Matrix

- Qwen3-0.6B-Base (tied) and Qwen2.5-7B-Base (untied).
- Hidden LoRA rank: 0, 1, 8. Rank 0 means no hidden adapter is installed.
- Boundary: none, input A-LoRA, output A-LoRA.
- Seeds: 42, 43, 44, with all seed-42 jobs queued first. All three seeds run
  regardless of the observed direction. There is no significance-based stopping.
- The rank-0/no-boundary cell is one frozen-base evaluation per model.
- Total: 48 fresh training runs and 2 frozen-base evaluations.

These two models vary in size, family version, and tying simultaneously.
Their agreement is replication across settings, not a causal test of tying.

## Fixed training protocol

The existing corrected split is copied into this directory and checksummed:
22,780 train / 1,000 dev / 1,000 test conversations. Native chat templates,
assistant content plus im_end loss, maximum length 1024, one epoch, effective
batch 16, learning rate 2e-4, cosine decay, warmup .03, clipping 1.0, BF16 base
and FP32 trainable parameters/Adam moments.

All seven attention/MLP projections receive hidden LoRA when enabled; alpha
is twice hidden rank and hidden dropout is .05. A-LoRA rank is 16, alpha 128,
dropout zero. Input has its historical learned hidden bias; output has no
bias. This is the historical placement comparison, not an exactly equal
parameter comparison: input has d extra parameters. Absolute counts are audited.
No bilateral/shared adapter, energy, KL, anchor data, or Vocab-LoRA is used.

Qwen3 uses microbatch 8 × accumulation 2 without gradient checkpointing;
Qwen2.5-7B uses microbatch 4 × accumulation 4 with gradient checkpointing.
These settings remain fixed within each model across all cells.

The source implementation and data are snapshotted before launch. Historical
trained checkpoints are not reused because the shared trainer has changed.
Hidden initialization hashes must agree across placements at each model/rank/
seed; affine down/up initialization hashes must agree across input/output and
hidden ranks at each model/seed. Frozen base parameters remain non-trainable.

The final epoch checkpoint is evaluated after an independent reload. No dev
or test metric selects checkpoints or hyperparameters. Evaluation uses one
example per batch in all cells, FP32 token CE, and records per-example NLL.
The existing dev/test splits have been inspected in prior research; this is a
controlled reproduction, not a claim of validation on a virgin benchmark.

## Primary estimands

For each model and seed, let D(r) = CE(output,r) - CE(input,r).
The primary interaction is I(8) = D(8) - D(0); I(1) is secondary.
A negative interaction alone does not establish a reversal. A reversal also
requires D(0)>0 (input better alone) and D(8)<0 (output better with hidden).
Report paired seed means, sample SD, 95% t intervals, all individual seed
directions, and input/output incremental CE relative to hidden-only.

Seed intervals describe training randomness on this fixed test set. They are
not item-bootstrap intervals and do not establish cross-task generalization.
Partial one/two-seed results are explicitly exploratory. Three-seed intervals
are nominal per contrast; no claim of family-wise significance is made.

No generation-quality or universal parameter-efficiency claim follows from
this P0 matrix alone. P1/P2 are not launched by this scheduler.

## Operation

`python run.py prepare` creates/verifies the immutable source/data snapshot.
`bash launch.sh` runs four GPU workers on physical GPUs 1,2,3,4. Each worker
checks free memory before assigning a new job. Other GPU jobs are never stopped.
Before the full matrix, four two-step smoke jobs cover both models and both
placements at hidden rank 1, including independent checkpoint reloads.
`state.json`, `events.jsonl`, `logs/`, and `RESULTS.md` record progress.
Failures are retained and halt that job; successful jobs continue. Rerunning
the scheduler skips verified completed stages and retries incomplete stages.
