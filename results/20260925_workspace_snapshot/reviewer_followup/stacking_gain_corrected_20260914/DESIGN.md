# P1: incremental gains from stacking A-LoRA on hidden LoRA

The central question is whether boundary adaptation adds useful performance
on top of an existing hidden LoRA, including stronger hidden ranks. Placement
reversal is supporting analysis, not the primary estimand of this experiment.

## Fixed matrix and outcomes

- Models: Qwen3-0.6B-Base and Qwen2.5-7B-Base.
- Hidden ranks: 8, 32, 64; seeds 42, 43, 44, all run regardless of outcomes.
- Arms: hidden-only (`none`), hidden + output A-LoRA (`output`), hidden +
  independent input/output A-LoRA (`both`), and hidden-only with an additional
  near-matched hidden parameter budget (`hidden_budget`).
- 72 cells. Reuse the 12 audited P0 rank-8 hidden-only/output checkpoints and
  CE results; train the other 60 cells from scratch. Reused endpoints are
  referenced read-only and hashed. All 12 must first pass configuration,
  finite FP32 tensor/count and shared-initialization audits, then independent
  full dev/test reload evaluations. Require aggregate CE differences <=2e-6
  and maximum per-example CE difference <=5e-5 against the original reports,
  with exactly matching record IDs and supervised token counts. Only these
  freshly recomputed, verified CE reports enter the new matrix. Generation
  is evaluated anew. `reuse_audit/APPROVED.json` is a mandatory launch gate.
- Every cell receives independently reloaded dev/test CE (1,000 examples per
  split, batch 1) and all 541 IFEval prompts with native templates, greedy
  decoding, batch 8, max 512 new tokens, thinking disabled where supported.
  Stop at either the base model EOS or the native supervised `<|im_end|>`
  (IDs 151643 and 151645 for both models). Base generation defaults alone do
  not stop at the assistant turn boundary. All earlier generation artifacts
  lacking this stop rule are archived and excluded; generation is rerun for
  every cell, while unchanged completed training/CE stages must pass an
  explicit retention audit before reuse.
- CE is token-weighted assistant content plus im_end NLL. IFEval primary
  outcome is strict prompt-level accuracy; loose and instruction-level scores
  and generation length/cap-hit rates are diagnostics.
- Prelaunch scorer audit found that the frozen upstream letter-frequency
  checker replaces punctuation with random ASCII letters for keys 1122 (`#`)
  and 1129 (`!`). Keep the unmodified official all-541 scoring with Python
  random seed 0 at every scoring invocation. Also report strict/loose accuracy
  on the fixed remaining 539 prompts, with every paired strict contrast.
  A generation-gain claim must survive this sensitivity. No prompt or model
  result is removed from saved responses or per-example scores.
- Primary comparisons: output versus hidden-only and bilateral versus
  hidden-only at every rank; bilateral versus the budget control measures
  whether relocating the same extra capacity helps. Report every contrast,
  seed direction, mean, SD and nominal paired 95% t CI. With three seeds,
  intervals describe training randomness on fixed evaluation sets, not
  cross-task generalization or simultaneous confidence across all contrasts.
- Rank-8 evidence already exists; rank-32/64 are the capacity extension.
  These SFT splits and IFEval have been inspected in previous research. There
  is no claim that they are a fresh untouched benchmark.

## Training held fixed

Use the frozen corrected trainer/data from P0: 22,780 training conversations,
native templates, maximum length 1024, one epoch / 1424 steps, effective batch
16, LR 2e-4, cosine decay, warmup .03, gradient clip 1, BF16 base and FP32
trainables/Adam moments. Hidden LoRA targets all seven attention/MLP
projections, alpha/rank = 2, dropout .05. Boundary rank 16 / alpha 128,
dropout 0; learned input bias and no output bias. No energy regularizer,
anchor data, KL term, or checkpoint selection is introduced.

Qwen3 uses batch 8 x accumulation 2 without checkpointing. 7B uses batch 4 x
accumulation 4 with gradient checkpointing. The model pairs differ in more
than tying, so their agreement is replication across settings, not a causal
test of tying. This round does not establish superiority to Vocab-LoRA or
other PEFT algorithms.

## Parameter and initialization controls

Output adds 32d parameters; bilateral adds 65d, including the input bias.
The budget control retains the same base rank in every hidden target. It
adds one rank to evenly spaced q_proj layers first, then k_proj layers until
reaching the bilateral budget. This is one predeclared allocation, not an
optimized hidden-budget frontier.

- Qwen3: 21 q_proj layers plus one k_proj, exactly 66,560 extra parameters.
- Qwen2.5-7B: all 28 q_proj layers plus eight k_proj, 233,472 extra versus
  bilateral's 232,960; 512 extra (0.22% of the added budget) favors the control.

Create the ordinary hidden LoRA first, then extend the selected matrices in
an isolated RNG stream. Preserve all existing A rows / B columns, append
zero B columns, and keep each module's alpha/rank = 2. Save explicit rank and
alpha patterns so the ordinary PEFT loader recreates the control exactly.
Shared hidden initialization hashes must match across all arms at each
model/rank/seed. Bilateral affine modules are independent parameters but use
the same initial down/up values as each corresponding P0 one-sided arm.

## Operation and validity gates

Eight physical GPU workers share a queue. A worker only claims a job when
its GPU has less than 3 GiB used and utilization below 15%; occupied GPUs
wait without reserving jobs or preventing other workers from draining the
queue. No existing GPU process is stopped. Training and evaluation for one
cell stay on the same GPU. Completed verified stages are restartable.

Before the full matrix, run eight two-step smoke jobs (both models x four
arms, hidden rank 64), independent CE reloads, short IFEval generations and
official scoring. A CPU check separately verifies budget counts, common
initialization, nonzero learning, and PEFT save/reload equivalence. Freeze
source, data, scorer, protocol and reused endpoint hashes before launch.
Audit errors stop interpretation and remain visible; failed jobs are retained.
No condition is dropped or retuned based on dev/test/IFEval performance.
