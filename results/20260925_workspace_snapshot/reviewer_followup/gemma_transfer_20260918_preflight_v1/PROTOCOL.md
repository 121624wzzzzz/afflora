# Gemma-2-9B Base: a third-source-family stacking replication

This study was requested after Qwen and Llama results were known. Its question is
whether boundary A-LoRA adds downstream value on Gemma under equal parameter and
LR-search budgets. Preserve every earlier result, including negative conditions.
Freeze this protocol and executable science before any Gemma-9B benchmark output.
No outcome-dependent candidate, task, seed, checkpoint or metric changes.

## Model and native semantics

Google pretrained `google/gemma-2-9b`, revision
`33c193028431c2fde6c6e51f29e6f17b60cbfac6`. Obtain the public Google distribution on
ModelScope, and verify all runtime file bytes against official Google HF git/LFS
identities before admitting a worker. No Instruct checkpoint or trained adapter
reuse. Native BOS=2, EOS=1, PAD=0. Verify physical input/output weight tying.

The checkpoint stores FP32 weights. All arms, including Base, load the same
BF16-rounded frozen weights for training and cast those weights back to FP32 for
evaluation. Thus evaluation is FP32 arithmetic on the common BF16-rounded base,
not recovery of the original FP32 checkpoint precision. Adapters and Adam moments
remain FP32. Disable TF32. This precision policy is common within this study.

Use native eager attention, preserving attention logit softcap=50; this installed
SDPA implementation would ignore the softcap. Preserve final logit softcap=30 in
both selected-target training loss and candidate scoring, with independent native
full-forward checks. For E, apply the affine map to raw word embeddings before
Gemma's native sqrt(hidden_size) scaling. U maps final hidden states before the
LM head and its native softcap. Independent E/U residuals act on tied frozen base
weights; their effective adapted vocabulary matrices need not remain tied.
Verify zero-residual identity, nonzero effective-vocabulary equivalence using a
synthetic CPU Gemma oracle, actual hook execution, and loss masking/alignment.

## Data and evaluation

Use exactly the sealed Llama LR study's WikiSQL train/dev/confirmation sets
(2048/1024/2048 examples; 1938/854/1628 tables) and the sealed Qwen expansion's
TREC50 train/dev/test sets (2048/256/500 examples; official full 500-question test).
Verify each copied input against its source manifest. Retain the original prompts,
training order rule, labels and database source splits; re-encode with Gemma's
native tokenizer without truncation. Check table/text disjointness across splits,
every prompt and label against its source, gold SQL executions, SQL mutations,
and native BOS/EOS plus concatenation consistency.

These are shared benchmarks already evaluated on other model families. They are
not globally unseen project data and do not establish absence from pretraining.
Gemma choices use development only; confirmation is opened only after all 72
development fits and their audits pass and SELECTION.json is immutable. Base
confirmation evaluations are admitted at that same gate, after model selection.

WikiSQL primary: official execution accuracy, greedy native generation with cap
256. Secondary: logical-form equality, valid-query/strict-JSON/stop/cap rates and
the recorded alternative parameter-binding check. Re-execute all valid predicted
SQL with the official evaluator. TREC50 primary: accuracy from the complete
two-token log probability of each fixed category code 00..49, no EOS score and no
length normalization. Each code is exactly two native tokens. Verify cached scores
against five independent complete native forwards for short/long examples on each
evaluated split; record every candidate score. Secondary: macro F1 over all 50
classes and coarse accuracy. These are different task metrics, not pooled scores.

## Training and exact budget control

Three arms: hidden LoRA H; exact-budget H; H+E+U. All original model parameters
remain frozen. H targets q/k/v/o and gate/up/down projections, rank8, alpha16,
dropout .05. E/U each rank16, alpha128, dropout0; E bias present, U bias absent.
H has 27,009,024 parameters; H+E+U and budget H each have 27,241,984. The extra
232,960 parameters are assigned by a fixed integer-rank rule: minimize excess,
then maximize added q ranks; one rank on 23 evenly spaced q layers and 10 evenly
spaced k layers. Preserve all common rank8 initial tensors and alpha/r=2 exactly.
All extra parameters must affect actual computation and change during training.

Each fit: 2048 examples, one epoch, 64 optimizer updates, effective batch32,
microbatch2. Same seed-paired shuffle and common H initialization across arms.
AdamW betas(.9,.999), eps1e-8, weight_decay0, 2-step warmup then cosine decay,
joint global gradient clipping at1. True target-token mean loss across microbatches
including EOS, prompt masked. Use the final checkpoint only. Record CPU-side
gradient-group and actual step-update norms without extra diagnostic GPU buffers.
Validate frozen weight digests, parameter/gradient whitelists, FP32 optimizer
moments, exact adapter save/destructive-zero/reload loss, and finite values.

## Equal search and independent seed confirmation

Every task and every arm searches the identical common LR grid, fixed tie order:
2e-4, 1e-4, 4e-4, 5e-5, 3e-4, 8e-4. E/U LR equals H LR; no extra boundary-ratio
dimension. Use paired development seeds7400/7401: 6 candidates x 2 seeds x 3 arms
x 2 tasks =72 full development fits. Select each task/arm's greatest two-seed mean
primary development score, break exact ties by the fixed candidate order.
The finite grid is not a claim of global hyperparameter optimality. Equal fits
and processed examples do not imply identical FLOPs or runtime across methods.

Refit each of the six selected task/arm configurations with five new seeds
7500..7504: 30 confirmation fits. Evaluate one frozen Base per task, seed7398;
Base has no fitted trainable parameters and is reported descriptively. Ten short
technical smoke fits (seed7399,64 training examples/2 updates each) cover all arms
and low/high H+E+U LRs on both tasks. Main queue is finite: 114 jobs total.
No old Gemma fit exists for compatibility comparison. Smokes verify native
semantics, maximum training-length memory and actual common initial tensors.

## Prespecified inference and descriptive diagnostics

Four primary paired contrasts: H+E+U minus H, and H+E+U minus exact-budget H,
separately on WikiSQL and TREC50. Report all five differences, mean, sample SD,
df4 paired t intervals both marginal95% and Bonferroni family4 simultaneous95%.
Intervals condition on fixed data and selected configs; five seeds are too few
to validate the distributional assumption. No pooling with historical families.

Supplementary paired cluster bootstrap, fixed before Gemma outputs: 20,000 draws,
seed20300918 for WikiSQL and20300919 for TREC50. Average correctness over the same
five fitted models, sample tables/questions with replacement, retain every member
of a sampled cluster, and compute the ratio of summed correctness differences to
sampled question count. Report percentile marginal95% and family4 intervals
(.625%,99.375%). This covers cluster sampling conditional on fitted models, not
joint seed, tuning and data uncertainty; it never replaces the primary analysis.

Training norms/clipping, secondary metrics, recovered/regressed questions, and
the fixed-full-set-denominator both-valid/other-validity WikiSQL partition are
descriptive. They cannot identify a causal mechanism, pure placement effects or
universal architecture superiority. Preserve all negative and uncertain findings.

## Execution and closeout

Admit one worker per GPU with at least68GiB free, without terminating other users'
processes. Any failed worker or audit stops new admissions while active workers
drain. Technical corrections before search must preserve all failed artifacts,
explain the correction, and repeat the relevant gate; never retry by score.
After all runs: recheck source/data/code/model hashes, all selected configs and
saved outputs, native tokenization, physical tying, actual common H initialization
and paired orders; produce tables/figures/Chinese interpretation and a complete
seal manifest, followed by separate full verification. No wall-clock speed claims.
