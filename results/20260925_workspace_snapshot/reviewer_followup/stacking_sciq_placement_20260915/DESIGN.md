# SciQ: boundary placement and additive task accuracy

This is a new task-specific experiment, not a continuation of the known CMRC
probability test. The motivation is to distinguish task decisions, parameter
budget, and placement while retaining the existing bilateral A-LoRA topology.
No outcome is assumed positive. Older experiments remain immutable.

## Data and endpoint

Use allenai/sciq at a pinned upstream revision: original train (11,679),
validation (1,000), test (1,000). No support passages: prompts contain only the
question and four candidate answers. Deterministic per-question shuffling avoids
the source's answer-position leakage. Remove normalized-question duplicates
from training if they appear in validation/test, retaining evaluation unchanged;
report duplicate/ambiguous-option audits. Use all remaining training examples.
Preflight found14 training questions and2 test questions whose correct answer
also appears as a distractor after NFKC/lowercase/whitespace normalization.
Before any training, exclude these14 from training and define the primary test
on the998 unambiguous questions. Preserve and report the full1000 nominal-label
score as secondary; retain duplicate-wrong-only choices. Validation has no
ambiguous gold. These decisions use data validity, not model predictions.

Answer with one A/B/C/D token and im_end. Native no-thinking chat template,
assistant label plus EOS loss, no prompt or thinking-prefix supervision. No
truncation is allowed. Training and inference use the same prompt prefix.

Primary: fixed-permutation, four-candidate accuracy on998 unambiguous questions (%). Secondary: candidate
NLL, Brier score, correct-minus-best-wrong logit margin, unrestricted first-token
accuracy and valid-label rate. Test-only robustness: four cyclic permutations
of each original option ordering, reporting mean accuracy and all-four-correct.
This is a discriminative MC protocol, not open-ended generation evidence or a
claim about the official leaderboards. Public benchmark pretraining exposure
cannot be ruled out. Test was not examined during this study's design/tuning.

## Treatments

Qwen3-0.6B and verified Qwen2.5-1.5B-Instruct; fresh adapters for every run.
All use full internal LoRA r8/alpha16/dropout .05 on q/k/v/o/up/down/gate.

1. none: internal LoRA only.
2. both: independent input/output affine maps, r16/alpha128/dropout0;
   input bias on, output bias off. Identical down/up initialization between
   the two maps, with independent parameters.
3. interior: **the identical two maps**, instead applied after decoder blocks
   floor(L/4)-1 and floor(3L/4)-1 (zero indexed). Same initialization, rank,
   scale, biases and optimizer; only placement changes. This is a linear
   placement control, not a reproduction of nonlinear Houlsby adapters.
4. hidden_budget: spend the identical 65*d extra parameters on the previously
   audited evenly distributed q/k rank-expansion rule. Preserve shared r8
   initialization and scale2. Exact budget match required for both models.

Internal positions are fixed before results. A negative comparison rules out
superiority over these positions, not every possible position. The q/k allocation
is a specified equal-budget control, not an exhaustive optimum. The comparison
does not isolate input versus output contributions or prove bilateral optimality.

## Optimization and selection

One epoch; effective batch32 (micro8, accumulation4); AdamW (0.9,0.999), eps1e-8,
weight decay0; cosine LR, 3% warmup, global clip1. Base weights stored BF16,
trainable parameters/optimizer FP32, BF16 autocast training. Full-vocabulary
CE on the two supervised positions, exactly equivalent to masked SFT with two
tokens per example. Use eager attention, TF32 disabled. FP32 eager inference
promotes the same stored base weights; it does not restore original precision.

Tuning: seed2001, each model and each arm gets LR {5e-5,2e-4}, giving16 runs.
Select final-epoch LR independently by validation accuracy, tie by lower
candidate NLL, then lower LR. No intermediate checkpoint selection. No adaptive
grid expansion. This is symmetric LR tuning, not exhaustive optimization.

Confirmation: new paired seeds2002..2006, selected LR for each arm,40 fresh runs.
Always execute the prespecified confirmation, including after negative tuning
results. Freeze selection and all analysis rules before accessing test metrics.
Do not pool tuning seed or historical CMRC results into confirmation.

Primary contrasts both-minus-{none,hidden_budget,interior} within each model.
Report five paired seed differences, paired-t intervals, and Bonferroni-adjusted
95% simultaneous intervals across six primary contrasts. Task-stacking evidence
for a model requires both-versus-none and both-versus-budget lower limits >0;
placement superiority additionally requires both-versus-interior lower limit >0.
Also report whether mean gain reaches a prespecified practical reference of
0.5 percentage points. Failure to pass is reported as inconclusive/negative,
not rescued by secondary metrics. Seed intervals condition on this fixed test.
Question-paired bootstrap is supporting only, conditional on the fitted seeds.

## Verification

Verify reused source hashes against the sealed prior manifest and verify full
base-model SHA256/tokenizer/config hashes before launching. No trained adapter
reuse. Audit splits, option identities/order, native template/token boundaries,
zero truncation, exact budgets, shared initialization, FP32 trainables, finite
gradients and nonzero affine updates. Two-step smoke for every model/arm checks
independent save/reload tensors and FP32 logits, hook invocation and zero-start
equivalence to hidden-only. Check full-sequence masked-loss equivalence on smoke
data. Freeze source/data/protocol hashes before formal tuning and preserve
preflight changes. Independent reload for all final evaluations. Record all
jobs, failures, durations, GPU peak memory, selected settings and final hashes.

References: https://huggingface.co/datasets/allenai/sciq ;
https://arxiv.org/abs/1707.06209 ;
https://proceedings.mlr.press/v97/houlsby19a.html .
