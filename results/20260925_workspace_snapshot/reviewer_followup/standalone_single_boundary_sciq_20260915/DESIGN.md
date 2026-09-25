# Standalone single-boundary downstream validation

The question is whether adapting only one vocabulary boundary can improve
actual downstream behavior while every existing base-model parameter remains
unchanged. No hidden LoRA is installed. Input-only and output-only are separate
treatments; no bilateral or shared treatment is substituted for a single layer.

This is a sequential study motivated by earlier results, on an already inspected
SciQ test set. New training seeds do not make it a previously unseen benchmark.
Only model/code/data files verified against the prior sealed SciQ study are
reused; all adapters start from zero residual and are trained from scratch.
Native Qwen3-0.6B and Qwen2.5-1.5B-Instruct model files are fully SHA256-verified.

Data, official source revision, duplicate handling, native no-thinking prompts,
label ids and four cyclic test option orders are identical to the prior study.
Training has 11,646 questions; validation has 1,000. The test contains 1,000
questions, with the previously fixed 2 gold-alias ambiguities excluded from
primary analysis. No new exclusions, training data, labels or scoring rewrites
are chosen from this study's test scores.

Each treatment uses rank16, alpha128, no dropout. Input has one d-vector bias,
output has no bias. This yields input/output 33,792/32,768 trainable parameters
for Qwen3 and 50,688/49,152 for Qwen2.5. Input and output have different raw
budgets; their contrast is descriptive, not an exact-budget method comparison.
The underlying tied embedding/head parameter remains frozen; a single-side
functional hook changes only the designated computation. This does not claim
that merging the single-side update preserves effective input/output tying.

Training: one epoch, effective batch32, microbatch8, AdamW betas(.9,.999),
eps1e-8, weight_decay0, cosine LR with 3% warmup, clip1, BF16 base/AMP,
FP32 adapter and optimizer moments, eager attention, TF32 disabled. The loss is
full-vocabulary CE on the answer letter and its EOS only. Final epoch only.
Tune each model/side symmetrically over LR 5e-5,2e-4,8e-4 using seed3001.
Choose highest validation candidate accuracy, then lowest candidate NLL,
then lower LR. Freeze all four selections before any new test evaluation.
Run all five confirmation seeds3002–3006 regardless of observed effects.

Frozen-base references are independently evaluated with the same implementation.
Previously sealed hidden-r8 results may be shown as a larger-budget reference
after hash/data/protocol checks, not as an equal-budget comparison or paired
new-seed estimate. The study does not establish superiority over other methods
at the same parameter budget.

FP32 evaluation promotes the stored BF16 base values; it does not recover a
pre-quantization checkpoint. Candidate scores use batch16 and the same right
padding/last valid prompt position as the old SciQ study. Greedy generation
uses native left padding, original EOS ids plus im_end, max16 new tokens,
no sampling, repetition_penalty1. Preserve all generated tokens/text and stop
status. A strict answer is decoded text.strip() exactly equal to the gold
uppercase letter. Non-label text is invalid; do not extract a favorable letter
from explanations. Full generation is evaluated on canonical option order;
candidate accuracy and first-token validity also cover all four rotations.

Primary family: 2 models × 2 single-side methods × 2 task metrics = 8 contrasts
against the same frozen base, on the fixed998 canonical questions. Metrics are
candidate accuracy and strict greedy answer accuracy. Report all five seed
effects, nominal and Bonferroni family8 paired-to-fixed-reference t intervals.
A method/model passes the positive-gain criterion only if both corrected
interval lower bounds exceed0. A null result is not equivalence or proof of no
harm; no post hoc noninferiority margin is introduced. Also report NLL/Brier,
format validity, length cap/EOS rates and rotation results. Question bootstrap
conditions on the fitted seeds; it is not a replacement for training-seed
uncertainty. Do not call success on every downstream task from this one task.

Every training audits: explicit optimizer whitelist; every base parameter
requires_grad=False; all base gradients remain None after every backward;
bytewise digest of all frozen parameters before/after training; adapter-only
checkpoint tensor whitelist, exact count and finiteness; fresh zero-start logits;
independent save/reload replay. Input gradients must flow through frozen
Transformer layers; frozen parameters do not mean frozen activations.
Smoke tests use only training/validation rows, check full masked-loss equivalence,
and compare native generation's first-step logits with direct left-padded logits.
All smoke checks must pass before protocol freeze and formal training.

Any correction discovered in smoke is documented before formal results. Source,
data, selection and final artifacts are sealed, preserving earlier studies.
