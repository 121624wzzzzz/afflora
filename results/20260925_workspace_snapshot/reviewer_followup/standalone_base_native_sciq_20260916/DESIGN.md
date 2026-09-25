# Single-boundary post-training from pretrained Base checkpoints

The primary question is whether input-only or output-only A-LoRA can improve
actual task behavior when starting from a pretrained Base checkpoint, with all
original weights frozen. The task is supervised SciQ adaptation, not a claim to
reproduce a full general-purpose SFT/RLHF post-training pipeline.

Use official Qwen3-0.6B-Base and Qwen2.5-1.5B (the Base version). Check every
loaded model/tokenizer file against official revision-pinned blob/LFS hashes.
Reuse the sealed preceding study's SciQ questions, label order and exclusions
after full seal verification. Retokenize native completion prompts with verified
Base tokenizers and check prompt/answer concatenation exactly. All adapters are trained afresh, with no fitted adapter
reuse. This is an already inspected benchmark, not a fresh unseen test set.

The initial attempt to retain ChatML was aborted during validation tuning,
before any selection or test evaluation. Weight-only diagnostics showed that
im_end's frozen vocabulary row is identical to 290/266 other rows in these Base
checkpoints. Any hidden/boundary-only adapter leaves these output rows identical:
im_end cannot be uniquely selected, and its CE is bounded below by log(291/267).
This is a structural supervision incompatibility, not evidence that Base SFT
or A-LoRA cannot work. Keep the abandoned attempt and diagnosis intact.

The corrected protocol uses common.prompt(question, options) followed by
"\n\nAnswer:\n", with no chat template or special tokens in the prompt. Supervise
the gold letter then native EOS151643; each native EOS vocabulary row is unique.
Verify tokenizer(prompt + letter) == tokenizer(prompt) + label_id for every row.
Use native EOS151643 for generation stopping. Hold this protocol fixed across all
Base experimental arms. Keep the preceding data split, one-epoch budget, LR grid
and candidate scoring. Prior post-trained results used ChatML and im_end, so they
are contextual references only: any difference confounds initialization stage
with serialization/stopping protocol. Do not attribute it solely to Base versus
Instruct. A stage-only causal comparison would require a new matched-protocol
experiment across both starting checkpoints.

Data: 11,646 training questions, 1,000 validation, 1,000 test. As fixed in the
source study, two ambiguous-gold test rows are excluded from primary analysis
(998 questions); no new exclusions are selected. Four cyclic option orders are
evaluated with candidate scoring; real generation uses canonical order.

Arms from the same Base checkpoint:
- input-only A-LoRA rank16, alpha128, dropout0, input bias enabled;
- output-only A-LoRA rank16, alpha128, dropout0, no bias;
- conventional hidden LoRA r8, alpha16, dropout.05 on all q/k/v/o and
  up/down/gate projections, no boundary adapter;
- unadapted frozen Base reference.

The two A-LoRA arms contain no hidden LoRA. All original parameters, including
the tied vocabulary matrix, stay bitwise unchanged in every trained arm. A
single-side hook changes the effective computation at that side; it need not
preserve effective input/output tying when merged. Hidden LoRA is a larger-budget
quality reference, NOT an equal-budget parameter-efficiency control. No superiority
at matched budget will be claimed. Input includes d more parameters than output.

Training matches the previous experiment: final checkpoint after one epoch,
effective batch32, micro8/accumulation4, AdamW(.9,.999), eps1e-8, weight decay0,
3% warmup and cosine decay, clip1, BF16 frozen weights and autocast, FP32 trainable
parameters and optimizer moments, eager attention, TF32 disabled. Full-vocabulary
CE on the answer letter and native EOS has equal weight, prompt tokens ignored.
Every model/arm gets the same three LRs 5e-5,2e-4,8e-4 with tuning seed5001.
Select highest validation candidate accuracy, then lowest candidate NLL, then
lower LR. Freeze all six choices before any new test evaluation. Confirmation
uses all five seeds5002–5006 regardless of results; same training order is used
across arms within seed. No test-driven LR/epoch/prompt adjustment is allowed.

FP32 evaluation promotes stored BF16 values, not recovery of pre-quantization
precision. Candidate accuracy is the primary content endpoint. Strict greedy
answer accuracy is a co-primary deployment-format endpoint: decoded output
stripped of whitespace must be exactly the gold letter, no explanation parsing.
Greedy native generation: left padding, max16 tokens, no sampling, repetition
penalty1. Save every token/text, stop and length-cap status. A zero strict score
may reflect formatting failure; never equate it with zero task knowledge.

Primary family: 2 models x 3 trained arms x 2 endpoints = 12 contrasts versus
fixed unadapted Base. Report paired-to-fixed-reference differences over five
training seeds, nominal95% and Bonferroni-family12 t intervals. A model/arm passes
the predeclared positive-gain criterion only if both corrected lower bounds are
positive. Also report conditional-on-fitted-models 10,000 question bootstrap
intervals, per-seed corrected/regressed counts, NLL/Brier, first-token validity,
rotation sensitivity and generation validity/termination. Seed and question
uncertainty are distinct; null is not equivalence or noninferiority. Differences
from hidden LoRA are descriptive, not an additional confirmatory significance
family. No task retention or universal downstream success claim is tested.

Smoke (training/validation only) must pass zero-residual equivalence, full-model
versus selected-position logits/loss equivalence, adapter save/reload equivalence,
generation first-step logit checks, parameter/gradient whitelist, original-weight
hash equality and expected hook placement. Freeze code/data/provenance before
formal tuning. Every backward checks all frozen parameter gradients remain None;
every training hashes all original weights before/after; the audit canonicalizes
PEFT base_layer names to compare the same original tensors across arms.

GPU0 and GPU7 were occupied at preparation; use GPU1–6 without altering other jobs.
No old sealed artifact is changed. Final audit checks reuse, source/model hashes,
all checkpoints, selection, per-question arithmetic and generation decoding.
