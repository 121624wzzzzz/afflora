# Targeted follow-up: incremental utility and scalar-calibration controls

This protocol is specified after inspecting the complete Base-model P1 results.
It is a follow-up, not a preregistration of the original hypothesis or an untouched test set.
All P1 outcomes, including negative generation outcomes, remain part of the evidence.

## A. Mechanism diagnostic on existing, audited checkpoints

Use every rank-8 P1 checkpoint: two Base models, none/output/both/hidden_budget,
seeds 42/43/44 (24 checkpoints). Independently reload all 24; replay complete
1,000-example dev/test CE with T=1 and require aggregate error <=2e-6 and
maximum per-example CE error <=5e-5. Preserve checkpoint/source/data hashes.

For each arm independently, choose one scalar temperature by minimum dev
token-weighted CE on the fixed grid
0.7,0.75,0.8,0.85,0.9,0.925,0.95,0.975,1,1.025,1.05,1.075,1.1,1.15,1.2,1.3,1.5.
Also choose a separate temperature using only non-im_end supervised tokens.
Save choices before reading test predictions. Report test CE at T=1 and selected T,
content-only CE, teacher-forced top-1 token accuracy, and any boundary grid choice.
This is a finite-grid scalar control, not proof against all possible calibration maps.
No test-based choice, no checkpoint fine-tuning, no generation changes.
Compare equally calibrated arms and calibrated hidden-only versus uncalibrated additions.
Positive scalar temperature preserves token rankings; teacher-forced token accuracy is
still distinct from free generation accuracy.

## B. Matched training on chat-ready checkpoints

Models: local Qwen3-0.6B (post-trained) and Qwen2.5-1.5B-Instruct snapshot
989aa7980e4cf806f80c7fef2b1adb7bc71aa306. Both are tied models; this experiment
does not establish causal effects of tying or replication on an untied model.
They are selected for available complete local weights and bounded replication cost,
before inspecting their follow-up outcome scores.

Primary hidden rank is 8 (the original stacking configuration).
Four arms: none, output, both, and hidden_budget; all seeds 42/43/44:
24 new training runs, plus two unadapted chat-model reference evaluations.
Eight two-step model/arm smoke jobs must pass before any full run.
All full cells run regardless of the first seed's outcome. No reuse of P1 Base adapters.

Same frozen 22,780-conversation train and 1,000-example dev/test splits as P1.
One epoch / 1,424 optimizer steps, max length 1,024, effective batch 16,
BF16 frozen base, FP32 trainables/Adam, LR 5e-5 for every arm, cosine/warmup .03,
clip 1, hidden alpha/rank 2 and dropout .05, boundary r16/alpha128/dropout0,
input bias but no output bias. The smaller common LR is a new fixed post-training
choice, not a tuned optimum; no claim of optimal hidden-only hyperparameters.
No checkpoint selection, extra epochs, KL regularizer, or test-driven tuning.

Native no-thinking templates. Mask supplied empty think prefixes from assistant loss,
so supervised tokens are actual assistant content plus turn-end, consistently with
the generation prompt. Audit single/multi-turn labels, train/generation continuation,
actual output-row distinctness, real batched EOS termination, shared hidden initialization,
FP32 finite trainables, save/reload, and exact parameter counts.

Budget control adds rank to predeclared q/k layers, with the smallest nonnegative
parameter overshoot and then the greatest number of q layers. Layers are evenly spaced.
The allocation depends only on dimensions, never scores. Preserve all shared A rows,
B columns, scaling, and RNG. Qwen3: q21/k1, +66,560; Qwen2.5-1.5B: q22/k18,
+99,840, both exactly matching bilateral boundary parameters. This is one allocation,
not an optimized hidden-budget frontier.

Full dev/test CE and all 541 IFEval prompts for all endpoints, with fixed valid-539
sensitivity (exclude official punctuation-checker defects at keys 1122/1129).
Use greedy no-thinking generation, batch8, max1,024 new tokens; use native configured
EOS plus im_end. Greedy is a controlled common decoding rule, not a claim to match
the Qwen3 model card's recommended sampled decoding performance. Report cap-hit rates,
mean lengths, strict/loose prompt and instruction accuracy. Longer common cap is
declared before new generations, not changed in response to arm outcomes.

Report all paired seed differences, nominal paired t 95% intervals (no multiplicity
correction), same-prompt paired uncertainty, and the unadapted chat references.
An extra practical-benefit claim requires consistency against both hidden-only and
the budget control and must survive valid-539; distinguish improved absolute quality
from merely reducing SFT degradation. This small two-model/rank-8 replication cannot
establish broad cross-task efficacy. No claim that IFEval was absent from pretraining.

## Execution

Run the calibration smoke/full queue first on currently free GPUs. Prepare chat
checks while it runs; then perform chat model/mask/budget/EOS checks, eight smokes,
and the full fixed matrix. Use separate frozen manifests and keep old results immutable.
Poll in the foreground as requested and retain failures rather than dropping cells.

## Sources for the design

- https://proceedings.mlr.press/v70/guo17a.html (scalar temperature calibration)
- https://huggingface.co/Qwen/Qwen3-0.6B (post-training, native no-thinking switch)
- https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct (native chat checkpoint)
- https://arxiv.org/abs/2311.07911 (verifiable instruction-following evaluation)
