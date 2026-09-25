# Post-training task pilot: function calls and structured extraction

Date: 2026-09-16. Status: preparation; no new model results observed.

Purpose: test whether supervised generative tasks offer measurable content
learning and then test standalone and additive A-LoRA effects. Retain SciQ,
CMRC, IFEval and their negative/null findings. A low Base score alone is not
evidence for a useful task or for A-LoRA. These public datasets cannot certify
absence of pretraining contamination.

## Stage A: task and implementation pilot

Use the previously authenticated Qwen3-0.6B-Base and Qwen2.5-1.5B original
weights, after rehashing their model/tokenizer/config files. Fresh adapters;
never reuse fitted SciQ or CMRC adapters. Copy and hash the audited affine
implementation and hidden-budget helper before adapting the training wrapper.

Tasks: ToolACE first-turn function-call generation, and CLUENER2020 entity
extraction serialized as JSON. Dataset conversion, filtering, split rules,
prompts and gold-independent scoring must be frozen before model evaluation.
ToolACE development partitions must separate tools through connected components
of normalized function names/signatures; report component sizes and any loss of
coverage. CLUENER uses an internal development partition from official train;
official public dev is kept out of pilot and hyperparameter selection. No use of
gold answers for generation length, extraction, input truncation or prompt choice.

Initially compare unadapted Base with all-layer hidden LoRA r8 (alpha16,
dropout0.05) on both tasks/models. One pilot seed 6100, LR2e-4, effective batch32,
up to 2,048 deterministic training examples, one epoch. Report learning curves,
content and syntax separately. Frozen original weights; FP32 trainable parameters
and Adam moments, BF16 training autocast, gradient clip1, cosine/3% warmup,
weight decay0. Native plain completion template and EOS151643; no unused
ChatML tokens. Identical prompts and decoding across each comparison. Generation
is greedy with a predeclared task length budget; save raw tokens and cap rates.

Do not select tasks according to A-LoRA gains. Pilot results can diagnose an
unlearnable/noisy target or implementation error; all attempted tasks and results
remain in the record. Model-generated calls are parsed as data, never executed
against external services.

## Stage B: controlled adapter pilot and confirmation

Following Stage A, freeze a separate protocol for Base, input-only A-LoRA,
output-only A-LoRA, hidden LoRA, hidden+input+output A-LoRA, and a hidden-only
control spending the same total parameter budget. An output-only stacking arm
may be specified before evaluation as a mechanistic secondary comparison.
Boundary r16/alpha128, input bias only; input/output maps independent. Compare
the stacking arm to BOTH ordinary hidden LoRA and the matched-budget control.
Standalone effectiveness does not establish stacking effectiveness.

All corresponding hidden initialization, example orders, optimization budgets,
and hyperparameter-search budgets are paired. Use validation content metrics
for selection; loss/EOS metrics are diagnostics. Formal confirmation uses two
Base models and five paired training seeds; no best-seed selection. BFCL single-
turn external evaluation needs its own pinned category specification, overlap
audit, format compatibility check and official scorer before execution; a local
ToolACE holdout must not be called a BFCL result.

## Scoring and audit requirements

Tool calls: whole-call semantic exact match, correct function selection and
argument values/types, and syntax/schema validity separately. Preserve call
multiplicity and define ordering before evaluation. Never salvage by reading
the gold label. Entity extraction: typed-span micro F1 and typed-text micro F1,
plus JSON/schema validity, with explicit duplicate/invalid-span handling.
All rows, including invalid/capped generations, contribute to denominators.

Preflight: zero-residual equivalence, original-weight freeze, optimizer/gradient
allowlists, independently checked causal loss masking, standard HF shifted-loss
agreement, correct EOS supervision, save/reload equivalence. Recompute scores
independently from stored responses. Record source hashes, processed row IDs,
data exclusions, token lengths, training settings, checkpoint hashes, and failures.

Report seed variation and task-item/cluster sampling separately. Formal main
comparisons and multiplicity correction must be fixed before final test scores.
Changing a pilot protocol requires a new version and preserves prior outputs.
