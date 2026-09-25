# Repaired SciQ task-adaptation protocol

Scope: supervised four-choice science task adaptation, not general instruction
SFT, knowledge acquisition or proof of additive gains. This benchmark has been
inspected repeatedly. This is an explicitly post hoc repair and added control;
it is not an untouched confirmation set. Preserve all old sealed artifacts.

## Comparisons and reuse

Primary strata are the two official pretrained Base checkpoints with their
verified original plain completion prompts and unique native EOS 151643. Compare
unadapted, input-only A-LoRA, output-only A-LoRA, larger-budget all-layer LoRA r8,
and a new small-budget middle-layer q_proj LoRA. Never change checkpoint, prompt,
answer position, mode, tokenizer or decoding budget within an adapter comparison.

Existing Base adapters (five seeds per arm) and previous non-thinking post-trained
adapters are reused only after every sealed file/model/adapter hash is verified.
Their training and selection are unchanged. Rerun canonical candidate scoring
and actual generation for all 54 existing reference/confirmation checkpoints;
write all new results here, never into the old checkpoint directories. Existing
post-trained models form separate secondary strata; do not pool seeds across
stages or interpret their absolute scores as a pure post-training effect.

The prior official Qwen3 thinking diagnostic is contextual only, linked by its
sealed manifest. Do not compare a direct-answer trained adapter to an unadapted
thinking run as if their inference conditions were matched. No claim is made
that the adapters preserve thinking behavior, general capabilities, or additive
benefits on top of hidden LoRA.

## Fixed evaluation

Same cleaned train/validation/test and prior exclusion of test rows 718 and 884.
Canonical test n=998; all 1,000 raw rows retained. No support passage or gold label
in inputs. Candidate labels A/B/C/D use full vocabulary scores at the same
verified answer location. Candidate accuracy is a co-primary content metric.

Actual generation: FP32 evaluation of stored BF16 base values, TF32 disabled,
greedy, native stopping IDs, max_new_tokens=128 for every arm and reference,
batch 16 with left padding. No candidate restriction during generation.
Co-primary generated-answer accuracy uses a fixed deterministic extractor that
does not receive gold labels: unambiguous leading answer letters, a small fixed
set of answer prefixes, or an exact unique option text. Reject multi-letter
ambiguity and explicit letter/option conflicts. Save extraction route/errors.
This parser is finite and may miss paraphrases; report coverage. Do not alter
it in response to this run's test outputs. It does not grade explanations.

Separate secondary measurements: exact single-letter format, exact single-letter
correctness, native termination, missing/unparseable/conflicting answers,
truncation, completed-answer correctness, generated lengths, and answer versus
EOS teacher-forced cross entropy. Content can be correct before termination;
report both content and completed-content metrics rather than silently dropping
capped examples. Keep the legacy 16-token strict score only as historical context.

All baseline/fitted pairs must have identical serialized inputs and generation
settings. Recompute source candidate predictions; report any prediction flips
and fail if an unexplained discrepancy exceeds FP32 tolerance.

## New small-budget control

Only on the two Base strata. Choose the middle decoder layer floor(L/2), q_proj
only, before looking at any new control result. Rank is floor(33*d/(in+out)),
alpha=2*rank, dropout=0.05, no bias. This matches or slightly undercuts input
A-LoRA's 33*d parameters: anticipated Qwen3 rank11 = 33,792; Qwen2.5 rank16 =
49,152 versus 50,688. Audit actual dimensions/counts. Do not claim exact matching
for Qwen2.5 or global superiority over every possible internal placement.

Same data, one epoch/364 optimizer steps, effective batch32/micro8, AdamW,
FP32 trainable weights/moments, BF16 autocast, cosine schedule, 3% warmup, clip1,
weight_decay0, same three LRs [5e-5,2e-4,8e-4]. Select by validation candidate
accuracy, then lower candidate NLL, then lower LR, seed5001. Five confirmation
seeds5002..5006 match the existing Base runs; no best-seed selection or test-based
stopping. Training target remains letter + native EOS, exactly the earlier
controlled classification objective. Run smoke, frozen-weight/gradient/optimizer
audits, standard HF shifted-loss comparison and reload validation first.

## Statistical/reporting boundaries

Report all arms/seeds, means and differences. Primary descriptive family: two
Base models x four adapted arms x two content metrics =16 versus unadapted Base.
Use Bonferroni seed-t intervals for that family and separate paired-question
bootstrap intervals conditioned on the fitted seeds. Efficiency family: input
A-LoRA minus small_q, two models x two metrics =4. These separate families and
post hoc design must be visible. A positive standalone effect does not establish
stacking gains, efficiency superiority, non-inferiority, or general SFT success.

## Required checks and deliverables

Version/hash protocol, parser, inputs and implementation before evaluation.
Test parser boundaries, ambiguities, conflicts, native EOS handling and batches.
Verify gradient and optimizer allowlists; original weights must be bytewise
unchanged. New adapters start at zero residual and reload exactly. Validate
labels against standard causal shifted loss. Independent artifact audit must
redecode every new generation, rescore every saved prediction, verify expected
run counts, validation selection and all source manifests before sealing results.
Provide an active README entry and claim-to-evidence notes so old pure-loss and
strict-format studies cannot silently stand in for content or general SFT evidence.
