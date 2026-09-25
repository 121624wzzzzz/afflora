# Revised task pilot, version 2

The V1 pilot is retained at ../posttraining_tasks_20260916. No A-LoRA task score
was observed before this revision. Its Stage B scheduler was stopped before it
launched any adapter experiment. V1 identified two task issues: some ToolACE
gold arguments invent an absolute date from an unanchored relative date; asking
for numerical character offsets makes entity scoring depend heavily on offset
generation. V2 is a documented development revision, not a fresh blind test.

ToolACE V2: keep only positive first-turn calls whose every leaf argument value
is explicitly present in the user request (case-insensitive literal strings,
bounded numeric literals, literal true/false). Keep original API-disjoint
partitions. This conservative copy-heavy subset excludes legitimate inferred
arguments as well as unsupported values. It does NOT measure abstention or
general tool use. Whole-call multiset exact match remains primary, with function
selection, schema/JSON validity and termination separately reported. Documented
top-level defaults are normalized; values otherwise follow the frozen scorer.
Literal grounding does not prove the semantic correctness of every annotation.
Report all V1 outcomes; never claim its low baseline is entirely a model deficit.

CLUENER V2: generate type/text/occurrence, where occurrence is the zero-based
index among exact substring occurrences in the input. A deterministic parser
maps this to the original character spans without accessing gold. Score the
same official typed spans using micro F1; typed-text F1 remains secondary.
Keep all original partitions and fixed 2,048/200 pilot IDs. Gold round-trip,
repeated occurrences, invalid types/occurrences and duplicate predictions must
be checked before any new model inference. Do not reuse V1 fitted adapters.

Both tasks use the authenticated Qwen3-0.6B-Base and Qwen2.5-1.5B checkpoints,
fresh adapters, plain completion prompts and native EOS151643. Train one epoch
on 2,048 examples, batch32/micro2, LR2e-4, AdamW FP32 trainable weights and
moments, BF16 autocast, clip1, 3% warmup/cosine, no weight decay. Greedy generation
uses FP32 computation of original BF16 base values, SDPA, TF32 disabled, 512 new
tokens and batch8. Same serialized inputs and decoding across arms. Mask prompt
loss, supervise complete JSON plus native EOS. Audit frozen original weights,
optimizer/grad allowlists, masked/standard-HF loss and save/reload identity.

Stage A: Base and hidden LoRA r8/alpha16/dropout0.05, seed6100; retain step16 and
final step64. Stage B can start only after a recorded task review (TASK_GATE).
Five paired seeds6100..6104, arms hidden, input, output, hidden_both,
hidden_budget. Boundary r16/alpha128, input bias only, separate input/output
maps. Budget control expands q/k ranks at fixed positions with exact total
parameter matching and preserved shared initialization. Reuse only V2 hidden
seed6100 after source/checkpoint/input/order validation.

Main contrasts: hidden_both minus hidden and hidden_budget, both models and
tasks (family8), Bonferroni seed-t intervals and separate conditional paired
item/cluster bootstrap. No method-specific LR search in this bounded pilot;
five-seed stability cannot replace tuned/full-data/held-out-test confirmation.
Keep public test partitions unscored. BFCL external validation remains future
work and must use its separately pinned official categories/scorer, never
relabel this local ToolACE subset as BFCL. All failures and null results stay.
