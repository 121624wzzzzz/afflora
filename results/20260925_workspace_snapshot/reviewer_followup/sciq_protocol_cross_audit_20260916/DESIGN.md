# Post hoc diagnosis of Base versus post-trained SciQ scores

No training, no adapters, no modification of either sealed source study. This is
diagnostic work on an already inspected test set, not new confirmatory evidence.
All 1,000 canonical rows are evaluated; the previously excluded two ambiguous
gold rows remain excluded in summaries (998). No prompt or metric is selected
by its test accuracy. Report all conditions.

Use the four verified official checkpoints (Qwen3-0.6B Base / post-trained and
Qwen2.5-1.5B Base / Instruct). Within each family, serialize identical prompt text
for both checkpoints using the Instruct tokenizer's chat template, then encode
with each checkpoint's own verified tokenizer. Assert resulting input IDs are
identical across the two checkpoints. Four conditions are fixed before inference:

1. plain_answer: original task text + two newlines + `Answer:` + newline;
   this exactly reproduces the previous Base input.
2. plain_no_cue: original task text + two newlines.
3. chat: original Instruct chat input with thinking disabled where supported;
   this exactly reproduces the previous post-trained input.
4. chat_answer: chat input + `Answer:` + newline, as an explicit assistant prefix.

This is a prompt-sensitivity check, not a claim that chat input is appropriate
for Base or that plain input fully exercises post-training. Chat/no-chat differ
in more than one token; the Qwen3 chat template contains the official empty
thinking block. A subsequent thinking-mode check, if needed, must separately
declare its generation and extraction protocol, never score the first token
of a thinking trace as the final answer.

Independent inference uses AutoModelForCausalLM's full forward interface, left
padding and explicit position IDs, without importing the previous modeling,
batching, scoring or adapter code. Load stored weights in BF16 then promote to
FP32, as in the sealed reference. TF32 disabled; greedy next-token scores.
Check all original predictions and logits against their sealed reference;
report any discrepancies instead of substituting new scores silently.

Report candidate-label accuracy, unrestricted correct-label first-token accuracy,
label validity, per-label prediction frequencies, and literal decoded prompts.
Independently reparse already saved original greedy generations using a leading
letter with a boundary (optional punctuation / explanatory content allowed).
This content measure is descriptive; retain the old strict format score too.
Keep content, format and termination separate. Existing generation cap was 16;
do not interpret a partial continuation as a fully validated free response.

Verify every source manifest and checkpoint/tokenizer file hash before use.
Audit saved rows and archive the new diagnosis separately after completion.
