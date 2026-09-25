# Follow-up diagnostic: native Qwen3 thinking-mode answer generation

Declared after observing all Qwen3 cross-prompt results, before generating any
thinking outputs. Descriptive post hoc diagnosis, not a selected replacement
for earlier primary metrics and not an A-LoRA experiment.

Use the same 1,000 canonical questions, unchanged original task instruction and
options, same official post-trained Qwen3-0.6B weights/tokenizer. Enable the native
chat template's thinking mode. Generate unrestricted continuations, allowing
2,048 new tokens. Use the model card's recommended thinking sampling settings:
temperature 0.6, top_p 0.95, top_k 20, min_p 0, repetition_penalty 1; do_sample=True.
FP32, eager attention, cache enabled, no TF32, no adapter. Four deterministic
contiguous shards of 250 rows, batch 16, seeds 20260916 + shard index. This is
one sampled response per question, not a multi-seed estimate. Record exact IDs,
all outputs, stopping tokens, generation lengths and caps. Never count a reasoning
trace's leading token as an answer. Extract only text after the last generated
`</think>` token (151668). Missing closing token is invalid for this analysis.

Primary diagnostic content extraction is the same conservative leading-letter
rule as cross.py: A/B/C/D at the beginning after whitespace, followed by end of
text, whitespace or punctuation. Also report exact single-letter format, ending
rate, and invalid final-answer rate. Do not expand extraction rules after seeing
the results. Exclude only the same two previously fixed ambiguous-gold rows.
Report all caps as capped; do not silently rerun them with more budget or treat
missing final answers as proof of absent knowledge.

Comparison to previous non-thinking greedy scores changes both reasoning mode
and decoding policy/budget. It tests whether 62.53% characterizes this model's
available answer-generation performance, not an isolated causal effect of
thinking. This budget is shorter than the model card's general recommendation;
truncation is a limitation and its observed rate must be reported.

Official source consulted before implementation:
https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/README.md
