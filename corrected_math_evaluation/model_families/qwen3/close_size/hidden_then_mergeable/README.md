# Qwen3 close-size hidden-then-mergeable sweep

Purpose:

Run the same small-model structure check on Qwen3 models close to the Qwen2.5
sizes already tested.

Models:

- `Qwen3-0.6B-Base`
- `Qwen3-1.7B-Base`

Stage 1:

- hidden LoRA rank sweep
- ranks: 1, 2, 4, 8, 16
- seed: 42
- train data: `data/metamathqa_40k/train.jsonl`
- eval: MATH clean-4,995 and GSM8K

Stage 2:

- choose hidden ranks from stage 1:
  - best MATH
  - best GSM8K
  - best average
- then sweep mergeable tied input/lm_head AffLoRA ranks.

This follows the current experiment rule: finish the main structural trend first,
then decide whether seed expansion is needed.
