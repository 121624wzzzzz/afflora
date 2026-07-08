# Qwen2.5-0.5B hidden-rank then mergeable sweep

Purpose:

The previous small-model mergeable sweep fixed hidden LoRA at `hr4`. For 0.5B,
that may be the wrong hidden rank. This experiment is staged:

1. Sweep hidden LoRA rank on Qwen2.5-0.5B.
2. Pick the best hidden rank(s).
3. Sweep mergeable tied input/lm_head AffLoRA on top of those hidden ranks.

## Stage 1: hidden LoRA rank sweep

Model:

- Qwen2.5-0.5B-Base

Training:

- data: `data/metamathqa_40k/train.jsonl`
- epochs: 1
- max seq len: 1024
- batch size: 16 per GPU
- lr: 2e-4
- scheduler: cosine
- warmup ratio: 0.03
- dtype: frozen base bf16, trainable adapters fp32

Hidden ranks:

- 1, 2, 4, 8, 16

Seeds:

- 42, 43, 44

Evaluation:

- MATH clean-4,995
- GSM8K

## Stage 2 rule

After stage 1 completes, choose hidden ranks for mergeable AffLoRA follow-up:

- best MATH rank
- best GSM8K rank
- best average rank

If these collapse to one or two ranks, only run those. If MATH and GSM8K sharply
disagree, keep both.

Suggested mergeable AffLoRA ranks for the second stage:

- 1, 2, 4, 8, 16

The key second-stage metric is paired delta against the matching hidden-only
baseline at the same seed and hidden rank.
