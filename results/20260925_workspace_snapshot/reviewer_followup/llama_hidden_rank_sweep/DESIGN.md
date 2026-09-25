# Llama hidden-LoRA rank screening

## Goal

Select hidden-LoRA capacity before conditioning an A-LoRA rank sweep. This is a
single-seed screening experiment, not the final multi-seed comparison.

## Fixed setup

- Models: Llama-3.1-8B Base and Llama-3.2-3B Base.
- Training: MetaMathQA-40K train (39,500 examples), one epoch, seed 42.
- Hidden ranks: 1, 2, 4, 8, 16; hidden alpha is always `2 * rank`.
- LR: 2e-4, cosine schedule, 3% warmup, LoRA dropout 0.05.
- Llama-3.1: batch 4, gradient accumulation 4.
- Llama-3.2: batch 8, gradient accumulation 2.
- Existing rank-4 seed-42 checkpoints from `llama_cross_version` are reused.

## Selection metric

Rank is selected using answer-token cross-entropy on the disjoint 499-example
`data/metamathqa_40k/eval.jsonl` holdout. MATH and GSM8K test scores are not
used for rank selection.

## Scheduling

There are eight new training jobs (four per model after reusing rank 4). A
single mixed work queue lets a GPU that finishes training immediately evaluate
any completed checkpoint, without a training-to-evaluation barrier.
