# Llama A-LoRA rank screening

## Matrix

- Models: Llama-3.1-8B Base and Llama-3.2-3B Base.
- Hidden rank: 4.
- A-LoRA ranks: 4, 8, 16, 32.
- Seed: 42 screening; alpha/hidden-rank = 2 and alpha/A-LoRA-rank = 1.
- Llama-3.1 uses output/lm_head A-LoRA without output beta.
- Llama-3.2 uses the tied shared input/output adapter with beta acting through
  the input path.

The existing hidden-rank-4/A-LoRA-rank-16 checkpoints are reused. All other
treatments are trained from the same base model and component seed, so the
hidden-LoRA initialization is paired with its hidden-only baseline.

## Selection

The primary screening metric is answer-token CE on the disjoint 499-example
MetaMathQA holdout. Hidden-only losses come from `llama_hidden_rank_sweep`.
MATH/GSM8K test sets are not used to select ranks.

## Scheduling

The scheduler uses one mixed training/evaluation queue. Completed checkpoints
enter held-out evaluation immediately rather than waiting for a phase barrier.
