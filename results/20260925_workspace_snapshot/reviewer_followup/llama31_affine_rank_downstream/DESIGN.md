# Llama-3.1 A-LoRA rank downstream diagnostic

Evaluate the seed-42 hidden-rank-4/output-A-LoRA checkpoints at A ranks
4, 8, 16, and 32 on the complete MATH and GSM8K test sets. Rank 16 and the
hidden-only baseline reuse existing full outputs; ranks 4/8/32 are newly
evaluated. This is a diagnostic rank curve on test metrics, not an untouched
model-selection result. Any selected rank requires new-seed confirmation.
