# Qwen3.5-4B boundary learning-rate diagnostic

Posthoc diagnostic authorized after inspecting existing development results. No confirm/test evaluation, no replacement of the frozen parent study.

Fixed hidden LR 2e-4; input/output aLoRA LR 5e-5 (ratio 0.25), compared to verified existing ratio 1.0, hidden-only and equal-budget controls at the same hidden LR. Both WikiSQL and TREC50; paired seeds 7600/7601. Exactly four new fits, 2048 training examples and 64 updates each. Initialization, sample order, numerical kernels, rank, alpha, dropout, optimizer, warmup, clipping, decoding and scoring unchanged. Only final checkpoint evaluated on development data. Training implementation already supports boundary_lr_ratio and is copied unchanged. Stage search is retained solely for the existing spec/audit interface; no candidate selection is performed.

Before training: verify parent frozen source, data, model files and all completed parent search score/response/adapter hashes. After each run: existing independent audit plus full adapter initialization and sample-order identity with its ratio-1 paired run, and exact per-group LR schedule audit. Report both seeds and both tasks regardless of direction; no significance claim from two seeds.

Operationally: four GPU workers at most, choose GPUs with at least 56 GiB free; no other jobs terminated. Abort admission below 80 GiB disk headroom. Partial failures retained, no silent retry. Parent study remains untouched.
