# Qwen3.5-2B boundary learning-rate diagnostic

Posthoc diagnostic authorized after inspecting existing development results. No confirm/test evaluation, no replacement of the frozen parent study.

Fixed hidden LR 8e-4; input/output aLoRA LR 2e-4 (ratio 0.25), compared to verified existing ratio 1.0, hidden-only and equal-budget controls at the same hidden LR. WikiSQL only; paired seeds 7601 (previous collapse) and 7600. Exactly two new fits, 2048 training examples and 64 updates each. Initialization, sample order, numerical kernels, rank, alpha, dropout, optimizer, warmup, clipping, decoding and scoring unchanged. Only final checkpoint evaluated on development data. Training implementation already supports boundary_lr_ratio and is copied unchanged. Stage search is retained solely for the existing spec/audit interface; no candidate selection is performed.

Before training: verify parent frozen source, data, model files and all completed parent search score/response/adapter hashes. After each run: existing independent audit plus full adapter initialization and sample-order identity with its ratio-1 paired run, and exact per-group LR schedule audit. Report both seeds regardless of direction; no significance claim from two seeds.

Operationally: two GPU workers at most, choose GPUs with at least 40 GiB free; no other jobs terminated. Abort admission below 80 GiB disk headroom. Partial failures retained, no silent retry. Parent study remains untouched.
