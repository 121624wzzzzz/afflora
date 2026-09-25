# Corrected tied-energy lambda sweep: Qwen3-0.6B, seed 42

This sweep holds the forward-consistent input/tied energy geometry, Qwen3-0.6B,
corrected SFT, seed 42, tied affine rank 16/alpha 128, hidden LoRA rank
8/alpha 16, tau=0.00625, effective batch 16, and all optimizer settings
fixed. It trains lambda 1, 10, and 30 in parallel.

The completed lambda 0 (no energy) and lambda 100 checkpoints are reused as
endpoints. Each new run receives independent reloaded dev and eight-GPU full
held-out CE before any IFEval spending. IFEval is reserved for candidates that
are not dominated in the CE sweep.
