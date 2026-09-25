# Llama A-LoRA learning-rate screening

After the hidden-rank-4 A-LoRA rank sweep, select the lowest-dev-CE A rank
separately for Llama-3.1 and Llama-3.2. Keep hidden LoRA LR at 2e-4 and scan
the A-LoRA-only LR multiplier over 0.1, 0.25, 0.5, 0.75, and 1.0.

Scale 1.0 reuses the rank-sweep checkpoint and held-out result. The other four
scales per model are newly trained with identical component initialization.
Selection uses only the 499-example MetaMathQA holdout.
