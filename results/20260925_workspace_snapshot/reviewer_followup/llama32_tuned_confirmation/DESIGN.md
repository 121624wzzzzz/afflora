# Llama-3.2 tuned A-LoRA downstream confirmation

- Model: Llama-3.2-3B Base (tied input/output weights).
- Treatment: hidden LoRA rank 4 plus shared tied A-LoRA rank 32.
- Hidden LR: 2e-4; A-LoRA LR: 1e-4 (scale 0.5).
- Seeds: 42, 43, 44.
- Training: MetaMathQA-40K train, one epoch.
- Downstream evaluation: full MATH test (clean metric) and full GSM8K test.
- Baseline: existing same-seed hidden-LoRA-rank-4 outputs.

The seed-42 tuned checkpoint is reused. Seeds 43/44 are newly trained with the
same hidden-LoRA component initialization as their hidden-only baselines.
Training and 16-shard evaluation share one work queue, so seed-42 evaluation
starts while seeds 43/44 train.
