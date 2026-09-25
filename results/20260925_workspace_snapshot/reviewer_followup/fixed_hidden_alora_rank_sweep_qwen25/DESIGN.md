# Fixed-hidden A-LoRA rank sweep

This diagnostic tests whether the raw-budget-matched A-LoRA rank 50 endpoint
left the low-rank regime in which historical AffLoRA was strongest.

The frozen hidden-only Qwen2.5-1.5B corrected-SFT seed-42 checkpoint, data
order, optimizer, one-epoch endpoint, FP32 output residual, native frozen base
logits, disabled hidden dropout, no clipping, and TF32 denial are inherited
unchanged from the audited fixed-hidden boundary control.

The only intervention is A-LoRA rank:

`2 / 4 / 8 / 16 / 32 / 50`

All ranks use functional scale 16, hence alpha is `16 * rank`. Ranks 2–32 are
new runs. Rank 50 reuses the already validated endpoint from
`fixed_hidden_boundary_fp32_qwen25`.

Each endpoint is evaluated on:

- the first predeclared 1,000 corrected-SFT training records, as a descriptive
  in-sample diagnostic;
- all 1,000 corrected dev records;
- all 1,000 corrected test records.

Interpretation:

- improving train CE with worsening dev/test CE as rank rises supports
  overfitting;
- worsening train and dev/test CE supports optimization difficulty;
- a flat A-LoRA curve supports the frozen output-subspace bottleneck.

The training subset and dev/test sets have been inspected previously. This is
a mechanism diagnostic, not virgin held-out evidence, and seed 42 alone does
not establish training-seed robustness.
