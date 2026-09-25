# Strict output-only equal-budget comparison

This experiment isolates the output side of the corrected Qwen2.5-1.5B SFT
comparison.  It compares an output-only A-LoRA map against output-only direct
vocabulary LoRA while keeping hidden LoRA, data, initialization, optimization,
and boundary dropout matched.

## Exact trainable budget

For hidden size `d = 1536` and vocabulary size `V = 151936`:

- output-only A-LoRA rank 50: `2 * d * 50 = 153600`;
- output-only Vocab-LoRA rank 1: `V + d = 153472`.

The boundary budgets differ by only 128 parameters (0.0834%).  Both methods add
the same hidden LoRA rank 8 adapter with 9,232,384 trainable parameters, making
the total trainable budgets 9,385,984 and 9,385,856 respectively.

Neither output adapter has a bias.  Boundary dropout is zero on both methods,
while the common hidden LoRA dropout remains 0.05.  The base embedding/lm-head
weight remains tied, but each residual is applied only on the output path.

## Phase 1: seed-42 scale selection

The common optimizer learning rate is fixed at `2e-4`.  The LoRA-style
functional scales `alpha / rank` are swept over `{1, 2, 4, 8}` for each
method:

- A-LoRA rank 50 uses alpha `{50, 100, 200, 400}`;
- Vocab-LoRA rank 1 uses alpha `{1, 2, 4, 8}`.

All eight jobs run concurrently on eight GPUs.  Configuration selection uses
only the 1,000-example corrected dev split.  The corrected test split is not
evaluated during selection.

If scale 8 is the best completed point for both methods, extend the symmetric
search by one point at scale 16.  Around scale 8, also test boundary-only
learning-rate multipliers 0.5 and 2 while keeping the common hidden-LoRA
learning rate fixed at `2e-4`.  This adaptive extension is triggered only from
dev results and still does not inspect test or IFEval.

If scale 16 remains better than scale 8, perform one final geometric extension
to scale 32 and test boundary-LR multiplier 2 at scale 16.  Freeze the best dev
configuration after this second extension; do not keep expanding the grid.

## Phase 2: held-out confirmation

After selecting the best dev scale separately for each method, freeze those
scales and train seeds 43 and 44.  Report all three seeds on the untouched
test split with paired per-example bootstrap comparisons.  Then evaluate
matched-generation IFEval for the fixed configurations rather than selecting
on IFEval.
