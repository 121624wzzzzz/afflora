# Post hoc diagnosis of the sealed SciQ result

This is explanatory analysis after seeing the completed study, not a new
confirmatory claim. Original training, scores and sealed artifacts remain
unchanged. No new training is planned here. Reused sources, data, checkpoints
and predictions must be checked against the original seal.

1. For all five seeds and three controls, decompose task differences into
corrected versus regressed questions; measure decision disagreement, baseline
margins, consistency across four cyclic option orders and across seeds.
Decompose centered logit changes into a question/semantic-option component
stable across rotations, a global answer-letter preference, and a remaining
order-dependent component. The decomposition is descriptive, not causal.

2. For every trained bilateral checkpoint (two models, five seeds), evaluate
the same four input/output on-off combinations in FP32 on the same 4,000 test
queries. Changing scale and bias_scale to zero removes the corresponding
residual. Hidden LoRA stays at its jointly trained value. Recompute the fully
enabled path and compare with all saved original label logits/predictions.
Primary descriptive summaries use the same998 valid questions; all1000 queries
are retained. Report both canonical and four-rotation metrics, NLL and
unrestricted first-token behavior. These are conditional inference knockouts,
not independently retrained input-only/output-only models; no configuration
is selected from the test results.

3. Inspect the output classifier geometry: the three relative label rows
W_A-W_D, W_B-W_D, W_C-W_D have at most rank3. If these rows are independent,
an output A-LoRA of rank at least3 can express any linear correction of those
three relative scores at a fixed hidden representation. Verify the row rank
and a numerical right-inverse construction. This representability result is
restricted to four-candidate scores: it does not establish optimization,
generalization, full-vocabulary CE expressivity, or exact redundancy with an
internal LoRA through intervening normalization/nonlinearities.

4. Inspect the stored training curves, gradient-norm and clipping histories.
Do not interpret parameter norms or clipping differences alone as causes of
generalization. Additional probes, if needed, are separately documented.

Confidence intervals in this post hoc analysis are nominal descriptive paired
seed intervals. They do not replace the original six-comparison correction.
No outcome can retrospectively change the original success criterion.
