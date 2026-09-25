# Additional diagnostic: label versus end-token objective

Added after the first conditional knockouts showed small accuracy effects and
slightly worse candidate NLL when the output map is enabled. The training loss
is the equally weighted mean of full-vocabulary label CE and EOS CE.
We therefore evaluate the EOS term after the gold answer letter for every
original test query and all ten fixed bilateral checkpoints, in the same four
input/output states. No training or test-selected configuration follows.
This checks whether an unseen gain in termination probability offsets the
observed label-score change. It cannot identify which gradients dominated
earlier training. Combine this term with the already saved full-vocabulary
label NLL to reconstruct the teacher-forced two-token test objective.

Frozen sources/checkpoints are hash-checked again. All adapter tensors must
remain unchanged. A prefix label-logit replay with the appended gold token
checks causal alignment against the original saved scores (tolerance 1e-4).
