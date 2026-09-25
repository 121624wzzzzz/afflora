# Preserved first smoke failure and correction

The first Qwen3 small_q smoke training finished but its copied numerical check
failed: selected-logit CE 1.9640251398 versus full-shape CE 1.9640378952, difference
1.27554e-5, exceeding the old hardcoded 1e-5 tolerance. Independent checkpoint
reload reproduced this with maximum logit difference 4.38690e-5. FP64 CE applied
to the two already-computed logit tensors still differs by 1.27112e-5: the source
is the shape-dependent FP32 logits, not different target masks.

Before freezing the repaired protocol or running formal tuning, split the check:
(1) apply both masking schemes to the identical full logits in FP64 and require
agreement <1e-12; (2) retain the strict forward-logit tolerance; (3) bound the CE
difference by twice the observed maximum logit error plus 2e-6 reduction error.
The factor two follows from log-sum-exp and target-logit each being 1-Lipschitz
in the maximum norm. Also check the native HF shifted loss against reference
positions from its own full logits before comparing an optimized forward shape.

Training objective, data, optimizer and updates did not change. Both original
smoke checkpoints/logs and the failure record remain. Repeat both models with
the same smoke seed5000 and names ending attempt2. Only the successful second
attempts are required for freeze. This is a preflight implementation correction,
not a response to held-out performance.
