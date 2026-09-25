# Reuse verification: passed

12/12 rank-8 checkpoints approved: both models, hidden-only/output, seeds 42/43/44.

- Frozen corrected training source and SFT data match P0 byte for byte.
- Training arguments, final epoch/step, FP32 finite adapter tensors, parameter counts and paired initialization checked.
- Every checkpoint was independently reloaded for complete 1,000-example dev and test CE evaluation (24 reports; 24,000 evaluated examples).
- All record IDs and supervised token counts match. Maximum aggregate CE difference: **0.0**. Maximum per-example CE difference: **0.0**.
- Checkpoint/config/report SHA-256 values checked before and after reload, and checked again here.
- Only freshly recomputed CE reports enter the new matrix; IFEval responses are generated anew.

Approval: `reuse_audit/APPROVED.json`; hashes and configuration audit: `reuse_audit/preflight.json`.

This verifies checkpoint/protocol compatibility and reproducibility. It does not establish stacking gains; those require the paired matrix results.

The initial new-queue attempt is archived under `protocol_history/superseded_attempt_before_final_scorer` and contributes no new model results. The final protocol adds deterministic scoring plus fixed 539-prompt sensitivity for the two upstream punctuation-checker defects.

Generation stop-rule correction: all prior generations without native `<|im_end|>` stopping are archived in `protocol_history/generation_without_native_stop` and excluded. P0 reuse approval is unchanged. Ten completed new/smoke training checkpoints passed a separate `TRAINING_RETENTION_AUDIT.json` (unchanged training source/data, weight hashes, finite FP32 values/counts, and available CE report validation). All generation is repeated under the corrected stop policy; the incomplete 7B training restarts.
