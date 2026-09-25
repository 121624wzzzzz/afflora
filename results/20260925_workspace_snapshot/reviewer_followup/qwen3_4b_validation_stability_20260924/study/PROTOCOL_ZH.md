# Qwen3-4B WikiSQL validation stability follow-up

Diagnostic follow-up after inspecting prior test results; not a new unseen-task claim. Freeze all choices before new test confirmation. No new model families or tasks in this stage.

Reevaluate all 72 previous WikiSQL dev-only checkpoints: affine shared r16/r30, ordinary shared r1/r2; boundary ratios [1/64,1/16,1/4,1,2,4], three original dev training seeds 307100..307102. Check source code, original audit, initial/final checkpoint hashes and original prediction hashes before reuse. No retraining or test evaluation during candidate selection. Equal search budget across families.

Construct 1024 validation rows deterministically from original official WikiSQL dev, excluding IDs and normalized prompt duplicates found in locally prepared WikiSQL datasets. Record scanned file hashes and exclusions; this establishes local prepared-data nonoverlap, not universal absence of any historical exposure. Do not mix old 256 dev into selection. Check official gold execution, native token encoding and length caps. Train and test sets remain identical to prior experiments.

Select LR separately for every fixed architecture by new validation mean over the three seeds; tie favors ratio closest to 1 then smaller. Produce dev-only sensitivity diagnostics and parameter/performance curve for all four configurations. Retain old selections as historical comparison, not override frozen new choices with old test results. After selection freeze, train each architecture with five fresh seeds 607100..607104; train paired H-only five seeds. Five new smoke jobs validate training/evaluation path. Total 102 jobs:72 evaluation-only reuse,25 full fresh fits,5 smoke. Formal fits use same 2048 examples/64steps/H r8 LR2e-4.

Main contrasts affine r16/r30 vs ordinary r1, Holm family2; affine vs H separate family2. Ordinary r2 is capacity diagnostic. Near-parity r30 has156160 boundary parameters vs154496 for ordinaryr1 (+1.077%); r16 has84480,ordinaryr2 has308992. Report total trainable parameters too. No claim of exact parity or reduced overall training cost.

Source checkpoint symlinks are read-only reuse references; no source files are modified. New adapters/predictions stored under /home/wz/experiment_artifacts/qwen3_4b_validation_stability_20260924. Audit decoded tokens/scoring/official SQL for all outputs. Stop dispatch below40GiB artifact free or5GiB workspace free. Negative outcomes never trigger cancellation or extra test-driven LR search.
