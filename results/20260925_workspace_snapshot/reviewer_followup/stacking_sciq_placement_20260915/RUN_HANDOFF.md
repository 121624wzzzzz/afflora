# COMPLETE: SciQ placement study

16 tuning runs, 40 fresh confirmation runs, 8 smoke runs all complete. Formal
training/evaluation finished 2026-09-15 17:44:34 Asia/Shanghai. No experiment job
remains active. Independent final audit passed all 64 jobs and 176,000 prediction
rows. Numerical near-tie audit rechecked 21 queries with 0 prediction flips.

No prespecified primary comparison passed. Both-minus-budget accuracy means:
Qwen3 -0.04008 pp, Qwen2.5 +0.08016 pp; corrected intervals cross zero. Four cyclic
option rotations give -0.05511 pp for both model comparisons. Ordinary unadjusted
95% seed intervals and question bootstrap intervals also include zero.

Read FINAL_INTERPRETATION_ZH.md for the full interpretation and unstarted future
experiment suggestions. RESULTS.json contains all frozen statistical analyses.
DESIGN.md, FROZEN_PROTOCOL.json and SELECTION.json retain the original protocol;
PROTOCOL_CLARIFICATIONS.md clarifies raw-data inspection versus test-score
access. Original test scoring was never replaced by post hoc diagnostics.

Figures are in figures/sciq_placement.png and .pdf. FINAL_AUDIT.json records
integrity checks. ARTIFACT_MANIFEST.json seals all final files. Keep this
completed study and every earlier completed experiment immutable; future
studies must use a new directory. No trained adapters from older studies were
reused. Verified source and base-weight reuse is recorded in REUSE_AUDIT.json
and DATA_AUDIT.json.

Python: /home/wz/anaconda3/envs/torch24/bin/python
No goal or persistent monitor was installed. Foreground polling continued
through completion as requested. No paper manuscript was rewritten.
