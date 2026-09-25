# COMPLETE — CMRC five-new-seed confirmation

All 30 fresh training runs and full evaluations completed successfully. No formal run failed. Final audit passed on2026-09-15 at16:31:56+08:00. No experiment process remains running.

Design: two models × three arms × seeds1001–1005. Same known public CMRC dev, FP32 eager inference, canonical unique-reference batch1. 30 runs ×570 steps; no old trained adapter reused.

All four fixed primary probability comparisons pass Bonferroni family4 paired-t intervals. Mean probability deltas in percentage points: Qwen3 vs ordinary/budget +0.326672/+0.319153; Qwen2.5 +0.514119/+0.554035. Three corrected interval lower bounds are close tozero; do not describe this as broad or overwhelming evidence.

Generation is mixed: Qwen3 AVG +0.199058/+0.226440pp, intervals cross zero. Qwen2.5 vs ordinary LoRA EM−0.254737pp and AVG−0.112487pp, all five seeds negative; unadjusted intervals below zero. Qwen2.5 macro content CE worsens against both controls in all five seeds. Probability improvement is concentrated in already-correct EM cases. Do not claim universal probability modeling or generation improvement.

FINAL_AUDIT.json independently reconstructed125640 unique public reference probabilities and96570 generated outputs, checked240 exact canonical replays,30 finite FP32 checkpoints,10 paired initialization groups, all108 seed contrasts and2027 source files. Six generation caps, all Qwen3; no Qwen2.5 cap.

Reports: FINAL_INTERPRETATION_ZH.md, RESULTS.json/md, paired_effects.csv. Plots: figures/new_seed_confirmation.png/pdf, visually reviewed. Descriptive diagnostics: GENERATION_PROBABILITY_LINK.json/md and GENERATION_CAP_DIAGNOSTIC.json. Initial smoke tolerance failure and amendment remain in protocol_history; old experiments unchanged.

Final archive: ARTIFACT_MANIFEST.json, produced by seal.py after all output is fixed. Run seal.py without redirecting its stdout to a file inside this folder, because that would change a sealed file. No further training, tasks, sweeps, or seed additions are planned.
