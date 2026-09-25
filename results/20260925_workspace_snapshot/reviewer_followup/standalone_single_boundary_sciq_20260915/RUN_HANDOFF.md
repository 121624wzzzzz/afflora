# Single-boundary SciQ study — completed

All 4 smoke, 12 validation tuning, 20 fresh-seed confirmation, and 2 frozen-base reference jobs completed. No experiment sessions remain active. Analysis and final audit passed; the report and figure were reviewed.

User objective: update only one vocabulary boundary and assess actual downstream performance. Each experiment trains input-only OR output-only A-LoRA with all original parameters frozen; no hidden LoRA is installed.

Canonical SciQ candidate accuracy (998 clean questions; adapter means over five fresh seeds):
- Qwen3-0.6B: base 62.525%, input 82.405%, output 78.617%.
- Qwen2.5-1.5B-Instruct: base 89.279%, input 90.180%, output 89.339%.

Qwen3 improves robustly under seed and question uncertainty. Qwen2.5 input improves across all five seeds but its conditional question-bootstrap interval crosses zero. Qwen2.5 output has no reliable content gain. Both unadapted bases have zero strict-format generation accuracy because their responses include option text or punctuation; this is NOT zero answer knowledge. All 20 fitted checkpoints produce valid answer-only format and terminate normally in the evaluated task.

FINAL_AUDIT.json verifies all frozen original weights remained bitwise unchanged in 36 training jobs, optimizer/gradient whitelists, 100,000 candidate rows, 22,000 decoded generation rows, independent base replay, and all 861 prior sealed files. Reused model/data/source files were hash checked. All new adapters were freshly trained.

FINAL_INTERPRETATION_ZH.md and RESULTS.json are the final interpretation and full results. COMPLETION.json records completion; ARTIFACT_MANIFEST.json seals final artifacts. Do not mutate sealed files. Any future experiment belongs in a new directory.

Scope: one scientific multiple-choice task, not universal retention, unrestricted generation, equal-budget superiority, or equivalence to larger hidden LoRA. The earlier larger-budget hidden LoRA remains more accurate. No manuscript edit, commit, or publication was performed.
