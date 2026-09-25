# Base native-completion study — completed and sealed

The user asked to try Base checkpoints as the research starting point for single-boundary adaptation. All work in this study is complete; no study subprocess or tool session remains active.

Official Qwen3-0.6B-Base and Qwen2.5-1.5B were hash-verified. Frozen original weights; separate input-only A-LoRA, output-only A-LoRA, and larger-budget hidden LoRA r8. Six smoke,18 validation tuning,30 fresh confirmation trainings plus2 unadapted Base references completed. Five confirmation seeds5002–5006.

Canonical SciQ candidate accuracy (998 previously fixed clean test questions):
- Qwen3: Base85.070%, input86.653%, output85.030%, hiddenr8 89.579%.
- Qwen2.5: Base89.178%, input91.082%, output88.798%, hiddenr8 92.305%.

Input gains+1.583/+1.904pp pass the predeclared Bonferroni-family12 seed criterion; nominal conditional question-bootstrap intervals also exclude zero. Qwen2.5 seed lower bound+0.067pp and Qwen3 question lower bound+0.180pp are narrow. Output arms lack reliable canonical primary gain; rotation-average point gains are both about+0.426pp, so do not call them universally ineffective. Larger-budget hidden LoRA remains more accurate. Every fitted checkpoint generates valid answer-only format and terminates normally on all1000 questions; strict-generation gains include format learning.

CRITICAL protocol correction: directly copying prior ChatML/im_end targets was structurally unsuitable because im_end has291/267 identical frozen vocabulary rows. The initial standalone_base_sciq_20260916 attempt was stopped during validation tuning, before selection/test, and sealed in ABORTED_MANIFEST.json. Do not mutate it. Native EOS151643 has a unique row. This corrected study uses plain completion prompts and native EOS, all adapters initialized afresh. Prior Instruct results therefore differ in both starting weights and prompt/stop protocol and are contextual, not stage-only causal evidence.

Final audit passed:54 original-weight freeze and optimizer/gradient checks;146000 candidate rows recomputed;32000 generations redecoded; zero near-tie prediction flips; zero generation/candidate shape disagreements;475 prior source-study sealed files rechecked. Independent audit reproduces all54 training shuffles and confirms distinct initial/final adapter states across each five-seed group. Archive audit rechecks220 old aborted-study sealed files. Official model paths intentionally refer to immutable verified model files inside that archived directory.

Read FINAL_INTERPRETATION_ZH.md, RESULTS.json, FINAL_AUDIT.json, INDEPENDENCE_AUDIT.json, ARCHIVE_AUDIT.json and PROTOCOL_DIAGNOSIS_ZH.md. Both figures were visually reviewed; figures/base_content_effects.png/pdf separates content gain from format gain. All frozen code/data hashes rechecked before sealing. No manuscript edit, publication or commit was performed.

Interpretation scope: Base is appropriate for the question of adaptation from pretrained initialization. Current evidence supports small-parameter standalone input-boundary adaptation on this one scientific MC task. It does not establish universal post-training, cross-task retention, equal-budget superiority, or stacking benefits. Future experiments must use a new directory; leave this sealed study unchanged.
