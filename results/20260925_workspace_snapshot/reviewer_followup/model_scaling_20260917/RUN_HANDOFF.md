# V5 model-size extension completed

User request: 扩充一下实验的模型尺寸. All authorized fitting, evaluation and analysis work completed on 2026-09-17. No experiment jobs remain. The scheduler session 40052 exited successfully after 64 formal jobs, four smoke runs, parameter/output audits and the frozen analysis. Final seal status is recorded in COMPLETION.json / FINAL_AUDIT.json / ARTIFACT_MANIFEST.json; verify_seal.py is read-only.

Read FINAL_INTERPRETATION_ZH.md for complete five-seed results and limits, TEST_ANALYSIS.json for exact primary numbers, DEV_RESULTS.md for the auxiliary development results, and METHOD_AND_LIMITS_ZH.md for settings. Two figures were visually reviewed; fixed qualitative panel IDs match all 16 previously reviewed examples. This is assistant analysis, not independent human assessment.

## Outcome

Qwen2.5 Base 3B / 7B, CLUENER and WikiSQL, ordinary LoRA / budget LoRA / stack each have five paired seeds. All four test means favor stack against both controls. CLUENER 3B and 7B and WikiSQL 3B have all five seeds positive against both. WikiSQL 7B final seed is -0.09765625 pp against ordinary and ties budget. Only WikiSQL 3B vs exact-budget LoRA passes the pre-specified eight-comparison corrected interval; none of four new conditions passes both controls. All other seven intervals cross zero. Do not rewrite these as four confirmed positive conditions or proof of no effect.

Budget increments over ordinary are 133120 / 232960 parameters. 3B stack and budget are exactly 15099904; 7B stack 20418048, budget 20418560 (+512). Historical 1.5B anchors were independently reverified, not pooled into this new comparison family. Tasks were selected because of earlier positive results; size trends are descriptive and confounded with width/depth/tying differences and fixed hyperparameters.

## Verified artifacts

90432 new responses, 132 evaluations, 13838 new token records, 68 parameter scopes, 20 new paired initialization/order groups and 10 historical paired groups passed. 40035 valid SQL predictions were cross-checked with the official engine. Historical 37872 outputs and 6919 token records were reverified before reuse. No trained adapter initializes a new model. All 60 formal fits used 2048 examples and 64 steps, frozen original weights, FP32 trainables/optimizer and exact checkpoint reload. No outcome-based stopping, extra seeds or failed-fit reruns.

Frozen data manifest SHA256: 3951b884f72b6fd0135f5ad2ccfe41c88e3a651e933e67bf5780fc6de3feb21b
Frozen code manifest SHA256: 92e3d1374a89e18ea9631b6f8adf319fb2a38e10ea240b344eee841c7865c51e

## Operational history

The original scheduler session 7782 used a near-empty GPU gate. An unrelated eight-GPU job began after our workers. After the original eight formal jobs completed with no active workers, only our own quiescent scheduler was deliberately stopped (exit 143). Completed artifacts were verified in OPERATIONAL_RESUME_CHECK.json before resumption. RESOURCE_AMENDMENT.json / .md records the pre-resumption change: pool 2..6, at least 48 GiB free before launch, one of our workers per GPU. pipeline_shared.py changes resource admission only; original frozen fitting/scoring/statistics are unchanged. No external job was killed or modified. Shared wall-clock is not a method-speed comparison.

## Project documentation

The lora README, docs/RESULTS_SO_FAR.md and paper-2/EXPERIMENT_EVIDENCE_STATUS.md include full new results and boundaries; prior negative and uncertain studies remain intact. Old sealed archives were not modified. No commit, PR or publication was requested. After sealing, do not rerun writers inside this archive or modify archived artifacts.
