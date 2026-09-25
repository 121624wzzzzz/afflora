# Execution handoff

Execution completed on 2026-09-19. Revision-2 scheduler session `27186` exited with code 0 after all 114 jobs, analysis, and final audit passed. No study training/evaluation process remains active. Final audit timestamp: 06:06:44 CST. Selection SHA256: `1779b875680a29417e41fb8d36d852a4a5345e9e081b7e8544b8be6c5a6a8035`. WikiSQL selected LR: H 2e-4, exact-budget H 4e-4, H+E+U 2e-4; TREC50 selected 4e-4 for all three. Read `FINAL_INTERPRETATION_ZH.md` for complete results. Do not restart or alter frozen scientific code. The dated notes below are history, not instructions to repeat completed preparation.

Closeout: the external reporting helper produced the Chinese interpretation, CSV, and PDF/PNG figures; all three global evidence documents were updated. Final audit covers 86,908 outputs, 68,884 official SQL checks, 5,700 designated-sample native candidate checks, 7,924 reencodings, and 55,860 actual initialization tensor comparisons. Final seal status is recorded in `SEAL.json` and `SEAL_MANIFEST.json`; independent verification is recorded outside this directory in `../gemma_transfer_20260918_seal_verify.log`. Do not write into this directory after sealing.

Status at 2026-09-18 22:15 CST: all 15 model files verified; downloader exited successfully. The first scheduler has exited with its technical gate closed (9 smokes passed, 1 score-path tolerance failure, all 72 formal search jobs pending). Its entire root is now sealed as sibling `gemma_transfer_20260918_preflight_v1`, manifest `7f4be4db244a554c8c72fbb15a3dfb95931d8bcce240d62623cde5a23d4da733`. This fresh root contains verified copied inputs and a preflight scoring revision; it must be frozen again before restarting the finite queue.

Update 22:18:51 CST: revision 2 is frozen, SHA256 `670219115fffb7bbe8e49f25b978c9543a2982408c8d20f26f63eaaeb55db207`. The only live scheduler is exec session `27186`, writing `scheduler.log`. Its first eight repeated smokes have started. Do not start a second scheduler. Scoring uses five native full-prefix forwards and independent native loss checks; tolerance remains 2e-4.

Update 22:24:52 CST: all ten repeated smokes and the gate passed. All ten initializations, training orders, step losses and final adapter tensor hashes exactly reproduce V1. All five TREC native likelihood/loss audit errors are zero. Eight of 72 formal development fits have started; confirmation is still closed. Continue monitoring the same scheduler session `27186`.

Workspace: `/commondocument/wz/cross_encoder_workspace/im_exp`.
Interpreter: `/home/wz/anaconda3/envs/torch24/bin/python`.
Set `PYTHONDONTWRITEBYTECODE=1`; sealed prior studies must remain unchanged.

Model download is complete. Its resumable downloader is `../download_gemma9_verified_20260918.py`; its external log is `../gemma_transfer_20260918_download.log`. `models.json` and `MODEL_IDENTITY_AUDIT.json` were written only after all original shard hashes passed.

The tokenizer/data preparation and independent synthetic native-matrix oracle have passed. No scientific full-model run may begin before `freeze.py` succeeds. The frozen scheduler then admits 10 smokes, all 72 development runs, freezes selection, and admits 30 confirmation runs plus two Base evaluations. The scheduler stops new admission on failure while active workers drain. Do not restart it over an existing `STATE.json`; preserve any failed preflight and log a technical revision before creating a fresh execution root. Do not change candidates in response to scores.

After all jobs, `scheduler.py` invokes `analyze.py` and `final_audit.py`. Wait for the scheduler process itself to exit, since the queue-complete marker precedes those two final steps. Add final reporting, then seal and independently verify the finished directory. Keep verification logs outside the sealed directory.

The external read-only snapshot helper is `../monitor_gemma_transfer_20260918.py`. It reports download totals, phase/job counts, current training or scoring progress, and recent failures without importing study code.

Update 2026-09-19 03:46 CST: six confirmation fits/audits passed; eight active, zero failures. External `../diagnose_gemma_development_20260919.py` has produced `DEVELOPMENT_OPTIMIZATION_DIAGNOSTICS.json` and `DEVELOPMENT_INSTABILITY_ZH.md`, covering all 36 TREC development fits only. At LR8e-4 / seed7400 H+E+U predicted code45 for all256 examples (2.734375%); seed7401 scored83.59375%. This is a preserved post-hoc descriptive optimization-sensitivity finding, not a causal mechanism claim. No confirmation scores were inspected for this diagnostic. The external report helper now includes it, and the external sealer requires and independently checks its input hashes/prediction histograms/loss/gradient maxima. Both updated helpers passed AST syntax checks; they still must run after final audit. No frozen scientific code was changed. The diagnostic helper is already copied into provenance.

## 2026-09-19 04:24 CST: WikiSQL complete block inspected

All 15 WikiSQL confirmation fits passed their individual audits; first read-only aggregation of these confirmation scores occurred only after the complete task block was available. Summary/response SHA-256 and 30,720 primary correctness records independently rechecked. H / exact-budget H / H+E+U means: 83.349609375 / 84.072265625 / 83.80859375. HEU-minus-H paired deltas: [-0.48828125, 0.341796875, 0.146484375, 0.927734375, 1.3671875]; mean +0.458984375 pp, prespecified family-4 seed interval [-0.923477748026089, 1.841446498026089]. HEU-minus-budget deltas: [-2.1484375, 0.146484375, -0.244140625, 1.66015625, -0.732421875]; mean -0.263671875 pp, interval [-2.930707382598309, 2.403363632598309]. Communicated as no confirmed increment over exact-budget control, not proof of zero effect. Whole-study final audit still pending. No change to queue, selection, seeds, grid or endpoints. TREC scores not yet inspected. Scheduler session 27186 remains the sole live scheduler.

Presentation helper now explicitly notes that exact parameter equality does not isolate placement from branch scaling/dropout/initialization/bias; no scientific code changed.

## 2026-09-19 05:01 CST: all formal training finished

All 30 confirmation fits reached step 64. 22 complete fit/evaluation audits passed, eight TREC confirmation evaluations remain active; both zero-fit Base evaluations are queued. All 72 development and 10 smoke jobs passed. No failed jobs in revision 2. Sole live scheduler is still exec session 27186; do not duplicate or restart. TREC confirmation scores have not been inspected. Continue polling until the scheduler exits after analyze.py and final_audit.py, then run external report helper, inspect output/figures, update three global evidence docs and study README/handoff, seal, and run the external sealer with --verify. Finish every write before sealing; verification log must be outside the study root.

## 2026-09-19 05:30 CST: all 30 confirmation audits passed

TREC complete 15-fit block first inspected after all individual audits passed. All 7,500 correctness records and summary/response hashes rechecked. H scores [87.8,90.8,89.4,89.6,89.2], mean 89.36; budget H [88.2,89.4,89.4,88.6,90.2], mean 89.16; HEU [88.4,88.2,88.4,84.8,89.0], mean 87.76. HEU-minus-H mean -1.60 pp, family-4 seed CI [-5.7384726983224175,2.5384726983224235]; HEU-minus-budget -1.40 pp, CI [-4.222721721871906,1.4227217218719064]. Both contrasts four negative/one positive. Communicated honestly as no stable stacking advantage on this new family, not proof of zero effect. No changes to protocol/selection/queue. Both Base evaluations continue.

Post-hoc descriptive selected TREC HEU prediction-code counts [42,42,40,42,44]: no single-code collapse. Do not attribute confirmation degradation to the separate unselected LR=8e-4 development collapse. External report helper now derives these counts and states this limitation. All 15 selected TREC training histories read descriptively; no additional significance tests or fits.
