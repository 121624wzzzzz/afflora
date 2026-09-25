# Completed learning-rate follow-up — do not restart scheduler

**Completion update, 2026-09-18:** all 69 jobs and final audit passed; scheduler
and both mechanism probes exited successfully. Table bootstrap, report, CSV and
PDF/PNG figures are complete. Selected LR=4e-4 for all arms, boundary ratio=1.
Tuned H/budget H/HEU means: 82.8125/82.6171875/83.271484375. Stacking deltas
+0.458984375/+0.654296875 pp have corrected seed intervals crossing zero.
Tuning gains HEU+1.54296875 and budget H+2.138671875 pp have positive corrected
intervals. Original-LR stacking on this new holdout is already +1.25 pp, with
corrected seed interval crossing zero. Full interpretation supersedes the partial
updates below. The remaining closeout is represented by SEAL.json and the external
verification log; once those exist, no work remains. Never rerun the report script
against a sealed study: it would overwrite the reviewed interpretation.

Everything below is the preserved historical handoff, not an active work queue.

**Latest phase update, 2026-09-18 15:58 CST:** all 36 development fits and audits
passed. Selection frozen at 15:57:57, SHA256
`c5aaf5442d5f086539f9bf07e8dfd261da2c8e76529e3030c436a60e20c099d5`.
H=c2, budget H=c2, HEU=c4: all use LR=4e-4, HEU E/U ratio=1.
Selected dev means: H82.666015625, budget82.763671875, HEU83.59375.
HEU selected seed scores83.49609375/83.69140625. All three selected configs
differ from anchors, so confirmation has25 distinct fits, main total69 jobs.
Confirmation launched; 7 running initially (GPUs1–7),18 pending. Read STATE.json
for live counts. No confirmation score exists at this update. All subsequent
probe/bootstrap/report/seal tasks below remain pending. Exec session46346 remains
the active scheduler. Do not mistake development advantage+.830078pp for a
confirmed holdout advantage.

User asks reasonable experiments and necessary deep analysis of Llama hyperparameters.
Earlier user authorization includes running experiments and continuing to poll without
ending the conversation while work remains. No extra permission needed. No subagents.

Workspace: `/commondocument/wz/cross_encoder_workspace/im_exp`.
Python: `/home/wz/anaconda3/envs/torch24/bin/python`; plain `python` is unavailable.

## Current process and frozen design

- Main scheduler is running in exec session **46346**, started 2026-09-18 14:49 CST.
- V2 CODE_FROZEN SHA256:
  `33de53ce18307aa704f67adc52a531dac9d75aef3330ad3dd87e5ba33ea52bbe`.
- Do not edit frozen files, start another scheduler, change candidates, or select manually.
- Read-only monitor: `../monitor_llama_lr_20260918.py`.
- 6 smoke + 2 compatibility fits have passed. Search: 36 fits; at 15:19 CST,
  15 passed, 8 running, 13 pending. All counts include preflight in STATE.json.
- Scheduler automatically selects only after all search fits and audits pass, then
  runs confirmation, frozen analysis and final audit. Monitor through completion.
- Expect last GPU 0 job to be slower due to an unrelated small GPU process. Do not
  kill other processes. Admission requires 64 GiB free, one of our workers per GPU.
- Fits: Llama-3.1-8B WikiSQL, same 2,048 training examples / 64 updates. Three arms
  H / exact-budget H / H+E+U. H 20,971,520 params; both other arms 21,237,760.
- Each arm 6 candidates x search seeds 7200, 7201. H and budget LR candidates in
  tie order: 2e-4, 1e-4, 4e-4, 5e-5, 3e-4, 8e-4. HEU (H LR, E/U ratio):
  (2e-4,1), (2e-4,.25), (1e-4,1), (1e-4,.25), (4e-4,1), (4e-4,.25).
- Dev 1,024 examples from 854 previously unused tables; confirmation 2,048 from
  1,628 previously unused tables, all excluded against enumerated historical data.
  Official gold checks 5,120; mutation checks 768. Original DB source_split retained.
- Alpha, ranks, global norm=1 clipping, training epochs and FP32 evaluation fixed.
- Selection: largest two-seed dev execution mean per arm; exact tie follows grid order.
- Confirm seeds 7300..7304: tuned H, tuned budget H, tuned HEU, original budget H,
  original HEU; identical selected/anchor configs share explicitly mapped runs.
- Five prespecified paired contrasts, df4 seed t intervals, marginal95 and Bonf5.
  See PROTOCOL.md / design.py / analyze.py. Never compare new-vs-old absolute scores
  as tuning gains because new holdout excludes old tables.

## Preserved preflight failure, fixed and verified

- Initial preflight archived outside current root at `../llama_lr_20260918_preflight_v1`.
- Added GPU diagnostic snapshots changed budget-H's numerical trajectory (first
  gradient difference step2; maximum eventual parameter difference .0037).
  HEU reproduced exactly. No formal search or confirmation happened in v1.
- Two runs of the unchanged original training code (GPUs 1 and 7) reproduced
  initial/final tensors and all64 loss/norm records exactly. Moving diagnostic
  snapshots/norm computations to CPU also reproduced exactly. Specific CUDA
  allocation/kernel cause was not identified; do not invent one.
- V2 uses CPU diagnostics only, retaining the original update/clipping operations.
  All 6 smokes and both full compatibility runs were repeated and passed exactly.
- Full provenance in PREFLIGHT_REVISION.json, COMPATIBILITY.json and archived
  REPRODUCTION_DIAGNOSTIC_RESULT.json / PREFLIGHT_ARCHIVE_MANIFEST.json.
- All old sealed Qwen/Llama studies remain untouched. Always disable bytecode
  when importing their modules (`PYTHONDONTWRITEBYTECODE=1`).

## Completed descriptive mechanism work

- BOUNDARY_WEIGHT_SCALES_POSTHOC.json and BOUNDARY_DISPLACEMENT_POSTHOC.json
  were derived from SHA-verified sealed weights and adapters, training tokens only.
- Native-training-token-weighted embedding RMS: Llama8 .0084884 vs Qwen3-8 .0261936.
- Across original seeds7100..7102, E bias absolute RMS .001199 vs .001215;
  relative to embedding RMS 14.13% vs4.64%; E low-rank displacement3.66% vs3.48%;
  total E displacement15.00% vs6.31%. FP32 map computation, not full-model eval.
- Exactly reproduced original Llama HEU7100: mean per-step gradient-energy shares
  H18.32%, E77.09%, U4.59%;31/64 clipped. Group logs cannot separate E bias vs matrix.
- Lower alpha alone does not lower bias (`bias_scale=1`, alpha/r scales matrix only).
- This is a mechanism hypothesis, not proof of the cause or preferred LR direction.
- Notes: BOUNDARY_SCALE_DIAGNOSIS_ZH.md. Probe sources are external parent scripts.

## Additional work promised; run after main completion

1. Two tiny initial-gradient probes (zero updates, no heldout eval):
   `../probe_boundary_initial_gradients_20260918.py --model llama31_8b_base`
   and `--model qwen3_8b_base`. Use two free GPUs with CUDA_VISIBLE_DEVICES,
   PYTHONDONTWRITEBYTECODE=1, OMP_NUM_THREADS=4, TOKENIZERS_PARALLELISM=false.
   Script requires SCHEDULER_COMPLETE passed, verifies sealed source/model files,
   reproduces original seed7100 first batch loss, separates H/E bias/E matrices/U
   matrices gradients, checks no parameter changes. Outputs mechanism_probe/*.json.
   Plan+script hash in MECHANISM_PROBE_PLAN.json. Single initial batch only.
2. Supplementary table-cluster bootstrap:
   `../bootstrap_llama_lr_tables_20260918.py` (CPU, OPENBLAS_NUM_THREADS=2).
   Requires FINAL_AUDIT passed; 20,000 draws, fixed seed19260918, same five contrasts,
   mean across fixed five fitted models, cluster-resample 1,628 tables. Marginal95
   and Bonf5 percentile intervals. Plan/script hash fixed before selection/confirmation
   in TABLE_BOOTSTRAP_PLAN.json. This describes table sampling, not joint seed+tuning
   uncertainty; primary frozen t analysis remains unchanged.
3. `../report_llama_lr_20260918.py` after FINAL_AUDIT passed: generates CSV, paired
   recovered/regressed cases, fixed-denominator both-valid partition, PDF/PNG figure,
   and FINAL_INTERPRETATION_ZH.md. It leaves a conclusion placeholder that MUST be
   replaced after reviewing actual results, supplemental bootstrap and probes.
4. Inspect figure using view_image. Write accurate Chinese interpretation, preserving
   all negative/uncertain findings. Update README status and global evidence docs
   if appropriate (`lora/docs/RESULTS_SO_FAR.md`, `lora/README.md`, paper-2 evidence).
5. When all workers/audits/probes/reporting are finished, external sealer:
   `../seal_llama_lr_20260918.py` then `--verify`, verification log OUTSIDE root.
   Sealer verifies current frozen files and entire preserved v1 archive, requires
   mechanism outputs/bootstrap and completed interpretation; snapshots external
   reporting/probe/sealer source. Do not modify study after sealing.
6. Finish with self-contained Chinese result summary, actual tuning choices,
   paired effects/uncertainty, limitations, link to report. No unsupported claim
   that architecture is universally effective or hyperparameter changes necessarily
   fix the original negative result.

## Partial development results already communicated (NOT confirmation)

All seed7200 only; search still underway. Execution %:
- H: LR2e-4 81.152344;1e-4 80.664063;4e-4 82.128906;5e-5 79.101563;3e-4 81.445313.
- budget H:2e-4 80.859375;1e-4 80.468750;4e-4 81.445313;5e-5 78.613281;3e-4 81.933594.
- HEU:(2e-4,1)81.933594;(2e-4,.25)82.226563;(1e-4,1)80.957031;
  (1e-4,.25)81.738281;(4e-4,1)83.496094. Remaining candidates/seeds pending.
- At H LR2e-4, HEU ratios1 and.25 both clipped28/64; tail8 loss .056797/.057192.
  Lower LR did not automatically reduce clipping count. H actual update-path length
  was larger with HEU than H alone; do not claim clipping implies a proportionate
  Adam effective-LR reduction.

Old sealed Llama result: WikiSQL8 H80.859375, budget80.924479, HEU80.631510,
three seeds; HEU-budget −.292969 pp, all intervals crossed0. NER gains were positive.
Old study root `../llama_transfer_20260918`, seal manifest SHA256
`a995f154415e13d1a3200ebcb85b377b9ad704c5628d8e2d545e845d58fd7599`.
Old Qwen root `../model_architecture_14h_20260917`, manifest SHA256
`e0c21b5788e013a90b9700fe8dcdf0c55959afb5e35fd1d69d8107f581262e00`.
