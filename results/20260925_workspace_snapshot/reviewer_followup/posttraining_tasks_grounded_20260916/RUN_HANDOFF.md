# Completed experiment handoff

All training, development evaluation and106heldout configurations completed
without failure. All owned supervisors and subprocesses have exited; sessions
75101,6401and8634wereclosedwith exit0. No remaining experiment or report work
is pending. Final sealing status is recorded in COMPLETION.json and
ARTIFACT_MANIFEST.json; do not run report/training/analysis writers after sealing.

Authoritative interpretation: FINAL_INTERPRETATION_ZH.md. Main results and all
paired effects: HELDOUT_ANALYSIS.json; data/reuse/rescoring: HELDOUT_AUDIT.json,
RESULT_AUDIT.json,PARAMETER_AUDIT.json,FINAL_AUDIT.json. Both plots inspected and
recorded in VISUAL_REVIEW.json. Runtime history retained in RUN_HISTORY.md.

Primary finding: Qwen2.5-1.5B Base entity extraction gains3.392608F1 over ordinary
LoRA and3.325014overexact-budgetLoRA; family8seed-t95%intervals
[2.258720,4.526495]and[2.254037,4.395990]. Other task/model means are positive
but corrected seed intervals crosszero. Input-only improves allfourcases;
output-only declines on0.6Btoolcalls. Preserve all outcomes and stated limits.

104trainingrunsincluding4smokes;110checkpoint scopes;46,168tokenrecords;
22,000developmentresponsesand109,442heldoutresponses independently redecoded
andrescored. Sharedhiddeninitializations/order, exactbudget, originalweights,
optimizerwhitelist, checkpointidentity and frozen source/data checks passed.

Use verify_seal.py for read-only archiveverification. Original model files are
external, pinned inmodels.json and rehashed atsealing. Earlier V1 and previous
SciQ/CMRCarchives remain separate. Unrelated shared-GPU jobs were not changed.
