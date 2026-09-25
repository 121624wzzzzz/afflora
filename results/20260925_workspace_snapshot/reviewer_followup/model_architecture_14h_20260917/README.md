# 14-hour model, task and architecture experiment window

Scientific runs, final integrity audit and reports are complete. Authorized window: 2026-09-17 21:07:10 to 2026-09-18 11:07:10 Asia/Shanghai. Admission stopped at 09:37:10; all admitted workers and per-run auditors drained by 09:50:18. There were 377 new formal fits, 18 Base evaluations and 32 smoke checks, plus 104 verified historical rows. All admitted jobs passed; 125 planned formal jobs were not admitted within the time budget. Source files of both scientific phases were frozen before their first output.

Actual complete coverage: 16/16 core main-control three-seed conditions; 14/16 full nine-arm core conditions; 0/4 additional five-seed untied blocks; 9/16 new-task three-seed conditions. Both untied models have full three-seed architecture coverage on both core tasks. Missing full blocks are not replaced by means over fewer seeds. Final audit rechecked 518,836 new output records, 104,408 token records, 499 available runs in 82 initialization/order groups, and 48 fixed boundary matrix cells. Matrix checks are sampled local identities, not full-vocabulary or end-to-end merged-model evaluation.

Read the [Chinese scientific report](FINAL_INTERPRETATION_ZH.md), [complete-run CSV](ALL_AVAILABLE_TEST_RUNS.csv), [core contrasts](CORE_CONTRASTS.md), [new-task contrasts](NEW_TASK_CONTRASTS.md), [secondary diagnostics](DESCRIPTIVE_DIAGNOSTICS.md), [TREC error accounting](TREC_ERROR_ANALYSIS.md), and [final audit with every omitted job](FINAL_AUDIT.json). Four scientific figures are available in PNG/PDF/SVG under [figures](figures/), with their exact plotted values. The report supports independent low-parameter boundary adaptation and task/model-dependent stacking gains; it does not claim universal improvements or standalone budget optimality.

The immutable archive manifest is produced by `seal_results.py`; `verify_seal.py` independently rehashes its file set. Do not write into this directory after sealing. Verification output must be kept outside the archive.

At 22:36, shared memory pressure had reduced the original seven slots to six. A documented [resource-only amendment](RESOURCE_AMENDMENT_V2.json) introduced size-specific admission on GPUs 0–7, restoring eight concurrent jobs by placing small models in leftover memory. All six active workers retained their original PIDs and continued without restart; their completion required saved COMPLETE metadata and a new successful independent audit, with the non-child exit status explicitly recorded as unknown. That handoff preserved all job specs and priorities.

At 01:09, a [coverage scheduling amendment](COVERAGE_AMENDMENT_V3.json) advanced the 40 still-pending Qwen3-1.7B/4B main three-seed Base/H/budget-H/HEU jobs to priority18 and deferred 52 still-pending additional untied-seed fits to priority35. The capacity review, performed after some core scores were already known, found 158 jobs ahead of medium-model coverage. Its uniform rule uses coverage and runtime, not score ranking; it changes pending admission order, not the prepared experiment matrix or inferential rules. Original priority fields remain archived separately from effective admission priorities. All eight active workers kept their PIDs and progress. Specs, seeds, training, scoring, memory policy and deadlines are unchanged. Each supervisor version and its handoff evidence is hash-frozen; current supervisor log: scheduler_v3.log.

|Coverage|Planned scope|
|---|---|
|Qwen2.5 Base|0.5B, 1.5B, 3B, 7B|
|Qwen3 Base|0.6B, 1.7B, 4B, 8B|
|Core tasks|CLUENER, WikiSQL|
|Core architecture arms|Base, H, budget H, H+E, H+U, H+E+U, E-only, U-only, E+U without H|
|New tasks|TREC fine-grained classification; SQuAD2 contextual QA with abstention|
|New-task arms|Base, H, budget H, H+E+U|
|Seeds|Three paired seeds for breadth; pre-specified five-seed untied core extension|

H = internal LoRA; E = input embedding-side A-LoRA; U = output unembedding-side A-LoRA. Original base weights stay frozen. Standalone arms must contain no internal LoRA. Untied checkpoints are verified at the actual weight-pointer level.

Queue upper bound: 520 new formal jobs (496 fits + 24 Base evaluations), 32 two-step smoke checks; 104 planned core rows reuse previously audited runs. The reuse audit accepted 148 historical runs overall and re-decoded/re-scored 178,348 historical outputs, with only the pre-specified seeds entering paired summaries. The time budget may leave lower-priority blocks unstarted; final coverage must state this explicitly.

See [original priorities and limits](PLAN.md), the two amendments above, [core frozen protocol](core/PROTOCOL.md), [new-task frozen protocol](new_tasks/PROTOCOL.md), [deadline](WINDOW.json), [live state](STATE.json), and scheduler logs. `monitor.py` is read-only. No results-based model/task/seed selection is permitted.

This is exploratory coverage under a common training recipe. TREC is candidate classification; SQuAD2 uses a fixed balanced public-dev subset and reports answerable-only performance separately from abstention. Old selected-task size results remain unchanged in their sealed archives.
