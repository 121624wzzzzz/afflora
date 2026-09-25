RETIRED INCOMPLETE: all active workers finished normally; all41 completed jobs audited and archived in place. Controller and idle closeout terminated. Do not resume. Continue qwen35_fixed_multiscale_20260920 (four fresh sizes, shared fixed numerical policy). See RETIREMENT.json, PARTIAL_ARCHIVES.json and ../qwen35_numerics_20260920/REPORT_ZH.md.

URGENT CURRENT STATE (2026-09-20 00:05+): New admissions PAUSED via SIGSTOP to scheduler PID 3850523. Do NOT resume blindly. Active workers allowed to finish and all artifacts retained. A cross-process initial loss difference was found despite exact local repeats: 0.8B WikiSQL seed7600 hidden_budget c1 (GPU1) =1.2254207134246826 vs other 7 matched runs =1.2234999686479568; all16 microbatch losses differ. Shared initial H/full budget adapter, base and order hashes match. Same GPU1 later c2 has the usual loss. This suggests process-specific accelerated-kernel choice, not simply GPU identity. Probe does NOT establish root cause yet.

Diagnostic process tool session20080 runs probe_qwen35_kernel_configs_20260920.py on GPU6, --out chunk_o_configs; log qwen35_numerics_probe_chunk_o_20260920.log; results in qwen35_numerics_20260920/chunk_o_configs/. It builds the exact bad run's initial 0.8B budget adapter, performs existing native preflight/local repeats, then changes ONLY process-local Triton chunk_fwd_kernel_o config across its3 official choices, repeats the first32 examples with NO updates, saves losses and autotuner caches, checks all weights unchanged. Shared library/study code untouched. Watch its result. Live standard scheduler STATE is stale while paused; inspect worker COMPLETE.json/progress directly. Scheduler can resume with SIGCONT (keeps subprocess handles intact) only once a scientifically defensible decision is made. Current frozen results must not be overwritten or selectively retried.

New primary-source investigation: local FLA0.4.2 kernels use triton.autotune, e.g. chunk_fwd_kernel_o BK/BV=128/64/32. Local Triton3.2 has only per-process cache, no cache_results support; per-process timing chooses configs. Could produce numerically different BF16 paths while local repeats stay exact. This is a hypothesis until controlled intervention reproduces it. Web primary docs: https://triton-lang.org/main/python-api/generated/triton.autotune.html ; https://github.com/fla-org/flash-linear-attention/blob/main/ENVs.md . Do not upgrade shared libraries. If kernel policy must change, create NEW studies/freeze and retain/archive all original runs, not silently modify frozen code or splice selected retries.

# Active task handoff

User asks to expand Qwen3.5 from 4B to multiple models and keep polling. Added verified official Base checkpoints 0.8B / 2B / 9B via ModelScope. No Gemma deletion was necessary. Do not end the task at launch; continue through audits, seals and analysis.

342 planned new jobs: per size 10 smokes + 72 equal-search fits + 30 five-seed confirmation fits + 2 frozen Base evaluations. WikiSQL/TREC50, H / exact-budget H / H+E+U. Family12 across three NEW sizes. Historical sealed 4B remains unchanged and family4, including its known first-loss numerical caveat. All new non-Base jobs repeat full first-batch forward/backward twice with preserved RNG before any optimizer update; loss <2e-4, gradient RMS<1e-4, actual first loss agrees. Do not change frozen study code/data/protocols or overwrite results.

Runtime: /home/wz/anaconda3/envs/qwen35_t26/bin/python. Always set PYTHONDONTWRITEBYTECODE=1. No modifications to shared environments. No agents spawned or authorized. No active goal. No Git root or relevant AGENTS found. No skills used.

All paths below are under lora/reviewer_followup:
- qwen35_08b_20260919: frozen, 10 smokes passed, formal search active.
- qwen35_2b_20260919: frozen, all 10 smokes passed, formal search admitted at 23:58 Sept19.
- qwen35_9b_20260919: all official files verified, all technical/repeat gates passed, frozen and admitted to smokes.
- qwen35_multiscale_20260919: global plan, scheduler state, eventual combined reports.

Controllers currently running (tool sessions):
- 59054: prepare_qwen35_multiscale_20260919.py has EXITED 0. All three sizes READY. Log qwen35_multiscale_20260919_prepare.log. Technical reservations released; the master queue can use GPUs 1/2/3/4/7 when eligible.
- 82482: schedule_qwen35_multiscale_20260919.py, log qwen35_multiscale_20260919_scheduler.log. One global queue; one worker per eligible GPU >=68GiB free; leave other workloads intact. It admits each size after READY and its 10 smoke audits; keeps paired seeds/settings fixed. Failure stops new admission and drains workers. Do not launch another scheduler.
- 16704 (restarted from idle 50497 to include final documentation updates): close_qwen35_multiscale_20260919.py, log qwen35_multiscale_20260919_close.log. Waits for global completion, then reports/seals/verifies all sizes and writes RESULTS_ZH.md/RESULTS.json/RESULTS.csv in master directory. Helpers prepared but not yet exercised on final results; diagnose any closeout issue without modifying sealed source/results.
- Download sessions 15845 and 51584 exited. Original 9B downloader was explicitly terminated because some ranges exhausted retries, then successfully resumed with longer timeouts and fewer streams. Record in qwen35_9b_20260919/provenance/DOWNLOAD_RESUME.json; both download logs retained.

Compact live read:
PYTHONDONTWRITEBYTECODE=1 /home/wz/anaconda3/envs/qwen35_t26/bin/python lora/reviewer_followup/status_qwen35_multiscale_20260919.py

First 0.8B complete formal fit had 64 steps, unchanged frozen base, exact saved-adapter reload, 1024 audited outputs. Full paired initial losses checked so far are identical across arms and repeated gradients exactly match locally. No claim of whole-trajectory/cross-device determinism. Do not infer benefits from single-seed development scores.

Historical root qwen35_transfer_20260919 is SEALED. Manifest SHA256 35af1951cb22c7c414927a847b3aa7848cc2ce0be052633d6b6b052543f114d9. Never modify or run its write helpers. 4B Base Wiki43.3105/TREC62.6; H84.2383/89.16; exactH84.3945/89.16; HEU83.4082/88.68. Five-seed corrected CIs cross0. Read-only posthoc analysis completed in qwen35_posthoc_analysis_20260919/REPORT_ZH.md: commonLR dev gains shrink after fair tuning; E dominates gradient energy but no causal attribution; only bilateral tested; initial BF16 discrepancies not reproduced/located. No final response to prior 'why' before user steered into more models.

After all jobs: review combined/per-size reports, inspect scientific plots, verify closeout logs, update global evidence docs outside sealed roots, and give a concise Chinese comparison retaining all positive/negative/uncertain findings. Keep polling without ending while running. Meaningful progress commentary about once a minute; waits <=60s.

After closeout succeeds, run plot_qwen35_multiscale_20260919.py (same isolated Python and PYTHONDONTWRITEBYTECODE=1) to render the combined scientific PNG/PDF. Inspect the image before final delivery. It reads only completed sealed sources and appends its link to the unsealed master report. It has not run yet.
