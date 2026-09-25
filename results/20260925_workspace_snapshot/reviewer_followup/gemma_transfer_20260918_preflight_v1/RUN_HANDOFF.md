# Execution handoff

Status at 2026-09-18 22:05 CST: all 15 model files verified; downloader exited successfully. Scientific freeze SHA256 is `96507ca196d07e3286c797c2c7f7a97383af43afb5ec398be4224493e140dd74`. The one scheduler is live in exec session `31847`, logging to `scheduler.log`. Eight smoke workers were admitted at 22:05:27 CST. Follow `STATE.json`; do not launch another scheduler.

Workspace: `/commondocument/wz/cross_encoder_workspace/im_exp`.
Interpreter: `/home/wz/anaconda3/envs/torch24/bin/python`.
Set `PYTHONDONTWRITEBYTECODE=1`; sealed prior studies must remain unchanged.

At preparation, the verified model download is the only live job. Its resumable downloader is `../download_gemma9_verified_20260918.py`; its external log is `../gemma_transfer_20260918_download.log`. It writes `models.json` and `MODEL_IDENTITY_AUDIT.json` only after all original shard hashes pass.

The tokenizer/data preparation and independent synthetic native-matrix oracle have passed. No scientific full-model run may begin before `freeze.py` succeeds. The frozen scheduler then admits 10 smokes, all 72 development runs, freezes selection, and admits 30 confirmation runs plus two Base evaluations. The scheduler stops new admission on failure while active workers drain. Do not restart it over an existing `STATE.json`; preserve any failed preflight and log a technical revision before creating a fresh execution root. Do not change candidates in response to scores.

After all jobs, `scheduler.py` invokes `analyze.py` and `final_audit.py`. Wait for the scheduler process itself to exit, since the queue-complete marker precedes those two final steps. Add final reporting, then seal and independently verify the finished directory. Keep verification logs outside the sealed directory.

The external read-only snapshot helper is `../monitor_gemma_transfer_20260918.py`. It reports download totals, phase/job counts, current training or scoring progress, and recent failures without importing study code.
