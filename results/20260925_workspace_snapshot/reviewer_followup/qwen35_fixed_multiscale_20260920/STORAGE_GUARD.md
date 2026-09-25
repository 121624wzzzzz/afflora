# Operational storage guard

Added 2026-09-20 11:09 Asia/Shanghai, outside every frozen study root.
User reaffirmed continuing the existing four-size matrix. No scientific protocol,
worker, installed package, frozen source, data, score, or selection was changed.

Shared-volume free space fell from 262 GiB at 09:26 to 230 GiB at 10:38
and 212 GiB at 11:07. The four current study roots together occupied about
17.5 GiB at 10:38. Much of the volume growth is outside these studies.

Helper: `../guard_qwen35_storage_20260920.py`.
Live tool session: 87623. Log: `../guard_qwen35_storage_20260920.log`.
It checks available space every 30 seconds and records `DISK_GUARD_STATUS.json`.
At less than 100 GiB free it uses the existing scheduler `CONTROL.json`
`pause_admission` interface. This conservative operational reserve allows active
workers to finish while new admissions stop. It is not a model memory requirement
or a guarantee against other workloads filling the disk. It does not signal any
process, delete or relocate files, or change any frozen study file.

On a low-space trigger it preserves other control fields and any existing pause,
writes `DISK_GUARD_TRIGGER.json`, then exits. There is no automatic resume.
It also exits when the existing scheduler records `SCHEDULER_COMPLETE.json`.
Temporary-directory checks verified the pause path, preservation of an existing
pause and unrelated control fields, and completion exit. The live initial check
confirmed state `watching`, no `CONTROL.json`, and admissions unpaused.

Continue checking capacity and guard state. If it triggers, inspect the reason
and resolve capacity before resuming. The user permits deleting the unused Gemma
model directories if needed; no Gemma deletion or other cleanup has occurred.
Do not touch sealed evidence, frozen source, or other workloads. Clear only a
pause owned by this guard after resolving capacity; preserve any independent
pause reason. Restart this helper after resolving a trigger, since it exits on
trigger. Do not rerun the non-resumable scheduler.
