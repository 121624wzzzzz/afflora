# Llama-8B WikiSQL learning-rate follow-up

Status: RUNNING. Separate from the sealed original Llama study.

- Frozen design: [PROTOCOL.md](PROTOCOL.md).
- Six candidates per arm, two development seeds; five fresh confirmation seeds.
- Identical 2,048-example training set; new table-disjoint 1,024-example development
  and 2,048-example confirmation sets, with audited source hashes and exclusions.
- Same parameter counts, alpha, clipping, epochs, prompts and evaluation settings.
- Equal search-fit budget for H, matched-budget H, and H+E+U. Different grids are
  explicit; neither global optimality nor equal grid dimensionality is claimed.
- Exact original-setting compatibility reproduction before search admission.
- Confirmation admitted only after immutable development-only selection.

Monitor from the workspace root:

```bash
/home/wz/anaconda3/envs/torch24/bin/python lora/reviewer_followup/monitor_llama_lr_20260918.py
```

The existing scheduler owns the queue; do not start a second scheduler or edit
frozen scientific files. It has a nonblocking lock, explicit phase gates, and a
minimum 64 GiB free-memory admission threshold. It does not terminate other GPU
processes. A failed audit stops new admission and retains all completed artifacts.

Expected outputs: `SELECTION.json`, `ANALYSIS.json`, `RESULTS.md`, `FINAL_AUDIT.json`.
The original results remain in `../llama_transfer_20260918` and are not overwritten.
