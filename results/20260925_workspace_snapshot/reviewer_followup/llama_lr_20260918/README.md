# Llama-8B WikiSQL learning-rate follow-up

Status: COMPLETE; final audit passed on 2026-09-18. Separate from the sealed original Llama study.

Read [FINAL_INTERPRETATION_ZH.md](FINAL_INTERPRETATION_ZH.md) for the complete
interpretation, [RESULTS.md](RESULTS.md) for all development/confirmation scores,
and [FINAL_AUDIT.json](FINAL_AUDIT.json) for verification counts.

All three arms selected LR=4e-4; H+E+U selected boundary/H ratio=1. Five-seed
confirmation means: H 82.8125%, exact-budget H 82.6172%, H+E+U 83.2715%.
The stacking differences (+0.4590/+0.6543 pp) cross zero in both marginal and
family-5 seed intervals. Tuning improves H+E+U by +1.5430 pp and budget H by
+2.1387 pp, with positive family-5 intervals. Original-LR H+E+U already exceeds
original-LR budget H by +1.2500 pp on this new holdout (corrected seed interval
crosses zero); do not attribute the old/new sign change solely to tuning.

Completed: 69 jobs, 88,160 response audits, 87,297 official SQL checks, two
zero-update mechanism probes, and 20,000 table-cluster bootstrap replicates.
The probes locate different initial gradient concentrations across Llama/Qwen;
they do not establish the causal source of a generalization gain.

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

The scheduler has exited successfully. Do not restart it or edit frozen scientific
files. The preserved v1 preflight is documented in PREFLIGHT_REVISION.json; formal
search ran only after the revised implementation passed exact compatibility.

Completed outputs: `SELECTION.json`, `ANALYSIS.json`, `RESULTS.md`, `FINAL_AUDIT.json`,
`TABLE_BOOTSTRAP.json`, `mechanism_probe/`, and `figures/`.
Seal and subsequent full verification: `SEAL.json`, `SEAL_MANIFEST.json`, and
`../llama_lr_20260918_verification.log`. Do not modify this study after sealing.
The original results remain in `../llama_transfer_20260918` and are not overwritten.
