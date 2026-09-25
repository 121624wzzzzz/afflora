# Gemma-2-9B Base: cross-family stacking confirmation

This study adds one model family to the AffLoRA stacking evidence. Its primary question is whether H+E+U improves over ordinary hidden LoRA and an exactly parameter-matched hidden LoRA control on WikiSQL and TREC50.

Completed on 2026-09-19: all 114 jobs and the final audit passed; the scheduler exited normally. Read [the final interpretation](FINAL_INTERPRETATION_ZH.md), [all individual results](ALL_RESULTS.csv), and [development instability diagnostics](DEVELOPMENT_INSTABILITY_ZH.md). No stable stacking advantage was added on Gemma in this study.

| Task (accuracy %) | Base | H, five-seed mean | Exact-budget H | H+E+U |
|---|---:|---:|---:|---:|
| WikiSQL execution | 19.7266 | 83.3496 | 84.0723 | 83.8086 |
| TREC50 | 27.8000 | 89.3600 | 89.1600 | 87.7600 |

All four prespecified corrected training-seed intervals cross zero. Both TREC corrected question-bootstrap intervals are below zero, conditional on these five fitted models. These are different uncertainty targets; neither substitutes for joint training, data, and tuning uncertainty. Negative and uncertain outcomes remain in the report.

Read `PROTOCOL.md` for the fixed design. `SOURCE_REUSE.json` identifies the individually verified inputs copied from sealed prior studies. The new runs do not reuse trained adapters or predictions. These public benchmark splits have already been evaluated on other model families; they are not globally unseen project data.

Technical revision 2 replaces cached TREC scoring with native complete-prefix forwards, checked against native causal-LM loss. The original 2e-4 tolerance is unchanged. V1's failed technical gate, including all ten smoke fits and a zero-update numerical diagnosis, is sealed in sibling `gemma_transfer_20260918_preflight_v1`; no formal development/confirmation run preceded this revision. All ten smokes were repeated and passed with exactly reproduced training trajectories. See `PREFLIGHT_REVISION.json`.

- Official pretrained model: `google/gemma-2-9b`, revision `33c193028431c2fde6c6e51f29e6f17b60cbfac6`.
- Native eager attention, attention/logit soft-capping, embedding scaling, and tokenizer are preserved.
- All arms load the original FP32 checkpoint into BF16; evaluation casts that same rounded frozen base to FP32. This is not evaluation of the original unrounded FP32 weights.
- H: 27,009,024 trainable parameters. H-budget and H+E+U: exactly 27,241,984 each.
- Six identical learning-rate candidates per method/task, two development seeds, then five fresh confirmation seeds after development selection is frozen.
- Finite queue: 10 technical smokes, 72 development fits, 30 confirmation fits, and two zero-fit Base evaluations.
- Four primary paired-seed contrasts, with family-wise intervals; a separate prespecified cluster bootstrap conditions on the five fitted models.

Evidence files:

| File | Meaning |
|---|---|
| `MODEL_IDENTITY_AUDIT.json` | Every model file matches the pinned official identity |
| `DATA_AUDIT.json` | Source, split, tokenizer, target, and scorer checks |
| `MODEL_FORMULA_AUDIT.json` | Independent effective-matrix oracle for native Gemma E/U semantics |
| `CODE_FROZEN.json` | Scientific code/protocol hash gate before any full-model runs |
| `STATE.json`, `EVENTS.jsonl` | Queue status and transitions |
| `PREFLIGHT_GATE.json` | All technical smokes and paired initialization checks passed |
| `SELECTION.json` | Development-only method/task choices, frozen before confirmation |
| `RESULTS.md`, `ANALYSIS.json` | All search scores and five-seed confirmation results |
| `CLUSTER_BOOTSTRAP.json` | Conditional table/question sampling uncertainty |
| `FINAL_AUDIT.json` | Final data/model/code/output and actual tensor checks |
| `FINAL_INTERPRETATION_ZH.md`, `ALL_RESULTS.csv`, `figures/` | Complete interpretation, individual results, and shareable PDF/PNG figures |
| `SEAL.json`, `SEAL_MANIFEST.json` | Final file inventory and hashes; independent verification is logged outside this directory |

Final audit: 86,908 outputs, 68,884 official valid-SQL executions, 7,924 token reencodings, and 55,860 actual shared-initialization tensor comparisons. The 5,700 native candidate-score checks cover all 50 candidates on two prespecified samples per TREC evaluation, rather than every evaluated question. See the final report for precise scope and limitations. The independent seal verification log is `../gemma_transfer_20260918_seal_verify.log`. Once sealed, do not modify this directory.
