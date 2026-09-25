# Gemma-2-9B Base: cross-family stacking confirmation

This study adds one model family to the AffLoRA stacking evidence. Its primary question is whether H+E+U improves over ordinary hidden LoRA and an exactly parameter-matched hidden LoRA control on WikiSQL and TREC50.

Read `PROTOCOL.md` for the fixed design. `SOURCE_REUSE.json` identifies the individually verified inputs copied from sealed prior studies. The new runs do not reuse trained adapters or predictions. These public benchmark splits have already been evaluated on other model families; they are not globally unseen project data.

- Official pretrained model: `google/gemma-2-9b`, revision `33c193028431c2fde6c6e51f29e6f17b60cbfac6`.
- Native eager attention, attention/logit soft-capping, embedding scaling, and tokenizer are preserved.
- All arms load the original FP32 checkpoint into BF16; evaluation casts that same rounded frozen base to FP32. This is not evaluation of the original unrounded FP32 weights.
- H: 27,009,024 trainable parameters. H-budget and H+E+U: exactly 27,241,984 each.
- Six identical learning-rate candidates per method/task, two development seeds, then five fresh confirmation seeds after development selection is frozen.
- Finite queue: 10 technical smokes, 72 development fits, 30 confirmation fits, and two zero-fit Base evaluations.
- Four primary paired-seed contrasts, with family-wise intervals; a separate prespecified cluster bootstrap conditions on the five fitted models.

Useful files as execution progresses:

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

Until confirmation and final audits finish, development scores are not evidence for the final claim. All prespecified conditions and seeds will be reported, including negative and uncertain results.
