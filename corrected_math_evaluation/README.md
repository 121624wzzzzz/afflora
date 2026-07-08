# corrected_math_evaluation

This directory contains repaired MetaMathQA-trained math SFT experiments and their full MATH/GSM8K evaluation artifacts.

## Directory layout

| path | contents |
| --- | --- |
| `shared/` | reusable evaluators, shard mergers, data-quality metadata, tests, and maintenance tools |
| `model_families/qwen3/qwen3_8b/lmhead_afflora/` | Qwen3-8B hidden LoRA + lm-head AffLoRA scale/rank experiments |
| `model_families/qwen3/close_size/hidden_then_mergeable/` | Qwen3-0.6B and Qwen3-1.7B hidden-rank, mergeable-rank, targeted multi-seed, and MetaMath eval-loss results |
| `model_families/qwen25/qwen25_0p5b/hidden_rank_then_mergeable/` | Qwen2.5-0.5B hidden-rank and hr8+mergeable follow-up |
| `model_families/qwen25/small_models/` | Qwen2.5-0.5B/1.5B affine-only and hidden+mergeable sweeps |
| `cross_family/small_models/` | mixed Qwen2.5/Qwen3 small-model comparison sweeps |
| `legacy_root_evals/` | early root-level outputs/logs/summaries kept for traceability |

## Important current result files

- `model_families/qwen3/close_size/hidden_then_mergeable/TARGETED_MULTISEED_ANALYSIS.md`
- `model_families/qwen3/close_size/hidden_then_mergeable/outputs/metamath_eval_loss/`
- `model_families/qwen25/qwen25_0p5b/hidden_rank_then_mergeable/HR8_AR4_ALL5SEEDS_ANALYSIS.md`
- `model_families/qwen25/small_models/hidden_mergeable_rank_sweep/ANALYSIS.md`

## Path migration

See `PATH_MIGRATION.md` for old-to-new path mapping, `ORGANIZATION_MANIFEST.json` for a machine-readable manifest, and `CLEANUP_MANIFEST.md` for the cleanup record.

Note: historical launcher scripts may still contain the old paths in comments or command examples. Use the new directory layout for future work.
