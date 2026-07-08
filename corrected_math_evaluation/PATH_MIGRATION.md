# corrected_math_evaluation path migration

Organized at: 2026-07-08T11:00:59+08:00

Initial organization moved experiment artifacts without deleting them. A later cleanup removed regenerable caches, stale process files, outer wrapper logs, and one explicitly aborted invalid run; see `CLEANUP_MANIFEST.md`.

| old path | new path | description |
| --- | --- | --- |
| `evaluate_math_full.py` | `shared/evaluators/evaluate_math_full.py` | MATH full greedy evaluator |
| `evaluate_gsm8k_full.py` | `shared/evaluators/evaluate_gsm8k_full.py` | GSM8K full greedy evaluator |
| `evaluate_metamath_loss.py` | `shared/evaluators/evaluate_metamath_loss.py` | MetaMathQA answer-token CE evaluator |
| `merge_shards.py` | `shared/merge/merge_math_shards.py` | MATH shard merger |
| `merge_gsm8k_shards.py` | `shared/merge/merge_gsm8k_shards.py` | GSM8K shard merger |
| `compare_results.py` | `shared/analysis/compare_results.py` | legacy comparison helper |
| `test_evaluator.py` | `shared/tests/test_evaluator.py` | evaluator unit tests |
| `known_contaminated_indices.json` | `shared/data_quality/known_contaminated_math_indices.json` | MATH contamination exclusions |
| `__pycache__` | removed in cleanup | root Python bytecode cache; regenerable |
| `reproducible_afflora_sweep` | `model_families/qwen3/qwen3_8b/lmhead_afflora/reproducible_scale_sweep` | Qwen3-8B rank1 scale sweep with paired seeds |
| `afflora_small_scale_sweep` | `model_families/qwen3/qwen3_8b/lmhead_afflora/rank1_small_scale_sweep` | Qwen3-8B rank1 small-scale follow-up |
| `afflora_rank_scale_sweep` | `model_families/qwen3/qwen3_8b/lmhead_afflora/rank_scale_sweep` | Qwen3-8B rank/scale lm-head AffLoRA sweep |
| `qwen3_close_size_hidden_then_mergeable_sweep` | `model_families/qwen3/close_size/hidden_then_mergeable` | Qwen3-0.6B/1.7B hidden-rank then mergeable sweep |
| `qwen25_05b_hidden_rank_then_mergeable_sweep` | `model_families/qwen25/qwen25_0p5b/hidden_rank_then_mergeable` | Qwen2.5-0.5B hidden-rank then mergeable sweep |
| `small_mergeable_rank_sweep` | `model_families/qwen25/small_models/hidden_mergeable_rank_sweep` | Qwen2.5 0.5B/1.5B hidden+mergeable rank sweep |
| `small_affine_only_rank_sweep` | `model_families/qwen25/small_models/affine_only_rank_sweep` | Qwen2.5 0.5B/1.5B affine-only rank sweep |
| `small_mergeable_math_sweep` | `cross_family/small_models/mergeable_math_scale_sweep` | Mixed Qwen2.5/Qwen3 small-model mergeable scale sweep |
| `outputs` | `legacy_root_evals/outputs` | early root-level evaluation outputs |
| `logs` | `legacy_root_evals/logs` | early root-level logs |
| `RESULTS.md` | `legacy_root_evals/summaries/RESULTS.md` | early corrected evaluator results |
| `MATH8_RERUN_RESULTS.md` | `legacy_root_evals/summaries/MATH8_RERUN_RESULTS.md` | 8-shard MATH rerun notes |
| `LMHEAD_RANK_SWEEP_RESULTS.md` | `legacy_root_evals/summaries/LMHEAD_RANK_SWEEP_RESULTS.md` | legacy lm-head rank sweep summary |
| `HIDDEN_PLUS_MERGEABLE_SMALL_MODEL_COMPARISON.md` | `cross_family/small_models/HIDDEN_PLUS_MERGEABLE_SMALL_MODEL_COMPARISON.md` | small-model hidden+mergeable comparison |
| `README.md` | `legacy_root_evals/summaries/README.legacy.md` | previous root README |
