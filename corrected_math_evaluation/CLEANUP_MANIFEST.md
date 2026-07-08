# corrected_math_evaluation cleanup manifest

Created: 2026-07-08T11:07:45+08:00

Estimated selected size before cleanup: 10.29 GiB

Scope: removed only regenerable caches, stale process files, outer launcher/orchestrator wrapper logs, and one explicitly aborted invalid run. Preserved checkpoints, train/eval logs, state files, evaluation outputs, JSONL/CSV metrics, and analysis documents.

| path | size | reason |
| --- | ---: | --- |
| `corrected_math_evaluation/cross_family/small_models/mergeable_math_scale_sweep/__pycache__` | 40K | Python bytecode cache; safely regenerable |
| `corrected_math_evaluation/cross_family/small_models/mergeable_math_scale_sweep/launch.log` | 4.0K | Outer launcher/orchestrator wrapper log; not training/eval provenance |
| `corrected_math_evaluation/cross_family/small_models/mergeable_math_scale_sweep/launch.pid` | 4.0K | Runtime residue; safely regenerable or stale |
| `corrected_math_evaluation/cross_family/small_models/mergeable_math_scale_sweep/logs/orchestrator.log` | 0 | Outer launcher/orchestrator wrapper log; not training/eval provenance |
| `corrected_math_evaluation/cross_family/small_models/mergeable_math_scale_sweep/monitor.pid` | 4.0K | Runtime residue; safely regenerable or stale |
| `corrected_math_evaluation/cross_family/small_models/mergeable_math_scale_sweep/orchestrator.pid` | 4.0K | Runtime residue; safely regenerable or stale |
| `corrected_math_evaluation/model_families/qwen25/qwen25_0p5b/hidden_rank_then_mergeable/__pycache__` | 84K | Python bytecode cache; safely regenerable |
| `corrected_math_evaluation/model_families/qwen25/qwen25_0p5b/hidden_rank_then_mergeable/hf_cache` | 3.1G | HuggingFace cache; safely regenerable and not part of experiment evidence |
| `corrected_math_evaluation/model_families/qwen25/qwen25_0p5b/hidden_rank_then_mergeable/launch.outer.log` | 0 | Outer launcher/orchestrator wrapper log; not training/eval provenance |
| `corrected_math_evaluation/model_families/qwen25/qwen25_0p5b/hidden_rank_then_mergeable/launch_hr8_ar4_extra_seeds.outer.log` | 0 | Outer launcher/orchestrator wrapper log; not training/eval provenance |
| `corrected_math_evaluation/model_families/qwen25/qwen25_0p5b/hidden_rank_then_mergeable/launch_hr8_ar4_multiseed.outer.log` | 0 | Outer launcher/orchestrator wrapper log; not training/eval provenance |
| `corrected_math_evaluation/model_families/qwen25/qwen25_0p5b/hidden_rank_then_mergeable/launch_hr8_mergeable.outer.log` | 0 | Outer launcher/orchestrator wrapper log; not training/eval provenance |
| `corrected_math_evaluation/model_families/qwen25/small_models/affine_only_rank_sweep/__pycache__` | 36K | Python bytecode cache; safely regenerable |
| `corrected_math_evaluation/model_families/qwen25/small_models/affine_only_rank_sweep/hf_cache` | 3.5G | HuggingFace cache; safely regenerable and not part of experiment evidence |
| `corrected_math_evaluation/model_families/qwen25/small_models/affine_only_rank_sweep/launch.log` | 0 | Outer launcher/orchestrator wrapper log; not training/eval provenance |
| `corrected_math_evaluation/model_families/qwen25/small_models/affine_only_rank_sweep/launch.outer.log` | 4.0K | Outer launcher/orchestrator wrapper log; not training/eval provenance |
| `corrected_math_evaluation/model_families/qwen25/small_models/affine_only_rank_sweep/launch.pid` | 4.0K | Runtime residue; safely regenerable or stale |
| `corrected_math_evaluation/model_families/qwen25/small_models/affine_only_rank_sweep/logs/orchestrator.log` | 0 | Outer launcher/orchestrator wrapper log; not training/eval provenance |
| `corrected_math_evaluation/model_families/qwen25/small_models/affine_only_rank_sweep/monitor.pid` | 4.0K | Runtime residue; safely regenerable or stale |
| `corrected_math_evaluation/model_families/qwen25/small_models/affine_only_rank_sweep/orchestrator.pid` | 4.0K | Runtime residue; safely regenerable or stale |
| `corrected_math_evaluation/model_families/qwen25/small_models/hidden_mergeable_rank_sweep/__pycache__` | 40K | Python bytecode cache; safely regenerable |
| `corrected_math_evaluation/model_families/qwen25/small_models/hidden_mergeable_rank_sweep/launch.log` | 4.0K | Outer launcher/orchestrator wrapper log; not training/eval provenance |
| `corrected_math_evaluation/model_families/qwen25/small_models/hidden_mergeable_rank_sweep/launch.pid` | 4.0K | Runtime residue; safely regenerable or stale |
| `corrected_math_evaluation/model_families/qwen25/small_models/hidden_mergeable_rank_sweep/logs/orchestrator.log` | 0 | Outer launcher/orchestrator wrapper log; not training/eval provenance |
| `corrected_math_evaluation/model_families/qwen25/small_models/hidden_mergeable_rank_sweep/monitor.pid` | 4.0K | Runtime residue; safely regenerable or stale |
| `corrected_math_evaluation/model_families/qwen25/small_models/hidden_mergeable_rank_sweep/orchestrator.pid` | 4.0K | Runtime residue; safely regenerable or stale |
| `corrected_math_evaluation/model_families/qwen3/close_size/hidden_then_mergeable/__pycache__` | 72K | Python bytecode cache; safely regenerable |
| `corrected_math_evaluation/model_families/qwen3/close_size/hidden_then_mergeable/hf_cache` | 3.8G | HuggingFace cache; safely regenerable and not part of experiment evidence |
| `corrected_math_evaluation/model_families/qwen3/close_size/hidden_then_mergeable/launch_hidden_rank.outer.log` | 0 | Outer launcher/orchestrator wrapper log; not training/eval provenance |
| `corrected_math_evaluation/model_families/qwen3/close_size/hidden_then_mergeable/launch_mergeable_rank.outer.log` | 0 | Outer launcher/orchestrator wrapper log; not training/eval provenance |
| `corrected_math_evaluation/model_families/qwen3/close_size/hidden_then_mergeable/launch_targeted_multiseed.outer.log` | 0 | Outer launcher/orchestrator wrapper log; not training/eval provenance |
| `corrected_math_evaluation/model_families/qwen3/qwen3_8b/lmhead_afflora/rank1_small_scale_sweep/__pycache__` | 40K | Python bytecode cache; safely regenerable |
| `corrected_math_evaluation/model_families/qwen3/qwen3_8b/lmhead_afflora/rank1_small_scale_sweep/logs/orchestrator.log` | 0 | Outer launcher/orchestrator wrapper log; not training/eval provenance |
| `corrected_math_evaluation/model_families/qwen3/qwen3_8b/lmhead_afflora/rank1_small_scale_sweep/monitor.pid` | 4.0K | Runtime residue; safely regenerable or stale |
| `corrected_math_evaluation/model_families/qwen3/qwen3_8b/lmhead_afflora/rank1_small_scale_sweep/monitor_manual.pid` | 4.0K | Runtime residue; safely regenerable or stale |
| `corrected_math_evaluation/model_families/qwen3/qwen3_8b/lmhead_afflora/rank1_small_scale_sweep/orchestrator.pid` | 4.0K | Runtime residue; safely regenerable or stale |
| `corrected_math_evaluation/model_families/qwen3/qwen3_8b/lmhead_afflora/rank1_small_scale_sweep/wait_then_launch.log` | 4.0K | Outer launcher/orchestrator wrapper log; not training/eval provenance |
| `corrected_math_evaluation/model_families/qwen3/qwen3_8b/lmhead_afflora/rank1_small_scale_sweep/wait_then_launch.pid` | 4.0K | Runtime residue; safely regenerable or stale |
| `corrected_math_evaluation/model_families/qwen3/qwen3_8b/lmhead_afflora/rank_scale_sweep/__pycache__` | 40K | Python bytecode cache; safely regenerable |
| `corrected_math_evaluation/model_families/qwen3/qwen3_8b/lmhead_afflora/rank_scale_sweep/logs/orchestrator.log` | 0 | Outer launcher/orchestrator wrapper log; not training/eval provenance |
| `corrected_math_evaluation/model_families/qwen3/qwen3_8b/lmhead_afflora/rank_scale_sweep/monitor.pid` | 4.0K | Runtime residue; safely regenerable or stale |
| `corrected_math_evaluation/model_families/qwen3/qwen3_8b/lmhead_afflora/rank_scale_sweep/orchestrator.pid` | 4.0K | Runtime residue; safely regenerable or stale |
| `corrected_math_evaluation/model_families/qwen3/qwen3_8b/lmhead_afflora/rank_scale_sweep/wait_then_launch.log` | 4.0K | Outer launcher/orchestrator wrapper log; not training/eval provenance |
| `corrected_math_evaluation/model_families/qwen3/qwen3_8b/lmhead_afflora/rank_scale_sweep/wait_then_launch.pid` | 4.0K | Runtime residue; safely regenerable or stale |
| `corrected_math_evaluation/model_families/qwen3/qwen3_8b/lmhead_afflora/reproducible_scale_sweep/__pycache__` | 40K | Python bytecode cache; safely regenerable |
| `corrected_math_evaluation/model_families/qwen3/qwen3_8b/lmhead_afflora/reproducible_scale_sweep/aborted_pre_component_reseed_20260703_0236` | 106M | Aborted invalid pre-component-reseed run; superseded by reproducible reruns |
| `corrected_math_evaluation/model_families/qwen3/qwen3_8b/lmhead_afflora/reproducible_scale_sweep/monitor.pid` | 4.0K | Runtime residue; safely regenerable or stale |
| `corrected_math_evaluation/model_families/qwen3/qwen3_8b/lmhead_afflora/reproducible_scale_sweep/monitor_stdout.log` | 0 | Outer launcher/orchestrator wrapper log; not training/eval provenance |
| `corrected_math_evaluation/model_families/qwen3/qwen3_8b/lmhead_afflora/reproducible_scale_sweep/orchestrator.log` | 0 | Outer launcher/orchestrator wrapper log; not training/eval provenance |
| `corrected_math_evaluation/model_families/qwen3/qwen3_8b/lmhead_afflora/reproducible_scale_sweep/orchestrator.pid` | 4.0K | Runtime residue; safely regenerable or stale |
| `corrected_math_evaluation/shared/analysis/__pycache__` | 12K | Python bytecode cache; safely regenerable |
| `corrected_math_evaluation/shared/cache/root_pycache` | 40K | Migrated root Python bytecode cache; safely regenerable |
| `corrected_math_evaluation/shared/evaluators/__pycache__` | 28K | Python bytecode cache; safely regenerable |
| `corrected_math_evaluation/shared/merge/__pycache__` | 12K | Python bytecode cache; safely regenerable |
| `corrected_math_evaluation/shared/tests/__pycache__` | 8.0K | Python bytecode cache; safely regenerable |
