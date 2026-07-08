# MATH 统一 8-shard rerun 结果

日期：2026-07-02

## 目的

这次 rerun 的目的不是重新训练，而是把 MATH test 的推理口径统一，确认之前的结果是否健康。

之前 `RESULTS.md` 里的 MATH 结果混用了不同 shard 数：base 2 shards、hidden 3 shards、input/lm_head 4 shards、部分模型 8 shards。greedy BF16 推理在不同 batch 形状、不同 padded width 下可能出现小幅差异，所以 sub-pp 级别的 MATH 排名不能直接拿 mixed-shard 结果下结论。

这次所有模型统一为：

- 数据：`data/math/test.jsonl` 全部 5,000 题；
- clean：排除 `known_contaminated_indices.json` 中 5 条已确认 MetaMathQA train 等价/重复题，共 4,995 题；
- evaluator：`corrected_math_evaluation/shared/evaluators/evaluate_math_full.py`；
- merge：`corrected_math_evaluation/shared/merge/merge_math_shards.py`；
- shards：8；
- batch size per shard：64；
- max new tokens：512；
- 解码：greedy；
- 输出目录：`corrected_math_evaluation/outputs/math8_rerun/`；
- 日志目录：`corrected_math_evaluation/logs/math8_rerun/`。

## 健康检查

已完成检查：

- 7 个模型均成功完成 8-shard MATH full eval 并 merge；
- 未发现 `Traceback`、`OOM`、`RuntimeError`、`nan`；
- `corrected_math_evaluation/test_evaluator.py` 8 个单测通过；
- 每个 merged JSON 都覆盖 `dataset_index=0..4999`，共 5,000 条，连续且无缺失；
- `py_compile corrected_math_evaluation/*.py` 通过。

结论：这次 eval 产物是健康的；旧 MATH 表的主要问题是评测口径不统一，不是训练 checkpoint 或 merge 本身明显坏掉。

## GSM8K + MATH clean 主表

GSM8K 结果沿用之前完整评测，所有模型均为 8 shards。MATH 使用本次统一 8-shard rerun。

| 模型 | GSM8K 正确/总数 | GSM8K Acc. | Δ vs hidden | MATH clean 正确/总数 | MATH clean Acc. | Δ vs hidden |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B frozen base | 446 / 1319 | 33.8135% | -52.3882 pp | 782 / 4995 | 15.6557% | -34.1742 pp |
| hidden LoRA r8 | 1137 / 1319 | 86.2017% | 0.0000 pp | 2489 / 4995 | 49.8298% | 0.0000 pp |
| AffLoRA input-only r16 | 1145 / 1319 | 86.8082% | +0.6065 pp | 2501 / 4995 | 50.0701% | +0.2402 pp |
| AffLoRA lm_head-only r16 | 651 / 1319 | 49.3556% | -36.8461 pp | 1427 / 4995 | 28.5686% | -21.2613 pp |
| pure AffLoRA input+lm_head r16 | 1143 / 1319 | 86.6566% | +0.4549 pp | 2498 / 4995 | 50.0100% | +0.1802 pp |
| hidden LoRA r8 + lm_head-only r16 | 1133 / 1319 | 85.8984% | -0.3033 pp | 2491 / 4995 | 49.8699% | +0.0400 pp |
| hidden LoRA r8 + AffLoRA r16 | 1127 / 1319 | 85.4435% | -0.7582 pp | 2496 / 4995 | 49.9700% | +0.1401 pp |

## MATH full / clean 详细结果

| 模型 | full 正确/总数 | full Acc. | clean 正确/总数 | clean Acc. | 污染 5 题 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B frozen base | 782 / 5000 | 15.6400% | 782 / 4995 | 15.6557% | 0 / 5 |
| hidden LoRA r8 | 2492 / 5000 | 49.8400% | 2489 / 4995 | 49.8298% | 3 / 5 |
| AffLoRA input-only r16 | 2504 / 5000 | 50.0800% | 2501 / 4995 | 50.0701% | 3 / 5 |
| AffLoRA lm_head-only r16 | 1427 / 5000 | 28.5400% | 1427 / 4995 | 28.5686% | 0 / 5 |
| pure AffLoRA input+lm_head r16 | 2501 / 5000 | 50.0200% | 2498 / 4995 | 50.0100% | 3 / 5 |
| hidden LoRA r8 + lm_head-only r16 | 2494 / 5000 | 49.8800% | 2491 / 4995 | 49.8699% | 3 / 5 |
| hidden LoRA r8 + AffLoRA r16 | 2499 / 5000 | 49.9800% | 2496 / 4995 | 49.9700% | 3 / 5 |

## hidden LoRA 配对对比：MATH clean

| 对比模型 - hidden | delta | 新模型对 / hidden 错 | hidden 对 / 新模型错 | bootstrap 95% CI | bootstrap P(delta > 0) |
| --- | ---: | ---: | ---: | ---: | ---: |
| AffLoRA input-only r16 | +0.2402 pp | 340 | 328 | [-0.7808, 1.2613] pp | 0.6690 |
| pure AffLoRA input+lm_head r16 | +0.1802 pp | 310 | 301 | [-0.7608, 1.1411] pp | 0.6357 |
| AffLoRA lm_head-only r16 | -21.2613 pp | 243 | 1305 | [-22.6426, -19.8398] pp | 0.0000 |
| hidden LoRA r8 + lm_head-only r16 | +0.0400 pp | 216 | 214 | [-0.7808, 0.8609] pp | 0.5320 |
| hidden LoRA r8 + AffLoRA r16 | +0.1401 pp | 241 | 234 | [-0.7207, 1.0010] pp | 0.6245 |

对应比较文件：

- `corrected_math_evaluation/outputs/math8_rerun/comparison_hidden_vs_affine_input_ar16.json`
- `corrected_math_evaluation/outputs/math8_rerun/comparison_hidden_vs_affine_input_lm_head_ar16.json`
- `corrected_math_evaluation/outputs/math8_rerun/comparison_hidden_vs_affine_lm_head_ar16.json`
- `corrected_math_evaluation/outputs/math8_rerun/comparison_hidden_vs_hidden_lm_head_ar16.json`
- `corrected_math_evaluation/outputs/math8_rerun/comparison_hidden_vs_afflora_combo.json`

## 与旧 MATH 结果的差异

旧 mixed-shard MATH clean 中，关键数字是：

- hidden：49.2492%；
- input-only：50.3904%，相对 hidden +1.1411 pp；
- hidden + AffLoRA combo：50.1702%，相对 hidden +0.9209 pp。

统一 8-shard 后变为：

- hidden：49.8298%；
- input-only：50.0701%，相对 hidden +0.2402 pp；
- hidden + AffLoRA combo：49.9700%，相对 hidden +0.1401 pp。

因此旧结果里“input-only / combo 在 MATH 上明显优于 hidden”的结论需要降级。更准确的表述是：在统一 MATH eval 下，input-only 仍是数值最高，但和 hidden 的差距很小，bootstrap 区间跨 0；不能把这一次 MATH clean 当成强证据。

## 当前结论

1. 训练与 eval pipeline 没看到结构性错误。修复后的 evaluator、merge 和 8-shard full coverage 都通过检查。
2. MATH 上，除了 lm_head-only 明显失败外，其余几个强模型基本在 49.83%–50.07% 之间，差距都很小。
3. input-only r16 仍是当前“观测最好”的方案：GSM8K 最高，MATH clean 也最高；但 MATH 优势只有 +0.2402 pp，不能按显著提升来讲。
4. input+lm_head r16 在 GSM8K 和 MATH 上也都略高于 hidden，但不如 input-only。
5. hidden LoRA + lm_head-only、hidden LoRA + AffLoRA combo 没有带来清晰收益：MATH 略高，GSM8K 低于 hidden。
6. lm_head-only 单独训练明显不成立，两个测试集都很弱。

补充：上面的 hidden LoRA + lm_head-only 结论只对应 ar16/hr8 单点。后续 rank sweep 显示该方向有明显 rank 敏感性，低/中 rank 在 MATH 上可能更好；详见 `corrected_math_evaluation/LMHEAD_RANK_SWEEP_RESULTS.md`。

如果继续推进，最合理的下一步不是再纠结这一次 MATH 的 0.1–0.2 pp 排名，而是给 input-only、hidden、input+lm_head 至少补 2–3 个 seed，或者固定同一推理设置后扩大到更多数学 benchmark。
