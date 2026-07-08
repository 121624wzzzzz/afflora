# 修复后完整 MATH test 结果

> 注意：本文保留了早期 mixed-shard 评测结果，其中 MATH 不同模型使用了不同 shard 数。最终判断请优先看新文档 `corrected_math_evaluation/MATH8_RERUN_RESULTS.md`，那里所有 MATH 模型均统一为 8 shards。旧表适合作为历史记录，不应再用于 sub-pp 级别排名。

## 评测口径

- 数据：`data/math/test.jsonl` 全部 5,000 题；
- full：包含全部官方测试题；
- clean：排除 `known_contaminated_indices.json` 中 5 条已确认训练等价题，共 4,995 题；
- 解码：greedy，最多 512 new tokens；
- 修复：按 batch padded width 截取生成内容，使用平衡括号提取最后一个 `\\boxed{...}`；
- 模型：Qwen3-8B frozen base、hidden LoRA r8、AffLoRA input-only r16、AffLoRA lm_head-only r16、pure AffLoRA input+lm_head r16、hidden LoRA r8 + lm_head-only r16、hidden LoRA r8 + AffLoRA r16。

## 结果

### GSM8K + MATH 总表

GSM8K 使用 `data/gsm8k/test.jsonl` 全部 1,319 题；未发现需要像 MATH 那样剔除的已确认污染样本，所以 GSM8K full/clean 等同。MATH clean 排除 5 条已确认 MetaMathQA train 等价/重复题。

| 模型 | GSM8K 正确/总数 | GSM8K Acc. | Δ vs hidden | MATH clean 正确/总数 | MATH clean Acc. | Δ vs hidden |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B frozen base | 446 / 1319 | 33.8135% | -52.3882 pp | 778 / 4995 | 15.5756% | -33.6737 pp |
| hidden LoRA r8 | 1137 / 1319 | 86.2017% | 0.0000 pp | 2460 / 4995 | 49.2492% | 0.0000 pp |
| AffLoRA input-only r16 | 1145 / 1319 | 86.8082% | +0.6065 pp | 2517 / 4995 | 50.3904% | +1.1411 pp |
| AffLoRA lm_head-only r16 | 651 / 1319 | 49.3556% | -36.8461 pp | 1417 / 4995 | 28.3684% | -20.8809 pp |
| pure AffLoRA input+lm_head r16 | 1143 / 1319 | 86.6566% | +0.4549 pp | 2498 / 4995 | 50.0100% | +0.7608 pp |
| hidden LoRA r8 + lm_head-only r16 | 1133 / 1319 | 85.8984% | -0.3033 pp | 2491 / 4995 | 49.8699% | +0.6206 pp |
| hidden LoRA r8 + AffLoRA r16 | 1127 / 1319 | 85.4435% | -0.7582 pp | 2506 / 4995 | 50.1702% | +0.9209 pp |

这个总表更清楚地显示：

- input-only AffLoRA r16 在 GSM8K 和 MATH clean 上都高于 hidden LoRA，是当前最稳的纯 affine ablation；
- input+lm_head r16 也在两个测试集上都高于 hidden LoRA，但幅度小于 input-only；
- hidden LoRA + lm_head-only 在 MATH 上高于 hidden LoRA，但 GSM8K 略低于 hidden LoRA；
- hidden LoRA + AffLoRA 组合版在 MATH 上更高，但 GSM8K 低于 hidden LoRA；
- lm_head-only 明显不成立。

### MATH 详细结果

| 模型 | full 正确/总数 | full Acc. | clean 正确/总数 | clean Acc. | 污染 5 题 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B frozen base | 778 / 5000 | 15.5600% | 778 / 4995 | 15.5756% | 0 / 5 |
| hidden LoRA r8 | 2463 / 5000 | 49.2600% | 2460 / 4995 | 49.2492% | 3 / 5 |
| AffLoRA input-only r16 | 2519 / 5000 | 50.3800% | 2517 / 4995 | 50.3904% | 2 / 5 |
| AffLoRA lm_head-only r16 | 1417 / 5000 | 28.3400% | 1417 / 4995 | 28.3684% | 0 / 5 |
| pure AffLoRA input+lm_head r16 | 2501 / 5000 | 50.0200% | 2498 / 4995 | 50.0100% | 3 / 5 |
| hidden LoRA r8 + lm_head-only r16 | 2494 / 5000 | 49.8800% | 2491 / 4995 | 49.8699% | 3 / 5 |
| hidden LoRA r8 + AffLoRA r16 | 2509 / 5000 | 50.1800% | 2506 / 4995 | 50.1702% | 3 / 5 |

## hidden LoRA vs AffLoRA

| 口径 | AffLoRA - hidden | AffLoRA 对 / hidden 错 | hidden 对 / AffLoRA 错 | bootstrap 95% CI | bootstrap P(delta > 0) |
| --- | ---: | ---: | ---: | ---: | ---: |
| full 5000 | +0.9200 pp | 258 | 212 | [0.0800, 1.8000] pp | 0.9813 |
| clean 4995 | +0.9209 pp | 258 | 212 | [0.0801, 1.7618] pp | 0.9843 |

McNemar exact two-sided p-value（full/clean discordant 都是 258 vs 212）：0.0378。

## hidden LoRA vs pure AffLoRA

这里的 pure AffLoRA 指 `affine_input_lm_head_ar16_s18_sd42`，即只训练 input embedding / lm_head 的 AffLoRA adapter，不包含 hidden LoRA。

| 口径 | pure AffLoRA - hidden | pure 对 / hidden 错 | hidden 对 / pure 错 | bootstrap 95% CI | bootstrap P(delta > 0) |
| --- | ---: | ---: | ---: | ---: | ---: |
| full 5000 | +0.7600 pp | 329 | 291 | [-0.2000, 1.7400] pp | 0.9351 |
| clean 4995 | +0.7608 pp | 328 | 290 | [-0.2002, 1.7217] pp | 0.9373 |

组合版 `hidden LoRA r8 + AffLoRA r16` 相比 pure AffLoRA：

- full：+0.1600 pp；
- clean：+0.1602 pp。

## 单侧 AffLoRA ablation

| 对比 | 口径 | delta | 新模型对 / hidden 错 | hidden 对 / 新模型错 | bootstrap 95% CI | bootstrap P(delta > 0) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| input-only r16 - hidden | full 5000 | +1.1200 pp | 349 | 293 | [0.1600, 2.1200] pp | 0.9887 |
| input-only r16 - hidden | clean 4995 | +1.1411 pp | 349 | 292 | [0.1602, 2.1421] pp | 0.9872 |
| lm_head-only r16 - hidden | full 5000 | -20.9200 pp | 253 | 1299 | [-22.3400, -19.5000] pp | 0.0000 |
| lm_head-only r16 - hidden | clean 4995 | -20.8809 pp | 253 | 1296 | [-22.3023, -19.4595] pp | 0.0000 |

横向看 pure affine 侧：

- input-only r16：clean 50.3904%，当前最高；
- input+lm_head r16：clean 50.0100%，比 input-only 低 0.3804 pp；
- lm_head-only r16：clean 28.3684%，明显不够。

## 污染样本影响

已确认 MetaMathQA train 中有 5 条与 MATH test 等价/重复的题：

`776, 1551, 1983, 2421, 2991`

这 5 条上：

- base：0/5；
- hidden LoRA：3/5；
- AffLoRA input-only：2/5；
- AffLoRA lm_head-only：0/5；
- pure AffLoRA：3/5；
- hidden LoRA + lm_head-only：3/5；
- AffLoRA：3/5。

所以污染样本对本次 hidden vs AffLoRA / pure AffLoRA 的差值没有实质影响。

## 运行信息

- base：2 shards，max shard elapsed 2280.36s；
- hidden：3 shards，max shard elapsed 1684.45s；
- AffLoRA input-only：4 shards，max shard elapsed 1253.21s；
- AffLoRA lm_head-only：4 shards，max shard elapsed 1249.90s；
- pure AffLoRA：8 shards，max shard elapsed 653.85s；
- hidden LoRA + lm_head-only：8 shards，max shard elapsed 719.84s；
- AffLoRA：3 shards，max shard elapsed 1664.58s；
- GSM8K：所有模型均为 8 shards；base 97.22s、hidden 85.28s、input-only 70.77s、lm_head-only 104.99s、input+lm_head 69.79s、hidden+lm_head-only 85.85s、组合版 83.43s；
- batch size per shard：64；
- max new tokens：512；
- 输出目录：`corrected_math_evaluation/outputs/full/`；
- GSM8K 输出目录：`corrected_math_evaluation/outputs/gsm8k/`；
- paired 对比文件：`corrected_math_evaluation/outputs/full/comparison_hidden_vs_afflora.json`。
- hidden vs pure AffLoRA 对比文件：`corrected_math_evaluation/outputs/full/comparison_hidden_vs_pure_afflora_input_lm_head_ar16.json`。
- hidden vs input-only 对比文件：`corrected_math_evaluation/outputs/full/comparison_hidden_vs_affine_input_ar16.json`。
- hidden vs lm_head-only 对比文件：`corrected_math_evaluation/outputs/full/comparison_hidden_vs_affine_lm_head_ar16.json`。

## 当前结论

把 GSM8K 和 MATH 放在一起看，input-only AffLoRA r16 是当前最清晰的方案：GSM8K 86.8082%，MATH clean 50.3904%，两个测试集都高于 hidden LoRA。input+lm_head r16 也双测集高于 hidden，但略低于 input-only。hidden+lm_head-only 与 hidden+input+lm_head 组合版都在 MATH 上高于 hidden，但 GSM8K 低于 hidden；lm_head-only 明显较弱。移除已确认污染的 5 条 MATH test 样本后，主要结论不变。
