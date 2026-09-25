# 新后训练任务：五种子受控预实验

两项任务、两个官方预训练 Base、五个配对种子的预实验已完成。每任务 2,048 条训练、200 条开发，统一 LR2e-4、单轮64步。以下是开发集结果；没有评测保留测试集，也没有为各方法独立搜索学习率。不得作为全量任务或最终确认性结论。

| 任务 | 模型 | 配置 | 主内容分 | 实体文本F1 | JSON可解析率 | 截断率 |
|---|---|---|---:|---:|---:|---:|
| toolace | qwen3_06b_base | base | 29.50 | — | 90.50 | 1.00 |
| toolace | qwen3_06b_base | hidden | 66.00 | — | 95.70 | 1.30 |
| toolace | qwen3_06b_base | input | 43.30 | — | 92.00 | 0.70 |
| toolace | qwen3_06b_base | output | 32.40 | — | 90.60 | 5.30 |
| toolace | qwen3_06b_base | hidden_both | 66.00 | — | 97.00 | 0.40 |
| toolace | qwen3_06b_base | hidden_budget | 66.40 | — | 95.80 | 1.10 |
| toolace | qwen25_15b_base | base | 29.50 | — | 70.00 | 3.00 |
| toolace | qwen25_15b_base | hidden | 67.50 | — | 96.80 | 0.00 |
| toolace | qwen25_15b_base | input | 48.50 | — | 94.40 | 0.60 |
| toolace | qwen25_15b_base | output | 41.90 | — | 92.50 | 2.20 |
| toolace | qwen25_15b_base | hidden_both | 66.80 | — | 96.50 | 0.10 |
| toolace | qwen25_15b_base | hidden_budget | 67.70 | — | 96.70 | 0.00 |
| cluener | qwen3_06b_base | base | 0.49 | 7.33 | 27.00 | 71.50 |
| cluener | qwen3_06b_base | hidden | 61.18 | 61.39 | 100.00 | 0.00 |
| cluener | qwen3_06b_base | input | 32.27 | 32.92 | 99.00 | 0.90 |
| cluener | qwen3_06b_base | output | 21.55 | 22.40 | 96.90 | 2.40 |
| cluener | qwen3_06b_base | hidden_both | 63.58 | 63.78 | 99.90 | 0.10 |
| cluener | qwen3_06b_base | hidden_budget | 61.43 | 61.60 | 100.00 | 0.00 |
| cluener | qwen25_15b_base | base | 0.00 | 0.00 | 99.50 | 2.50 |
| cluener | qwen25_15b_base | hidden | 63.02 | 63.91 | 99.90 | 0.10 |
| cluener | qwen25_15b_base | input | 35.10 | 36.08 | 99.40 | 0.60 |
| cluener | qwen25_15b_base | output | 34.11 | 35.74 | 97.10 | 2.90 |
| cluener | qwen25_15b_base | hidden_both | 65.13 | 65.92 | 99.70 | 0.30 |
| cluener | qwen25_15b_base | hidden_budget | 62.88 | 63.76 | 99.90 | 0.10 |

ToolACE 主分为完整调用集合正确率；本轮仅含参数值可从请求中直接找到的正调用样本，不含无调用/拒绝调用情形，也不是 BFCL 官方得分。CLUENER 主分为实体类别与字符跨度的 micro F1，实体文本 F1 单列。所有格式失败和截断输出保留。

| 任务 | 模型 | 叠加组相对 | 增量 pp | 五种子同时95%区间（8比较） | 条件题目/簇95%区间 |
|---|---|---|---:|---|---|
| toolace | qwen3_06b_base | hidden | +0.000 | [-3.529, +3.529] | [-1.608, +1.592] |
| toolace | qwen3_06b_base | hidden_budget | -0.400 | [-2.663, +1.863] | [-2.211, +1.250] |
| toolace | qwen25_15b_base | hidden | -0.700 | [-2.669, +1.269] | [-2.741, +1.386] |
| toolace | qwen25_15b_base | hidden_budget | -0.900 | [-2.434, +0.634] | [-2.967, +1.206] |
| cluener | qwen3_06b_base | hidden | +2.395 | [-0.782, +5.572] | [+0.636, +4.157] |
| cluener | qwen3_06b_base | hidden_budget | +2.146 | [-1.830, +6.121] | [+0.394, +3.917] |
| cluener | qwen25_15b_base | hidden | +2.107 | [+0.237, +3.976] | [+0.173, +4.044] |
| cluener | qwen25_15b_base | hidden_budget | +2.253 | [+0.646, +3.859] | [+0.295, +4.229] |

所有 hidden 初始化和样本顺序已逐种子核验。叠加组和 hidden_budget 参数严格相等；原始模型参数冻结。JSON率提高本身不证明语义能力提高；实体位置误差、类别/文本误差及工具参数误差必须分别解释。

统计区间描述本轮预实验的变异，不能消除开发集使用、公共数据预训练暴露或单一学习率的限制。主张稳定叠加收益需同时考虑普通 LoRA 和等参数对照，且不能用单个正向任务覆盖其他负结果。
