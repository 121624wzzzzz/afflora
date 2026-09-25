# 新后训练任务：五种子保留集评测

两项任务、两个官方预训练 Base、五个配对种子。每任务仍为 2,048 条训练，统一 LR2e-4、单轮64步；未重训或按测试结果选模型。以下为保留的710条工具调用及1,343条实体抽取 public-dev 结果。没有为各方法独立搜索学习率，不能据此宣称所有调优设置下的优势。

| 任务 | 模型 | 配置 | 主内容分 | 实体文本F1 | JSON可解析率 | 截断率 |
|---|---|---|---:|---:|---:|---:|
| toolace | qwen3_06b_base | base | 31.13 | — | 91.55 | 2.82 |
| toolace | qwen3_06b_base | hidden | 56.31 | — | 96.14 | 1.01 |
| toolace | qwen3_06b_base | input | 39.92 | — | 92.37 | 1.13 |
| toolace | qwen3_06b_base | output | 28.96 | — | 88.23 | 6.00 |
| toolace | qwen3_06b_base | hidden_both | 57.10 | — | 96.31 | 0.99 |
| toolace | qwen3_06b_base | hidden_budget | 56.03 | — | 95.94 | 1.30 |
| toolace | qwen25_15b_base | base | 23.66 | — | 69.30 | 3.66 |
| toolace | qwen25_15b_base | hidden | 61.41 | — | 97.92 | 0.48 |
| toolace | qwen25_15b_base | input | 42.85 | — | 94.14 | 1.41 |
| toolace | qwen25_15b_base | output | 35.13 | — | 92.90 | 2.62 |
| toolace | qwen25_15b_base | hidden_both | 62.99 | — | 97.75 | 0.42 |
| toolace | qwen25_15b_base | hidden_budget | 61.41 | — | 97.86 | 0.56 |
| cluener | qwen3_06b_base | base | 0.59 | 4.95 | 30.08 | 69.40 |
| cluener | qwen3_06b_base | hidden | 60.82 | 61.43 | 99.52 | 0.46 |
| cluener | qwen3_06b_base | input | 29.97 | 30.64 | 99.34 | 0.63 |
| cluener | qwen3_06b_base | output | 18.68 | 19.11 | 95.73 | 3.96 |
| cluener | qwen3_06b_base | hidden_both | 62.52 | 63.18 | 99.46 | 0.51 |
| cluener | qwen3_06b_base | hidden_budget | 60.91 | 61.54 | 99.57 | 0.42 |
| cluener | qwen25_15b_base | base | 0.00 | 0.00 | 98.88 | 2.01 |
| cluener | qwen25_15b_base | hidden | 59.17 | 60.06 | 99.63 | 0.37 |
| cluener | qwen25_15b_base | input | 36.44 | 37.20 | 99.40 | 0.46 |
| cluener | qwen25_15b_base | output | 31.39 | 32.39 | 96.90 | 2.99 |
| cluener | qwen25_15b_base | hidden_both | 62.57 | 63.57 | 99.67 | 0.33 |
| cluener | qwen25_15b_base | hidden_budget | 59.24 | 60.11 | 99.58 | 0.42 |

ToolACE 主分为完整调用集合正确率；本轮仅含参数值可从请求中直接找到的正调用样本，不含无调用/拒绝调用情形，也不是 BFCL 官方得分。CLUENER 主分为实体类别与字符跨度的 micro F1，实体文本 F1 单列。所有格式失败和截断输出保留。

| 任务 | 模型 | 叠加组相对 | 增量 pp | 五种子同时95%区间（8比较） | 条件题目/簇95%区间 |
|---|---|---|---:|---|---|
| toolace | qwen3_06b_base | hidden | +0.789 | [-1.698, +3.275] | [-0.208, +1.786] |
| toolace | qwen3_06b_base | hidden_budget | +1.070 | [-2.008, +4.149] | [+0.114, +2.039] |
| toolace | qwen25_15b_base | hidden | +1.577 | [-1.248, +4.403] | [+0.456, +2.731] |
| toolace | qwen25_15b_base | hidden_budget | +1.577 | [-1.917, +5.072] | [+0.462, +2.713] |
| cluener | qwen3_06b_base | hidden | +1.700 | [-0.589, +3.989] | [+1.128, +2.287] |
| cluener | qwen3_06b_base | hidden_budget | +1.606 | [-1.012, +4.223] | [+1.051, +2.175] |
| cluener | qwen25_15b_base | hidden | +3.393 | [+2.259, +4.526] | [+2.673, +4.128] |
| cluener | qwen25_15b_base | hidden_budget | +3.325 | [+2.254, +4.396] | [+2.602, +4.060] |

所有 hidden 初始化和样本顺序已逐种子核验。叠加组和 hidden_budget 参数严格相等；原始模型参数冻结。JSON率提高本身不证明语义能力提高；实体位置误差、类别/文本误差及工具参数误差必须分别解释。

统计区间描述本轮保留集评测的变异，不能消除公共数据预训练暴露、小训练集或单一学习率的限制。主张稳定叠加收益需同时考虑普通 LoRA 和等参数对照，且不能用单个正向任务覆盖其他负结果。
