# dev：下游迁移五种子结果

ANLI R1 为固定三标签条件概率准确率；WikiSQL 为官方执行正确率，测试仅固定抽样1024条。两项任务均2048训练样本、共同LR2e-4、单轮64步、固定末步。所有任务保留，无按效果筛选。

|任务|模型|配置|可训练参数|主指标|
|---|---|---|---:|---:|
|anli_r1|qwen3_06b_base|base|0|33.10|
|anli_r1|qwen3_06b_base|hidden|5,046,272|46.20|
|anli_r1|qwen3_06b_base|hidden_budget|5,112,832|46.80|
|anli_r1|qwen3_06b_base|hidden_both|5,112,832|46.44|
|anli_r1|qwen25_15b_base|base|0|36.50|
|anli_r1|qwen25_15b_base|hidden|9,232,384|52.04|
|anli_r1|qwen25_15b_base|hidden_budget|9,332,224|52.18|
|anli_r1|qwen25_15b_base|hidden_both|9,332,224|52.50|
|wikisql|qwen3_06b_base|base|0|5.47|
|wikisql|qwen3_06b_base|hidden|5,046,272|74.84|
|wikisql|qwen3_06b_base|hidden_budget|5,112,832|74.30|
|wikisql|qwen3_06b_base|hidden_both|5,112,832|75.31|
|wikisql|qwen25_15b_base|base|0|10.94|
|wikisql|qwen25_15b_base|hidden|9,232,384|74.38|
|wikisql|qwen25_15b_base|hidden_budget|9,332,224|74.61|
|wikisql|qwen25_15b_base|hidden_both|9,332,224|75.86|

|任务|模型|叠加组相对|平均增量 pp|校正八项比较的95%种子区间|
|---|---|---|---:|---|
|anli_r1|qwen3_06b_base|hidden|+0.240|[-2.008, +2.488]|
|anli_r1|qwen3_06b_base|hidden_budget|-0.360|[-4.045, +3.325]|
|anli_r1|qwen25_15b_base|hidden|+0.460|[-2.077, +2.997]|
|anli_r1|qwen25_15b_base|hidden_budget|+0.320|[-2.491, +3.131]|
|wikisql|qwen3_06b_base|hidden|+0.469|[-2.538, +3.475]|
|wikisql|qwen3_06b_base|hidden_budget|+1.016|[-0.890, +2.921]|
|wikisql|qwen25_15b_base|hidden|+1.484|[-1.791, +4.760]|
|wikisql|qwen25_15b_base|hidden_budget|+1.250|[-2.213, +4.713]|

区间依赖五种子 t 假设；公共数据预训练暴露未知，两个同系列模型、共同学习率和小训练集限制外推。JSON合法率和逻辑形式等诊断见对应 ANALYSIS.json。不得将不显著解释为证明无效。
