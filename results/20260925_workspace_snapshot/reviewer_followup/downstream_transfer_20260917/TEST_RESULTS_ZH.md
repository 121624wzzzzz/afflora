# test：下游迁移五种子结果

ANLI R1 为固定三标签条件概率准确率；WikiSQL 为官方执行正确率，测试仅固定抽样1024条。两项任务均2048训练样本、共同LR2e-4、单轮64步、固定末步。所有任务保留，无按效果筛选。

|任务|模型|配置|可训练参数|主指标|
|---|---|---|---:|---:|
|anli_r1|qwen3_06b_base|base|0|33.10|
|anli_r1|qwen3_06b_base|hidden|5,046,272|45.64|
|anli_r1|qwen3_06b_base|hidden_budget|5,112,832|46.68|
|anli_r1|qwen3_06b_base|hidden_both|5,112,832|46.54|
|anli_r1|qwen25_15b_base|base|0|39.00|
|anli_r1|qwen25_15b_base|hidden|9,232,384|54.00|
|anli_r1|qwen25_15b_base|hidden_budget|9,332,224|53.78|
|anli_r1|qwen25_15b_base|hidden_both|9,332,224|54.00|
|wikisql|qwen3_06b_base|base|0|5.37|
|wikisql|qwen3_06b_base|hidden|5,046,272|70.21|
|wikisql|qwen3_06b_base|hidden_budget|5,112,832|70.39|
|wikisql|qwen3_06b_base|hidden_both|5,112,832|71.82|
|wikisql|qwen25_15b_base|base|0|7.13|
|wikisql|qwen25_15b_base|hidden|9,232,384|70.25|
|wikisql|qwen25_15b_base|hidden_budget|9,332,224|70.45|
|wikisql|qwen25_15b_base|hidden_both|9,332,224|72.66|

|任务|模型|叠加组相对|平均增量 pp|校正八项比较的95%种子区间|
|---|---|---|---:|---|
|anli_r1|qwen3_06b_base|hidden|+0.900|[-6.791, +8.591]|
|anli_r1|qwen3_06b_base|hidden_budget|-0.140|[-6.014, +5.734]|
|anli_r1|qwen25_15b_base|hidden|+0.000|[-3.302, +3.302]|
|anli_r1|qwen25_15b_base|hidden_budget|+0.220|[-2.930, +3.370]|
|wikisql|qwen3_06b_base|hidden|+1.602|[+0.143, +3.060]|
|wikisql|qwen3_06b_base|hidden_budget|+1.426|[+0.090, +2.762]|
|wikisql|qwen25_15b_base|hidden|+2.402|[+0.395, +4.409]|
|wikisql|qwen25_15b_base|hidden_budget|+2.207|[+0.036, +4.378]|

区间依赖五种子 t 假设；公共数据预训练暴露未知，两个同系列模型、共同学习率和小训练集限制外推。JSON合法率和逻辑形式等诊断见对应 ANALYSIS.json。不得将不显著解释为证明无效。
