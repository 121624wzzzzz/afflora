# Banking77 与清洗版 E2E：五种子完整结果

新增两项下游任务均已跑完：两个官方 Base、三种适配方案、各五种子，共60次正式训练、4组Base与4个短程检查。四组任务/模型的叠加均值均超过普通和严格等参数LoRA，但本轮八个预定校正区间全部跨零。因此，本轮增加了小幅正向和任务边界证据，没有新增通过预定严格统计标准的任务条件。

1.5B在两项任务中，五个种子均同时优于两个对照；这值得进一步复现，但不能把方向一致直接替代校正后的区间。0.6B也有正均值，个别种子负向，完整保留。

## 完整测试结果

|任务 / 主指标|模型|Base|普通LoRA|等参数LoRA|LoRA+A-LoRA|
|---|---|---:|---:|---:|---:|
|Banking77 / 准确率%|Qwen3-0.6B Base|8.766|70.890|71.117|71.396|
|Banking77 / 准确率%|Qwen2.5-1.5B Base|20.455|69.266|69.299|69.935|
|E2E / BLEU|Qwen3-0.6B Base|8.898|39.123|39.179|39.357|
|E2E / BLEU|Qwen2.5-1.5B Base|10.107|38.836|38.830|39.270|

适配方案均为五种子均值；Base没有训练随机种子平均。Banking77保留完整3080测试文本，E2E保留全部1847测试MR和4533条去重参考。每任务2048条训练、一轮64步、共同LR2e-4，使用最终检查点，没有任务特定调参。

## 叠加增量与不确定性

|任务 / 模型|对照|平均增量|本轮八项校正95%种子t区间|五种子均正向|
|---|---|---:|---|---|
|banking77 / qwen3_06b_base|hidden|+0.5065|[-1.2904, +2.3034]|否|
|banking77 / qwen3_06b_base|hidden_budget|+0.2792|[-2.1668, +2.7253]|否|
|banking77 / qwen25_15b_base|hidden|+0.6688|[-0.8110, +2.1487]|是|
|banking77 / qwen25_15b_base|hidden_budget|+0.6364|[-0.3532, +1.6259]|是|
|e2e_clean / qwen3_06b_base|hidden|+0.2341|[-0.4931, +0.9613]|否|
|e2e_clean / qwen3_06b_base|hidden_budget|+0.1782|[-0.4916, +0.8481]|否|
|e2e_clean / qwen25_15b_base|hidden|+0.4342|[-0.0848, +0.9532]|是|
|e2e_clean / qwen25_15b_base|hidden_budget|+0.4406|[-0.0071, +0.8882]|是|

Banking77增量单位为百分点，E2E为BLEU分。E2E 1.5B相对等参区间下界约−0.0071，尽管接近零仍必须按跨零处理。区间条件于固定测试集，反映五次训练的配对变异；不是所有任务或充分调参后的总体保证。逐种子分数与开发集结果见 [TEST_RESULTS.md](TEST_RESULTS.md)、[DEV_RESULTS.md](DEV_RESULTS.md)。

## 生成收益的性质

|模型 / 方案|chrF++ ↑|规则总SER% ↓|规则无错误率% ↑|排除重复的规则SER% ↓|
|---|---:|---:|---:|---:|
|qwen3_06b_base / hidden|59.943|2.011|88.479|1.687|
|qwen3_06b_base / hidden_budget|59.964|2.022|88.446|1.681|
|qwen3_06b_base / hidden_both|60.103|1.780|89.594|1.484|
|qwen25_15b_base / hidden|59.793|1.332|92.247|1.108|
|qwen25_15b_base / hidden_budget|59.789|1.324|92.301|1.102|
|qwen25_15b_base / hidden_both|60.076|1.372|92.214|1.070|

0.6B的BLEU、chrF++和规则属性错误均值同时改善；1.5B的BLEU/chrF++改善，而含重复的总SER略差，排除重复后略好。这些是描述性辅助指标，不能据此宣布人工语义质量提升。

清洗数据和规则评分本身有边界：141个测试MR含同属性多值；重复计数修正后约8.6%的参考文本也被规则判为重复。规则使用gold属性消歧，且无法覆盖属性表以外的任意虚构细节。固定样本中，Base虚构的菜单/装修细节确实可能不增加规则added计数。保留原版与单行修正版评分，详见 [METHOD_AND_LIMITS_ZH.md](METHOD_AND_LIMITS_ZH.md)、[QUALITATIVE_REVIEW_NOTES.md](QUALITATIVE_REVIEW_NOTES.md)、[QUALITATIVE_PANEL.md](QUALITATIVE_PANEL.md)。

## 实验正确性与复用

全量复核通过：178,208条预测（含32条短程检查输出）、132次评测、19,328条重编码；68份参数范围和20组三方案配对初始化/顺序核验。公开SacreBLEU语料评分与逐条充分统计量一致；Banking77加速联合概率与独立完整前向一致。60次正式训练均保留原权重哈希、优化器范围、梯度检查和保存重载验证。

可训练参数：0.6B普通LoRA 5,046,272，等参与叠加均5,112,832；1.5B普通LoRA 9,232,384，等参与叠加均9,332,224。全部原始权重冻结，严格等参不是近似预算。

仅复用经前轮封存manifest核验的源码起点和重新哈希的14个官方模型文件，未复用已拟合adapter。数据/代码在首次模型运行前冻结，本轮无失败重跑、无结果驱动的设置变更。训练/开发/测试的规范化文本或MR交叉均为零；原始E2E MR家族也不交叉。

## 对论文主张的影响

当前最强的叠加内容证据仍是前轮1.5B CLUENER与两个模型的WikiSQL；本轮Banking77和E2E适合呈现扩展结果与适用边界，不能写成四组新增显著提升。ANLI未见稳定收益的结果继续保留。合适的主张是：边界A-LoRA在若干固定小数据后训练任务中有超出参数数量的额外收益；其幅度和稳定性依赖任务与模型。

本轮没有单侧A-LoRA或其同预算内部LoRA对照，因此不产生新的standalone优越性结论。单一共同学习率、单种提示、有限训练规模、同系列模型和公开数据预训练暴露继续限制推广。

方案和来源：[PROTOCOL.md](PROTOCOL.md)、[Banking77官方仓库](https://github.com/PolyAI-LDN/task-specific-datasets)、[清洗版E2E官方仓库](https://github.com/tuetschek/e2e-cleaning)。完成与封存入口：[COMPLETION.json](COMPLETION.json)、[FINAL_AUDIT.json](FINAL_AUDIT.json)、[ARTIFACT_MANIFEST.json](ARTIFACT_MANIFEST.json)。
