# Qwen3.5 四尺寸统一数值设置实验（进行中）

阶段效果见 [持续更新的结果表](stage_review/LATEST_ZH.md)，逐学习率、逐种子分数及输入哈希见 [机器可读记录](stage_review/LATEST.json)。每个任务完整开发网格，以及确认阶段完整配对的前1、3、5个种子均作阶段复盘。开发结果与确认结果分别标注；中途不进行显著性检验、提前停止或改参，最终仍按原定协议审计。

等参退化与训练稳定性的阶段诊断见 [复核报告](../qwen35_budget_review_20260920/REPORT_ZH.md)，包括全部已完成同学习率配对与2B高学习率异常；这部分为事后描述，不改动冻结协议。

模型：0.8B、2B、4B、9B Base。四者均为已下载且按固定官方版本逐文件校验的预训练模型。0.8B/2B/4B共享原生输入输出权重，9B不共享。

每个尺寸运行WikiSQL和TREC50，比较普通LoRA、严格等参数LoRA、LoRA＋双侧aLoRA。每组6个学习率、2个开发种子，选参冻结后5个确认种子。每尺寸114项任务，合计456项；全部16项主要比较统一校正，不按结果筛选模型。只训练适配器，原基座均冻结。

统一固定FLA执行配置和PyTorch数值策略。独立生成全部准入种子的首批参考，换GPU和进程核对后才准入；每次正式更新前必须精确匹配参考。详见各尺寸PROTOCOL.md和NUMERICAL_POLICY_SPEC.json，以及相邻qwen35_numerics_20260920/REPORT_ZH.md。

旧4B完整研究和旧0.8B/2B/9B部分运行全部保留，未复用其训练权重、预测或开发选参。新旧设置不可拼成一套效果结果。

四个尺寸均已通过全部技术准入：各16批训练参考、另一张GPU上的6项方法/任务核对。正式任务与逐项审计正在运行。

状态：STATE.json；进程事件：EVENTS.jsonl；技术准备：各尺寸PREPARATION_STATE.json。父目录status_qwen35_fixed_20260920.py提供只读进度摘要。CONTROL.json中的pause_admission=true仅暂停新任务准入，活动任务仍正常完成并审计。

队列完成后，close_qwen35_fixed_20260920.py自动生成逐尺寸报告、封存并独立复核，再生成本目录RESULTS_ZH.md、RESULTS.json、RESULTS.csv及跨尺寸图。
