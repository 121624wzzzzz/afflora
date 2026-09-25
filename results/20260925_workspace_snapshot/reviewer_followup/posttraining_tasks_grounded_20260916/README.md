# 新后训练任务 V2：有输入依据的工具参数与实体抽取

五种子训练、开发集和 106 个保留集配置全部完成，零失败。22,000 条开发输出与
109,442 条保留集输出均通过重新解码、评分和身份核验；104 次训练（含冒烟）、
110 个参数范围记录与 46,168 条 token 化输入通过核验。

最明确的结果：Qwen2.5-1.5B Base 的实体抽取从普通 LoRA 的 59.17 提高到
叠加组的 62.57 F1，比严格等参数 LoRA 高 3.33 分。两个五种子区间经八项
比较校正后均高于零。其余三组任务/模型的均值也为正，但种子区间跨零。

主入口：[最终解释与完整对照](FINAL_INTERPRETATION_ZH.md)、
[保留集统计](HELDOUT_RESULTS_ZH.md)、[开发集结果](RESULTS_ZH.md)、
[实验方法与边界](METHOD_AND_LIMITS_ZH.md)。

当前实验目录。V1 在 `../posttraining_tasks_20260916` 完整保留，未对 V1
运行任何 A-LoRA 架构比较。V2 修订发生在观察任何 A-LoRA 新任务分数之前。

V1 发现：部分 ToolACE 金标准使用请求中没有的具体日期；直接生成实体字符
下标使跨度分数主要受位置计数影响。V2 工具调用仅保留参数值在用户请求中有
字面依据的正调用；实体目标改为 type/text/occurrence，由程序定位原始跨度。
该映射不读取金标准，原始实体跨度指标不变。全部来源和剔除 ID 已记录。

这是 API 隔离的工具选择/参数复制任务，不覆盖缺参拒绝、日期推理或完整
ToolACE，也不是 BFCL 官方分数。字面支持检查不保证所有语义标注都无误。

每任务 2,048 条训练、200 条开发，两个官方 Base，单轮64步、LR2e-4。
Base/hidden 的可学习性检查后，五种子比较 input、output、hidden、
hidden_both 和精确等参数 hidden_budget。开发阶段完成后，依据预先固定的
[保留集协议](HELDOUT_PROTOCOL.md) 评测 710 条工具调用和 1,343 条实体样本，
不重训或按测试结果选配置。全量训练、独立调参和外部 BFCL 确认不包含在本轮中。

原权重逐文件重新核验，V2 全部适配器从零开始。代码、数据和评分器在 V2
推理前冻结，四组冒烟均通过。内容、JSON、终止、截断和答案/EOS loss 分报。

另有两条固定训练示例的 Base few-shot 诊断，检查提示敏感性；它不能替代
adapter 的相同提示零样本对照。

设计：[DESIGN.md](DESIGN.md)。最终状态：[COMPLETION.json](COMPLETION.json)、
[FINAL_AUDIT.json](FINAL_AUDIT.json)。归档清单：[ARTIFACT_MANIFEST.json](ARTIFACT_MANIFEST.json)。
来源与复用记录包括 `SOURCE_REUSE.json`、`PILOT_REUSE.json`、`HELDOUT_FROZEN.json`
以及各评测目录的 `REUSE.json`。图表提供 PNG、PDF、SVG 三种格式。

归档后使用以下命令进行只读清单校验；不要原地重跑会改写报告的训练、评分或分析脚本：

```bash
/home/wz/anaconda3/envs/torch24/bin/python lora/reviewer_followup/posttraining_tasks_grounded_20260916/verify_seal.py
```
