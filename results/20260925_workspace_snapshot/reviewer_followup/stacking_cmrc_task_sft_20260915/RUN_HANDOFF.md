# COMPLETE：CMRC 任务内训练及最终分析

完成于 2026-09-15T13:41:42+08:00。24/24 组训练和全部内部验证、CMRC 评测已完成；main_state.json phase=complete，FINAL_AUDIT.json status=passed。主调度器和最终分析/审计进程均已结束，没有待运行实验。用户要求的持续监控已执行到所有训练、评测、统计与审计完成。

both−hidden / both−等预算：Qwen3 +0.46745 / +0.19099 pp；Qwen2.5 −0.08260 / +0.00189 pp。四个主要比较的名义 95% 种子区间均含零。Qwen3 相对 hidden 三种子均正，但相对等预算仍有反转；所有 24 个端点生成截断率均为零。不能声称稳定的跨模型、等预算生成增益。

完整解读在 FINAL_INTERPRETATION_ZH.md，数字与区间在 RESULTS.json/md、paired_effects.csv，图在 figures/cmrc_task_sft.png/.pdf（已查看）。新报告也整理了此前中文迁移、对话 CE 与温度校准证据，给出论文主张建议。没有改写论文正文。参考复用审计通过；迁移轮 ARTIFACT_MANIFEST 的全部导出文件再次核验未改动。

本轮冻结训练输入及完成产物保持不变。ARTIFACT_MANIFEST.json 固定导出结果/图/报告/后处理代码及上轮报告审计引用；不要重新运行 analyze.py、plot.py 或 make_report.py 覆盖已审计产物，后续改动应另存。

以下为运行期间的历史记录，状态以本页顶部及最终审计为准。

---

# CMRC 任务内训练：当前活动工作

2026-09-15 12:31:49 正式启动，24 个任务，8 张 A100。Python `/home/wz/anaconda3/envs/torch24/bin/python`。用户已授权进行实验，并在此前要求持续前台轮询，不要只启动后结束对话。继续监控直到训练、评测、汇总、审计全部完成。不要创建 goal，不要启动子 agent。

上阶段 ../stacking_chinese_transfer_20260915/ 已完整完成：26 端点、52 任务评测，审计通过、图已查看。FINAL_INTERPRETATION_ZH.md 保存结论。CMRC both−hidden 均值 Qwen3 +0.3745pp、Qwen2.5 +0.8733pp，但三种子有反转；C3 分别 +0.2998pp / -0.0856pp。尚未建立稳定生成收益。6 组共 280 条重生成完全一致，排除了这批输出的重复推理随机变化。原始参考 CMRC Qwen3 45.5295，Qwen2.5 47.0165，后者明显高于上一轮通用 SFT 后的 38.8936。新一轮用于直接检验任务内适配，不是再挑迁移基准。

本轮固定：从已核验的原始聊天权重独立训练，**不续训旧 adapters**。2 模型 × none/output/both/hidden_budget × seeds 42/43/44。官方 CMRC train 按篇章分组为 train 9,114 题 / internal dev 1,028 题；公开 dev 3,219 题为最终评测（承认之前迁移评测已暴露该集合）。1 epoch、570 steps、LR 5e-5、batch 8×2、hidden r8 alpha16 dropout.05、boundary r16 alpha128 dropout0，等预算精确配对。所有输入无截断。

8 组两步烟测已经通过：训练、参数计数、共享初始化、预算、保存重载 plain/AMP logits ≤1e-5 和 adapter 张量逐位一致；internal-dev 上的答案 CE FP64 对照和生成评分。曾因遗漏旧环境 DS_IGNORE_CUDA_DETECTION=1，DeepSpeed 导入触发缺 CUDA_HOME；已补齐与旧训练相同的设置，失败状态/代码/日志/部分 checkpoint 原样保存在 protocol_history/smoke_missing_deepspeed_env。没有正式训练结果在修正前产生。

正式执行器 run.py 会先冻结 manifest.json，再排队 train→internal_dev_ce+CMRC likelihood+CMRC generation。当前 main_scheduler.log / main_state.json 为状态来源，各单元 train_results.json 记录 global_step。源代码/数据/协议已经冻结，**不要直接修改清单中的文件**。最终只读后处理脚本可另加。评测 helpers 与前轮一致，max_new_tokens256，官方 EM/F1，主指标 (EM+F1)/2。4 个主要效应（2模型×2对照），按 DESIGN.md 做三种子 t 区间与篇章簇 bootstrap，并报告 Bonferroni family=4。3seed 的不确定性不能由条件篇章区间替代。

后处理已实现：poll.py、analyze.py、final_audit.py、plot.py。analyze.py --partial 已运行通过旧参考的数据/评分算法 AST/输出来源哈希复用核验，当前暂无完整训练端点评测。正式评测全部完成后，顺序运行 analyze.py、final_audit.py、plot.py，查看图片，写 FINAL_INTERPRETATION_ZH.md 与总的用户结论；最终单独保存导出文件的哈希清单。final_audit.py 尚未执行，若失败应检查并修正外部审计实现，不能修改已冻结实验输入。不要重跑上一轮 analyze.py 覆盖已被其最终审计记录哈希的 RESULTS；上一轮已经形成 ARTIFACT_MANIFEST.json。

最近进度 12:37：首批 seed42 Qwen3 约 415/570 步，Qwen2.5 约 320/570 步，8 个任务均正常，无正式训练失败。剩余 seeds43/44 在队列中。训练完成后 evaluate.py 先测 internal dev CE，再测 9,657 个 CMRC 参考答案的概率和 3,219 题生成。推理配置与前轮相同，但本轮仅 CMRC，没有 C3。

目前还在运行的工具 session：98332 是正式训练执行器。旧阶段其他 session 均已收尾。主进程可以用 write_stdin 轮询，日志输出已重定向；推荐每约 45 秒查看文件并给简短中文进度。保持对话直到本轮完成；不要再扩展新的基准或训练 sweep 来追逐正结果。
