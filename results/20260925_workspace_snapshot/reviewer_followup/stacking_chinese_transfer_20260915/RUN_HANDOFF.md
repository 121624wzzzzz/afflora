# 中文迁移实验运行记录

2026-09-15 12:31：本轮 26 个端点全部完成，FINAL_AUDIT.json 已通过，图已人工检查，FINAL_INTERPRETATION_ZH.md 已写好。后续 CMRC 任务内训练在 ../stacking_cmrc_task_sft_20260915/ 进行，用户回合仍在继续。下文保留历史运行说明。

用户授权执行 CMRC2018 与 C3 中文任务实验，此前要求运行期间持续前台轮询。当前一次进程调度八张 GPU，26 个 checkpoint，每个先复现旧 CE，再测 C3 likelihood、CMRC likelihood 和 CMRC generation。不要在实验尚运行时结束用户回合。

已经完成：作者仓库固定版本数据下载、Git blob SHA-1 与 SHA-256 核验；旧清单 958 个不可变输入（明确保留派生 RESULTS.md 唯一例外）、两个基础模型完整权重哈希、26 组 adapter 身份检查；10 个训练样本烟测全部通过，旧 CE 重现误差为 0；所有正式输入长度检查通过，不需截断；正式协议已写入 `manifest.json`。

早期烟测发现 Transformers apply_chat_template 默认返回字典，显式设置 `return_dict=False` 后修正。失败代码、日志、状态与部分烟测记录原样保留在 `protocol_history/smoke_template_return_dict`；该故障发生在正式结果产生前。不得删除该记录，也不得覆盖原先两轮研究的任何产物。

`DESIGN.md` 与 `manifest.json` 已冻结。评测执行文件 common.py/evaluate.py/run.py、数据及其评分器均在清单中。勿直接更改。汇总、绘图、最终审计为外部只读后处理，不在评测输入清单内，可修正但必须保留最终来源哈希。

轮询：`python poll.py`；阶段汇总：`python analyze.py --partial`。`main_scheduler.log` 记录调度，各 GPU 子进程的日志在 logs/main.*.log，逐候选/逐题输出在 outputs/<checkpoint>/。

待全部 26 个 checkpoint 完成：运行 `python analyze.py`（8 个主要效应，三种子 t 区间及篇章簇 bootstrap，10,000 次，另报 Bonferroni 区间）；运行 `python final_audit.py`（只读检查全部输入/模型与输出，并用官方评分入口重算 CMRC，验证全部 C3/CMRC 聚合）；运行 `python plot.py`，人工查看图片；编写新的 `FINAL_INTERPRETATION_ZH.md` 并给用户完整结论。不要改旧实验报告来替代旧负结果。

当前只是已有中文 SFT checkpoint 的任务迁移，无任务内新训练。CMRC 使用 public dev，C3 使用 test。既有 IFEval 的英文约束迁移结果继续保留。不能由本轮正负结果单独归因到语言，也不能将 C3 likelihood accuracy 写成自由生成收益。
