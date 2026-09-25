最终结果已完成并通过审计：[FINAL_INTERPRETATION_ZH.md](FINAL_INTERPRETATION_ZH.md)。

# 范围修正：仅 tied 模型运行共享 aLoRA

按用户澄清，保留 Qwen2.5-1.5B 与 Qwen3-0.6B：两任务、共享 rank16/32、三种子，共24次正式训练、8次短训练。untied 的7B/8B共享组已撤销；已有独立单侧/双侧结果继续保留。此前误排的 untied 短训练仅存档，不纳入比较。

权威范围：[SCOPE_AMENDMENT.json](SCOPE_AMENDMENT.json)。当前状态：[STATE_TIED_ONLY.json](STATE_TIED_ONLY.json)。

---

以下为修正前方案，留作审计记录：

# Qwen 共享 E/U 消融

用户授权共享 rank16 和 rank32 均运行。四模型 × 两任务 × 两种共享 rank × 三种子，共48次正式训练；另有16次技术短训练。

- 模型：Qwen2.5-1.5B、7B；Qwen3-0.6B、8B。
- 任务：CLUENER、WikiSQL，沿用已冻结的训练和评测协议。
- 共享 r16 的额外参数为33d；共享 r32为65d，与原独立 E/U 各r16严格相等。
- 输出侧使用共享映射的转置；输入 bias 在输出对应全词表共同平移。不新增输出参数。
- untied 基座共享的是变换，原输入/输出权重仍独立。
- 所有负面结果保留；不按测试集挑rank。先检查小样本训练，再跑完整训练。
- 旧模型、数据、参数和结果重新核验后复用。单侧已有完整消融，不重复计为新增实验。

协议：[study/PROTOCOL.md](study/PROTOCOL.md)
输入复核日志：[verify.log](verify.log)
调度状态：[study/STATE.json](study/STATE.json)
阶段结果：[study/RESULTS_ZH.md](study/RESULTS_ZH.md)
完整配对数值：[study/RESULTS.json](study/RESULTS.json)

在全部训练与独立审计完成前，不将此目录视为最终研究结果。
