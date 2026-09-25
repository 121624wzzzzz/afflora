# 修复版 SciQ 下游任务实验

本目录修复当前 SciQ 选择题适配的评测与比较设置。它不是完整通用 SFT、知识注入或叠加增益实验。协议是对已查看过的基准进行事后修复，不能作为从未查看测试集的新确认研究。

## 固定设置

- 两个官方 Base 为主实验：Qwen3-0.6B-Base、Qwen2.5-1.5B。官方 repo/revision 与逐文件 SHA256 见 `models.json`；后训练 checkpoint 单独列为次要分析。
- 同一 checkpoint 的训练前后使用相同提示 token、原生 EOS、推理模式和解码预算。Base 用已核验的文本补全，后训练参考用原无思考聊天模板。不得横向混合解读。
- SciQ 训练 11,646、验证 1,000、原始测试 1,000。沿用事先固定的两道歧义题排除，测试汇总 n=998。
- 保留“答案字母 + EOS”的受控任务训练目标。原训练无标签移位问题，无需仅为修改评分而重训已核验 adapter。
- 每个 checkpoint 重新进行完整候选评分及真实 greedy 生成，统一上限 128 token。生成答案、严格格式、完整终止和截断分别计分。`scoring.py` 的解析规则冻结后不按测试输出修改。
- 主实验包含未适配、仅输入 A-LoRA、仅输出 A-LoRA、内部全层 LoRA r8，以及固定中间层 q_proj 的小预算对照。新对照在 Qwen3 上为 33,792 参数，在 Qwen2.5 上为 49,152；后者比输入 A-LoRA 少约 3%，不是精确同参数量。
- 新对照使用相同三学习率预算和验证集选择规则，五种子5002–5006，与原 Base 研究配对。所有原始权重冻结，只有指定适配器可训练。

完整边界和统计计划见 [DESIGN.md](DESIGN.md)。首轮冒烟检查触发的浮点容差问题及修正保留在 [PREFLIGHT_NUMERICS.md](PREFLIGHT_NUMERICS.md)，没有修改训练目标或正式训练结果。

## 运行与检查

固定 Python：`/home/wz/anaconda3/envs/torch24/bin/python`。GPU 队列使用 1–6，避开已被其他任务占用的 0/7。

`pipeline.py` 依次执行 smoke、freeze、evaluate_tune、select、confirmation。已存在的训练目录会拒绝覆盖；已封存目录拒绝追加训练或复评。当前修复的启动方式为：

```bash
python pipeline.py smoke
python pipeline.py freeze
python pipeline.py evaluate_tune
python pipeline.py select
python pipeline.py confirmation
python audit.py
python analyze.py
```

上述顺序记录本次运行方法，不应在已完成目录中从头重复执行。重新开展完整实验需建立新输出目录并重新绑定来源与冻结协议。

封存后只读复验使用 `python verify_archive.py`；它核对本轮、原来源及官方模型文件而不改写任何结果。不要在封存目录重新执行会生成审计/分析文件的 `audit.py` 或 `analyze.py`。

原研究的 54 个 reference/confirmation checkpoint 只读复用，新评估写到 `evaluations/`。新训练写到 `checkpoints/`。`job_status/` 和每个训练目录的 `PROGRESS.json` 记录进度；`COMPLETION.json` 存在且独立审计通过才表示本轮完成。

## 结果入口

- `RESULTS.json`：全部模型、方法、种子、两类内容指标及区间。
- `FINAL_AUDIT.json`：预测重算、生成重解码、冻结权重、来源与验证集选择审计。
- `FINAL_INTERPRETATION_ZH.md`：完成后的结果与适用范围。
- `INVALID_ANSWERS.json`：未能解析或出现冲突的输出，方便复查；不得据此回调本轮评分规则。
- `ARTIFACT_MANIFEST.json`：完成后封存的逐文件清单。

`base` 作为代码中的实验臂意味着“未适配当前 checkpoint”，不意味着当前 checkpoint 必然为预训练 Base。本文与结果报告始终另列模型身份。

此前 thinking 复测仅证明官方 Qwen3 的 62.53% 不能代表全部可用答题水平；它改变了模式、采样和预算，不是本轮 adapter 效果的配对对照。叠加、跨任务、保留能力与通用 SFT 不在本轮的验证范围内。
