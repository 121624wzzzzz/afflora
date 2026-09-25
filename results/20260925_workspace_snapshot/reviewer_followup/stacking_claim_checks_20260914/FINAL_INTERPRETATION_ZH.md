# 补充实验结论：边界适配能否在 hidden LoRA 上继续带来收益

完成时间：2026-09-15。24 个旧 checkpoint 的校准对照、24 组全新聊天模型训练、两个未微调参考及全部评测均已完成。没有按分数筛选模型、实验臂或种子。

**目前最有依据的主张是：在已检查的配置下，A-LoRA 边界适配为 hidden LoRA 提供了幅度较小、可重复的额外 token 建模收益；它超过一个等参数预算的内部层 LoRA 对照，且 Base 模型上的收益在逐臂温度校准后仍然存在。稳定的自由生成收益尚未建立。**

这轮补充强化了“叠加仍有增量”，也明确了该增量的适用指标。论文应把 CE/token 预测作为当前实证主线，并完整报告 IFEval 的限制。

## 1. 新训练：等预算后仍有 CE 增量

使用经官方文件哈希核实的 Qwen3-0.6B post-trained 和 Qwen2.5-1.5B-Instruct。每个模型比较 hidden LoRA、hidden + output、hidden + bilateral、hidden-budget 四组，全部运行 42/43/44 三个种子，共 24 次全新训练。隐藏层 rank 为 8，双边额外参数分别为 66,560 和 99,840，预算对照精确匹配这些额外参数。

| 聊天模型 | 双边叠加的对照 | Test CE 降低，三种子均值 | 降低量的名义 95% 种子区间 |
| --- | --- | ---: | --- |
| Qwen3-0.6B | hidden LoRA | 0.0054165 | [0.0049205, 0.0059125] |
| Qwen3-0.6B | 等参数预算对照 | 0.0052132 | [0.0051617, 0.0052648] |
| Qwen2.5-1.5B-Instruct | hidden LoRA | 0.0068795 | [0.0056270, 0.0081319] |
| Qwen2.5-1.5B-Instruct | 等参数预算对照 | 0.0067384 | [0.0055325, 0.0079442] |

两个模型、三个种子的全部六组配对中，双边叠加都降低了 CE，并且都优于两个主要对照。相对 output-only addition 再加入 input，平均 CE 还分别降低 0.0024003 和 0.0017227。该比较可描述双边配置的增量，不能据此证明 input/output placement 的普遍排序。

全部实验使用同一固定、未调优的 LR 5e-5、一轮训练、22,780 条训练对话及各 1,000 条 dev/test；共有 1,424 个优化步。预算对照采用预先按尺寸确定的一种 q/k rank 分配，不代表已优化的内部层容量分配最优边界。新聊天模型实验仅覆盖 rank 8，两个模型均 tied。

## 2. 校准对照：收益没有被正标量温度解释掉

完整重载上一轮两个 Base 模型的全部 24 个 rank-8 checkpoint。每个实验臂独立在 dev 上从固定 17 点网格选温度，并在 test 前保存选择；对全监督 token 与非结束 token 分别选择温度。没有温度落在网格边界。

| Base 模型 | 双边叠加的对照 | 校准后 CE 降低 | 非结束 token top-1 增益，百分点 |
| --- | --- | ---: | ---: |
| Qwen3-0.6B-Base | hidden LoRA | 0.005398 | +0.1058 |
| Qwen3-0.6B-Base | 等参数预算对照 | 0.005211 | +0.1021 |
| Qwen2.5-7B-Base | hidden LoRA | 0.005526 | +0.0853 |
| Qwen2.5-7B-Base | 等参数预算对照 | 0.005340 | +0.0953 |

非结束 token 的 top-1 增益在全部六个模型/种子配对中方向一致。正标量温度不改变 token 排序，因此不能把这些 top-1 变化解释为对同一 hidden-only 输出单纯施加正标量温度。有限网格仍不等于连续温度的全局最优，也没有排除更丰富的校准或联合训练机制，更不能单凭此证明完整的几何机制。

准确率增益很小。特别是 7B 相对 hidden LoRA 的 top-1 增益，名义 95% 种子区间为 **[-0.0033, +0.1738] 个百分点**，包含 0，不能写成所有准确率增益都具有统计显著性。完整区间见 [校准解释](calibration/INTERPRETATION_ZH.md) 和 [校准结果](calibration/RESULTS.md)。

## 3. 自由生成：没有同时超过两个主要对照的稳定收益

全部端点完成 541 条 IFEval，使用原生 no-thinking 模板、greedy、相同 1,024 新 token 上限、repetition penalty 1.0 及确定性的官方评分。另报告固定 valid-539 敏感性分析。

| 聊天模型 | 双边相对 hidden，百分点 | 双边相对预算对照，百分点 | 双边 strict 均值 | 未微调参考 strict |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | +0.308 | -0.246 | 50.770% | 58.226% |
| Qwen2.5-1.5B-Instruct | -0.431 | +0.062 | 36.229% | 41.590% |

| 主要对比 | 名义 95% 种子区间，百分点 | 配对题目 bootstrap 95% 区间，百分点 |
| --- | --- | --- |
| Qwen3：双边 − hidden | [-2.749, +3.365] | [-1.109, +1.787] |
| Qwen3：双边 − 预算 | [-2.857, +2.365] | [-1.725, +1.232] |
| Qwen2.5：双边 − hidden | [-1.907, +1.045] | [-1.972, +1.109] |
| Qwen2.5：双边 − 预算 | [-2.467, +2.591] | [-1.540, +1.664] |

上述区间全部包含 0；valid-539 的对应结论一致。种子区间基于固定题目上的三次训练；10,000 次配对题目 bootstrap 对每题先平均种子差值，区间以这三个已训练模型为条件。这是两种不同的不确定性，不能相互替代，均未作多重比较校正。未检出稳定收益也不等于证明所有可能设置下效果为零。

两个模型在本轮 SFT 后的 IFEval 均低于未微调参考。不能把 CE 改善写成绝对指令遵循质量改善。双边配置的平均截断率分别为 7.763% 和 6.100%，未微调参考为 5.730% 和 10.536%；完整长度、截断率、strict/loose、逐题与逐种子结果均保留。上一轮 Base 模型的不稳定生成结果也仍保留在 [P1 结论](../stacking_gain_corrected_20260914/FINAL_INTERPRETATION_ZH.md)，本轮不替换那些负结果。

## 4. 复用、模型身份与审计

复用的 24 个 Base checkpoint 均独立重载，完整 dev/test 共 48,000 次逐样本 CE 检查在原评测实现下重现误差为 0。温度比较另统一采用经过 FP64 对照的连续存储 FP32 CE 实现；原实现与该实现约 0.0000165–0.0000202 的测试平均偏差被单独记录，配对改善方向保留。旧结果未覆盖，见 [精度说明](calibration/PRECISION_NOTE.md)。

新聊天模型不复用旧 Base adapter。模型检查发现原本名为 Qwen2.5-1.5B-Instruct 的本地缓存权重内容不符，已在任何新聊天模型结果产生前拒绝使用；原缓存未修改，新实验使用隔离下载且匹配官方 SHA-256 的权重。两组聊天模型全部文件在最终审计中再次核实实际 SHA-256。八组两步试运行通过标签/初始化/保存重载检查，包括相同精度上下文中的 logits 一致与 adapter 张量逐位一致。见 [模型身份记录](chat/MODEL_PROVENANCE.json)。

SFT 的 train/dev/test 完整对话、首轮提问、完整 assistant 预测上下文及上下文加回答无跨划分精确重叠；IFEval 与 SFT 用户提问的空白归一化精确交集为 0。该检查不排除语义近重复或预训练污染，见 [数据检查说明](DATA_OVERLAP_NOTE.md)。

**最终审计有一项明确记录的范围修正。** 原清单误纳入持续自动更新的 `chat/RESULTS.md`，原调度器因此在所有任务完成后报哈希错误。已精确重建与原哈希相同的初始零结果报告，保留原清单、错误和现有结果。另行审计只排除这个派生输出，检查其余 **958 个冻结输入文件**、软件版本、实际模型文件 SHA-256、训练产物、**52 份 CE 和 26 份生成报告**，全部通过。聚合指标还独立从逐样本记录重算通过。不能描述为原清单每个条目均未改变。详见 [审计说明](REPORT_MANIFEST_NOTE.md)、[聊天实验最终审计](chat/FINAL_AUDIT.json)、[校准最终审计](calibration/FINAL_AUDIT.json)。

## 5. 论文主张与可直接使用的材料

建议主张表述：

> 在所检验的模型和训练配置下，低成本词表边界仿射适配能够在已有 hidden LoRA 上继续降低测试交叉熵，并优于一个等参数预算的内部层 LoRA 对照。该 token 建模增量跨训练种子复现，且在 Base 模型上经过逐臂标量温度校准后仍然存在；自由生成的指令遵循收益尚未稳定建立。

对应英文：

> In the tested configurations, vocabulary-boundary affine adapters provide small, reproducible reductions in held-out cross-entropy when stacked on hidden-layer LoRA, including against a parameter-matched hidden-LoRA allocation. The Base-model gains persist after per-arm scalar-temperature calibration, while consistent improvements in free-generation instruction following remain unestablished.

当前聊天实验没有 input-only 臂，不能验证论文故事线中的 input/output placement 反转；两个聊天模型均 tied，也不能给出 tying 的因果结论。固定且未调优的训练方案、三个种子、单一生成基准及已检查过的评测数据限制了外推范围。这轮属于看到 P1 后的明确补充实验，不是原始假设的预注册或全新未触碰测试集。

本报告未改写论文正文。可使用材料：

- [聊天模型完整汇总](chat/FINAL_ANALYSIS.md)、[机器可读统计](chat/FINAL_ANALYSIS.json)、[配对 CSV](chat/final_contrasts.csv)。
- [聊天模型图 PNG](chat/figures/chat_increments.png) / [PDF](chat/figures/chat_increments.pdf)。
- [校准完整汇总](calibration/RESULTS.md)、[配对 CSV](calibration/paired_contrasts.csv)。
- [校准图 PNG](calibration/figures/calibration_increments.png) / [PDF](calibration/figures/calibration_increments.pdf)。
- [固定设计](DESIGN.md)、[审计修正程序](finalize_chat_audit.py)、[逐样本重算与统计程序](analyze_chat.py)。
