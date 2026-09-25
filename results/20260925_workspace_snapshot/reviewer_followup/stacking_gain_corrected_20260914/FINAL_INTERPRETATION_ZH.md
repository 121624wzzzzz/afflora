# 叠加增益：完整实验结论

最终审计通过时间：2026-09-14T22:12:18+08:00。72/72 组完成，60 组新训练、12 个通过复用审计的 checkpoint；144 份 CE 报告、72 份完整 IFEval 报告，961 个冻结文件校验通过，审计错误为零。

当前证据支持：在已有 hidden LoRA 上增加独立输入/输出边界适配器，能带来可复现的小幅测试 CE 降低，而且优于本次预先固定的近等参数预算 hidden 对照。当前实验没有证明稳定的额外指令遵循收益。

## 完整配对结果

2 个 Base 模型 × hidden rank 8/32/64 × 4 个实验臂 × 3 个种子（42/43/44）。以下均为 3 个种子的配对平均；CE 降低取对照减去双边叠加，正数为收益；IFEval 增益为双边叠加减去对照，单位为百分点。

| 模型 | hidden rank | CE 降低：对 hidden LoRA | CE 降低：对预算对照 | IFEval strict 增益：对 hidden LoRA | IFEval strict 增益：对预算对照 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B-Base | 8 | 0.005374 | 0.005180 | -1.664 | -1.294 |
| Qwen3-0.6B-Base | 32 | 0.002026 | 0.001882 | -0.493 | -0.431 |
| Qwen3-0.6B-Base | 64 | 0.000692 | 0.000802 | +0.431 | +0.000 |
| Qwen2.5-7B-Base | 8 | 0.005469 | 0.005284 | -0.801 | -0.555 |
| Qwen2.5-7B-Base | 32 | 0.002483 | 0.002404 | -1.787 | -1.787 |
| Qwen2.5-7B-Base | 64 | 0.001622 | 0.001639 | -0.739 | -1.232 |

双边叠加对 hidden LoRA 和预算对照的 CE 改善分别出现在全部 18/18 个模型×rank×种子条件中；增益随 hidden rank 提升而减小。IFEval 对预算对照的 6 个模型×rank 平均结果为 5 个负值、1 个持平；固定 valid-539 敏感性分析不改变这一判断。Qwen3 rank 64 对 hidden LoRA 的平均小幅正增益，在预算对照下消失；不能用它支持普遍生成收益。

双边并非始终优于仅加输出侧：例如 Qwen3 rank 64 的三个种子都是 output 的 CE 更低。因此，“输入输出两侧始终互补”也不是当前数据支持的主张。

完整报告及图中的区间是固定评测集上、仅针对 3 个训练种子变化的名义配对 t 95% 区间，未做多重比较校正；不能据此宣称跨任务普遍有效。评测集此前已被查看，后续更广泛的主张仍需要新的独立验证。预算对照是一次预先固定的额外 hidden rank 分配，Qwen3 参数完全匹配，7B 对照多 512 个参数；这不等于比较过优化后的全部 rank 分配方案。

## 复用内容确实检查后才进入结果

12/12 checkpoint 检查了来源、数据、训练配置、步数、参数规模、成对初始化、FP32 张量及有限值，并核对文件 SHA-256。每个 checkpoint 独立重载，重新评测完整 dev/test 各 1,000 条，共 24 份报告、24,000 次样本评测。ID 与监督 token 数匹配；汇总 CE 和逐样本 CE 的最大差异均为 0。新矩阵只使用通过检查的新算 CE；全部 IFEval 重新生成。详见 [REUSE_CHECK.md](REUSE_CHECK.md)。

## 生成协议的实质限制

Qwen3 Base 冻结输出矩阵中，检查到的 56 个原生聊天特殊 token 行完全相同，其中包含 im_start / im_end。当前所有实验臂都不能通过改变隐藏表示来分开这些相同行的 logits。完整 36 组 Qwen3 报告的 512-token 截断比例为 99.815%–100%；36 组 Qwen2.5 报告为 60.444%–99.630%。Qwen2.5 对应行并非完全相同，不能把 Qwen3 的严格不可区分论断直接套给它。详见 [GENERATION_LIMITATION.md](GENERATION_LIMITATION.md)。

已修正停止 token 列表，旧生成全部归档排除，但停止列表修正不会修复 Qwen3 冻结输出行相同的问题。生成结果只能按此固定 Base 模型协议解释，不能把 CE 改善写成已证明的实用聊天增益。

Qwen3 rank 8 / seed 42 的完整 test token 分解已逐样本复现主评测：双边叠加 CE 改善的 92.70% 来自非 im_end token。这排除了该条件下“收益主要来自结束 token”的解释；它没有排除概率校准因素，也不能自动推广到所有条件。详见 [token_breakdown/RESULTS.md](token_breakdown/RESULTS.md)。

## 对论文主张的含义

目前较准确的表述是：边界适配在已有 hidden LoRA 上提供额外的似然拟合收益，且优于本次固定的近等预算 hidden 扩容；但尚未转化为稳定的指令遵循收益。若论文主张实用生成增益，应在兼容聊天输出头、各实验臂共同控制特殊 token 处理的新协议下做匹配实验，同时用校准对照检验 CE 收益的来源。该新协议需要新的匹配训练，现有 checkpoint 不能直接当作其公平基线。本轮未启动这项后续实验。

## 结果文件

- [全部实验与区间](RESULTS.md)
- [最终完整性审计](FINAL_AUDIT.json)
- [对比图 PNG](final_figures/stacking_increment.png) / [PDF](final_figures/stacking_increment.pdf)
- [配对差值 CSV](final_figures/paired_contrasts.csv) / [图注](final_figures/CAPTION.md)
- [图表来源哈希](final_figures/SOURCE_HASHES.json)
