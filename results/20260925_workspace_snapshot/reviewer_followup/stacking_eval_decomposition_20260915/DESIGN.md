# 评测分解：固定 checkpoint 的诊断补充

2026-09-15，用户要求深入审查和验证当前效果评测。在已看过 CMRC 任务内训练结果后设计，明确属于探索性诊断，不作为新的盲测或替换既有主要生成指标。

问题：为什么内部验证 CE 稳定改善，而公开 dev 的答案 CE / EM / F1 没有同步改善？需要分别检查数据划分、token/题目权重、参考答案重复、结束符和内容 token，而非把多个差异混成“CE 与生成不一致”。本轮不追加训练或新基准。

固定复用 CMRC 任务内训练全部 24 个 checkpoint（2 模型 × none/output/both/hidden_budget × seeds42/43/44）。先检查上轮最终审计与导出文件哈希、冻结源码/数据/基础权重哈希和每个 adapter 的完成标记。新诊断文件单独保存，不覆盖旧结果。

CPU 检查：比较两个模型全部 public-dev 参考和 internal-dev 回答的生成前缀+单独编码答案，与训练完整 chat template 的受监督片段是否一致；检查空 thinking 前缀、特殊 token 与边界位置。利用原始逐参考 nll/tokens/top1 计算原三参考题目等权 CE、全 token 加权 CE、去重参考题目等权 CE、第一参考 CE。保留全部口径，不能按增益选择新主指标。

GPU 检查：独立重载全部 24 个 checkpoint，在 internal dev 1,028 题与 public dev 3,219 题（9,657 个参考）上计算相同概念的内容 CE、结束符 CE、二者合计 CE，以及第一个内容 token、其余内容 token 的 NLL/top-1。额外记录 gold 的 top-5 命中与相对最佳非 gold token 的 logit margin。均为 teacher-forced 诊断，不称为序列生成准确率。

Top-1 严格沿用旧 argmax（同分取最小 token ID）；top-5 明确定义为 logit 降序、token ID 升序后的前五名，避免 topk 对同分 token 的任意排序。初次新诊断烟测误用 topk 的第一项替代 argmax 导致两组复用核验失败，原始代码/输出/日志保存在 protocol_history/smoke_topk_tie；旧评测本身使用 argmax 未受影响。完整新评测只在修正后八组烟测通过、清单冻结后开始。

Public dev 使用旧概率评测完全相同的输入 token、批次（最多16 / padded tokens4096）、顺序、模型加载方式和数值精度。无需把 EOS 加到输入：最后一个内容 token 的 logits 已给出下一个 EOS 的概率，因此原内容概率可以逐参考严格重现并核验。Internal dev 使用旧完整训练模板和 batch1，取同一前向的标签 token 并分解内容/EOS；同时核对逐样本旧 CE（原实现的 FP32 求和与新 FP64 求和允许绝对 nll 误差 1e-4，超界应保留失败并检查）。新 FP32 连续 CE 在首批对 FP64 验证最大误差 <5e-6。EOS 固定为训练使用的 im_end，不与其他结束符概率求和。

每个模型的四个实验臂先用 internal dev 前两题和 public dev 第一个完整原始批次烟测，保留原批次形状以严格重现概率。全部通过后冻结主执行输入。主评测完整运行所有种子。比较 both−none 和 both−hidden_budget；output 保留作辅助。所有新指标属于诊断家族，报告全部三种子效果，不能用其中一个正值宣称主假设成立；若给区间只作为描述性名义种子 t 区间，不作确认性显著性选择。

分析重点：同一划分同一权重下，total CE 改善有多少 NLL 来自 EOS 与内容；去掉 EOS 或改变权重后方向是否变化；第一个答案 token 与后续 token 的预测是否不同；同一题 CE 改善和生成胜负是否一致。分解是记账恒等式及关联诊断，不识别训练机制的因果贡献。参考答案等价性不由字符串包含自动判断。

在完整新分解运行前明确两项 CPU 汇总：以固定答案 token 长度 1–4、5–8、9–16、17+ 展示全部分桶，并核对 micro−macro = Cov(length, per-reference CE)/mean(length) 的加权恒等式；另计算每题去重后的完整参考答案集合 NLL，即 -log sum_y p(y, im_end | prompt)。结束符使不同完整答案对应互斥 token 序列事件；该概率只覆盖给定参考字符串，不等于全部语义正确答案，也不是 greedy accuracy。两项均为探索性解释，不能替换原 EM/F1 或选择最正的口径。

评测更新建议应基于上述核验：保留 EM 与 F1 的独立主列，AVG 只保留为上一轮固定汇总；把统一权重的内容概率与结束行为拆开；继续同时报告 hidden 和等预算对照、全部种子与条件于训练结果的题目不确定性。不得删除负结果或把见过的 public dev 改称隐藏 test。
