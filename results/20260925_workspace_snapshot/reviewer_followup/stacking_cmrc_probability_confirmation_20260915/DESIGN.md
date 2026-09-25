# CMRC 完整答案概率：新训练种子确认

2026-09-15，用户授权继续补实验。本轮针对前轮探索性观察提出固定的新训练重复，不新增任务，不根据结果调参、挑种子或运行到显著为止。新种子对训练随机性独立；同一 CMRC public dev 已被看过，因此不是独立新测试集，也不能称为全新盲测。

固定 2 模型：经完整 SHA-256 核验的 Qwen3-0.6B post-trained、Qwen2.5-1.5B-Instruct。固定 3 臂：none(hidden LoRA)、both(hidden+input+output A-LoRA)、hidden_budget(双边额外参数精确匹配的 hidden rank 分配)。固定新种子 1001/1002/1003/1004/1005，共 30 次从原始聊天模型开始的新训练，不续训或复用旧 adapter。output-only 已有探索性结果，本轮聚焦双边叠加对两个必要对照的确认。

训练完全沿用前轮 CMRC 配置和数据划分：9,114 train / 1,028 internal dev；按规范化篇章分组，与公开 dev 无精确篇章交集。1 epoch、570 steps、batch8×accum2、LR5e-5 cosine、warmup.03、max_grad_norm1、hidden r8/alpha16/dropout.05、boundary r16/alpha128/dropout0；input 有 bias、output 无 bias。额外参数 66,560 / 99,840。基础权重 BF16、adapter FP32、BF16 AMP 训练；各配对臂共享 hidden 初始化，边界初值配对及精确预算检查。仅评测最终 checkpoint，不按 dev 分数选择。

正式训练前做 2 模型×3 臂的两步烟测（seed1001），仅训练前32条、数值/生成烟测 internal dev 前8题。检查保存重载 adapter 张量逐位一致与原生 plain/BF16 AMP logits 差≤1e-5，然后验证本轮的 FP32 评测路径。全部通过后冻结源码、数据、token cache、模型文件和软件版本；正式训练全部完成，无结果依赖的提前停止或追加种子。

本轮概率和生成均使用 FP32 推理：把原始已存储权重提升到 FP32，adapter 同样 FP32；禁用 TF32 和 AMP，明确 eager attention。这不会恢复 BF16 训练前已丢失的精度。概率对每个唯一(prompt token IDs, answer token IDs)以 batch1 计算一次并缓存；三参考宏平均按映射复用值，答案集合按唯一字符串求和。输入 token 与训练完整模板逐条检查，排除 supplied thinking prefix，均无截断。

每个完成 checkpoint 先要求 internal dev 前8个固定案例的 canonical batch1 重复计算结果逐位一致。另比较单样本、不填充与复制成2样本、额外右填充17token的概率；形状敏感性容差为最大单 token NLL 差≤5e-4，完整答案概率差≤1e-4。新 FP32 CE 对 FP64 检查≤5e-6。差异失败保留原始记录并调查，不以主指标正负调整阈值。相同输入的重复参考只计算一次，彻底取消首次/均值去重取法差异。

概率前向只计算所需答案/EOS位置的 LM-head logits（logits_to_keep 张量索引），避免对被 mask 的 prompt 位置计算词表投影。每个数值检查首案例同时与完整 logits 前向比较，使用更严格的 NLL≤2e-4、概率≤1e-5 容差；不改变条件输入、位置或目标。

主要评测：完整 CMRC public dev 3,219题、去重后4,188个参考字符串。每题参考集合 Y 的概率 P=sum_y p(y, im_end|prompt)，使用内容 NLL 加训练结束符 NLL，稳定 logsumexp 聚合；主要指标为题目等权 mean(P)。该指标只覆盖给定参考字符串和 im_end，不覆盖全部语义等价表达，不是 greedy EM，也不宣称校准改善。

**4 个预先固定的主要对比**：两模型各自 both−none 与 both−hidden_budget。报告仅基于5个新种子的配对均值、全部逐种子差值、名义95% t区间及 Bonferroni family=4 双侧同时区间（df4）。每个模型需同时对两个对照的校正区间下界>0，才能称该模型的概率均值增量得到此次新种子支持；两模型都满足才能作跨这两模型的联合表述。此判据只涉及概率指标。若不满足，按实际效应与区间报告未确认，不继续加种子。旧42/43/44结果不进入主要统计，也不与新精度结果混合。

支持指标预先保留：题目等权完整参考集合 NLL=-log(P)（越低越好）、内容 micro/macro CE、结束符 CE、first/rest token NLL和top1。若均值概率上升但集合 NLL均值反向，应明确报告这种权衡，不笼统称所有概率建模指标改善。10,000次篇章簇配对bootstrap（seed20260915）以5个已拟合模型为条件，单独标注，不替代种子不确定性。

官方生成效果同样保留：FP32 eager、native no-thinking、greedy、max_new_tokens256、repetition_penalty1、原EOS加im_end、源顺序、最多16样本/padded prompt tokens8192、无输入裁剪。输出原始token/文本，按既有官方兼容算法报告EM、F1以及前轮固定AVG；本轮生成精度已变化，不能把旧BF16绝对分数直接归因到新种子或新方法。生成对比为明确的次要指标，不用于替换主要概率指标，也不会把概率成功写成生成成功。

适用范围仍为两个 tied Qwen 模型、一个任务、一种固定且未逐臂调优的训练配置和一个已暴露的公开划分。等预算对照只是一个预定 rank 分配，不代表内部适配所有分配的最优值。本轮不检验 input-only、tying 因果机制或跨任务普遍生成收益。

## 正式实验前的数值检查修订（2026-09-15）

最初形状检查设置 NLL≤2e-4、概率≤1e-5。6臂烟测中5臂通过；Qwen2.5 both 有1个案例的 NLL差为2.08855e-4，概率差3.64739e-6，原失败完整保留在 protocol_history/initial_fp32_shape_tolerance。后续受控调查8个案例的 canonical 重复均逐位一致，完整词表与选位置投影差≤1.56e-5；仅改变 batch/填充形状产生上述差异。

在任何正式30组训练/评测开始前，保持实际推理方法不变，增加 canonical 逐位重放门槛；将改变计算形状的诊断容差分别固定为5e-4和1e-4（概率0.01个百分点），完整投影核验保持原更严格容差。所有6臂重跑评测检查。该修订只涉及数值检查，与主指标收益方向无关；最终报告最大实测差异，不把有限案例的形状检查当成全测试集数值误差上界。主要统计判据和30组固定矩阵不变。
