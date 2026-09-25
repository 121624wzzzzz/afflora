# 叠加收益补充实验：CMRC 任务内训练与证据边界

完成于 2026-09-15T13:40:53+08:00。本轮 24 组全新 CMRC 训练及评测全部完成，最终审计通过。此前中文迁移阶段的 26 个端点、52 项任务评测也已完成。全部实验臂、种子和正负结果保留。

**本轮没有建立“在两个模型及两个主要对照上，边界叠加都能带来稳定的额外生成收益”。Qwen3 相对普通 hidden LoRA 的三个种子方向一致，但相对等参数 hidden 对照仍有种子反转。已有中文通用对话 CE 收益与当前任务指标的结论必须分别表述。**

CMRC 的主指标固定为官方 EM 与 F1 的平均值，单位为百分数；增量用百分点表示。四个适配臂均从核验过的原始聊天模型新训练，不继续训练上一阶段的通用对话 adapters。

| 模型 | 配置 | 种子数 | EM % | F1 % | (EM+F1)/2 % |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 原始聊天模型 | 1 | 29.6676 | 61.3914 | 45.5295 |
| Qwen3-0.6B | hidden LoRA | 3 | 64.2332 | 85.8912 | 75.0622 |
| Qwen3-0.6B | hidden + output | 3 | 64.6785 | 86.0444 | 75.3614 |
| Qwen3-0.6B | hidden + input + output | 3 | 64.9374 | 86.1219 | 75.5296 |
| Qwen3-0.6B | 等参数 hidden 对照 | 3 | 64.6163 | 86.0610 | 75.3386 |
| Qwen2.5-1.5B-Instruct | 原始聊天模型 | 1 | 29.5744 | 64.4586 | 47.0165 |
| Qwen2.5-1.5B-Instruct | hidden LoRA | 3 | 66.4803 | 86.9788 | 76.7295 |
| Qwen2.5-1.5B-Instruct | hidden + output | 3 | 66.4803 | 86.9503 | 76.7153 |
| Qwen2.5-1.5B-Instruct | hidden + input + output | 3 | 66.2939 | 87.0000 | 76.6469 |
| Qwen2.5-1.5B-Instruct | 等参数 hidden 对照 | 3 | 66.3353 | 86.9548 | 76.6450 |

**双边叠加的配对增量。** 下表均为名义 95% 区间。种子 t 区间描述固定评测集上三次训练的波动；篇章簇 bootstrap 区间以这三个已拟合模型为条件，描述评测篇章采样的不确定性。二者不能相互替代。

| 模型 | 对照 | 平均增量 pp | 三种子 t 区间 | 篇章簇 bootstrap 区间 |
| --- | --- | ---: | --- | --- |
| Qwen3-0.6B | hidden | +0.4675 | [-0.1884, +1.1233] | [+0.1880, +0.7610] |
| Qwen3-0.6B | 等参数 hidden | +0.1910 | [-0.7073, +1.0892] | [-0.0725, +0.4618] |
| Qwen2.5-1.5B-Instruct | hidden | -0.0826 | [-0.3257, +0.1604] | [-0.3811, +0.2155] |
| Qwen2.5-1.5B-Instruct | 等参数 hidden | +0.0019 | [-0.1509, +0.1547] | [-0.3030, +0.3073] |

| 模型 / 对照 | seed 42 | seed 43 | seed 44 |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B / hidden | +0.4554 | +0.2097 | +0.7373 |
| Qwen3-0.6B / 等参数 hidden | +0.3817 | -0.2260 | +0.4173 |
| Qwen2.5-1.5B-Instruct / hidden | -0.0734 | -0.1847 | +0.0103 |
| Qwen2.5-1.5B-Instruct / 等参数 hidden | -0.0662 | +0.0183 | +0.0535 |

四个主要比较的 Bonferroni 同时区间也全部保留，避免只挑一个正向比较。

| 模型 / 对照 | 校正后三种子区间 | 校正后篇章簇区间 |
| --- | --- | --- |
| Qwen3-0.6B / hidden | [-0.8830, +1.8179] | [+0.1175, +0.8401] |
| Qwen3-0.6B / 等参数 hidden | [-1.6587, +2.0407] | [-0.1386, +0.5344] |
| Qwen2.5-1.5B-Instruct / hidden | [-0.5831, +0.4179] | [-0.4646, +0.2858] |
| Qwen2.5-1.5B-Instruct / 等参数 hidden | [-0.3128, +0.3166] | [-0.3860, +0.3886] |

本轮四个主要比较的名义 95% 种子区间全部包含零。不能把较窄的条件篇章区间当作跨训练可重复性的替代证据；未检出稳定收益也不等于证明所有设置下效应为零。

**绝对提升的归属。**

Qwen3-0.6B 从未做本轮任务训练的 45.5295，到 hidden LoRA 的 75.0622、双边叠加的 75.5296。其中 hidden 相对原始模型已提高 29.5327 个百分点；双边方法相对 hidden 的额外贡献只有 +0.4675 个百分点。大幅绝对提升不能全部归给边界叠加。

Qwen2.5-1.5B-Instruct 从未做本轮任务训练的 47.0165，到 hidden LoRA 的 76.7295、双边叠加的 76.6469。其中 hidden 相对原始模型已提高 29.7130 个百分点；双边方法相对 hidden 的额外贡献只有 -0.0826 个百分点。大幅绝对提升不能全部归给边界叠加。

**辅助概率指标与输出长度。** Internal dev CE 是含答案结束符的 token 加权指标；public dev answer CE/top-1 只计算参考答案内容，先等权平均每题的三个参考，再平均题目。划分、监督范围和权重都不同，不能混算或把两者差值解释为过拟合幅度。

| 模型 | 配置 | Internal dev CE | Public dev answer CE | Answer top-1 % | 输出 token 均值 | 截断率 % |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | hidden LoRA | 0.169361 | 0.283294 | 92.5329 | 12.022 | 0.0000 |
| Qwen3-0.6B | hidden + output | 0.168995 | 0.283572 | 92.5426 | 12.047 | 0.0000 |
| Qwen3-0.6B | hidden + input + output | 0.167514 | 0.282609 | 92.5507 | 12.009 | 0.0000 |
| Qwen3-0.6B | 等参数 hidden 对照 | 0.169135 | 0.283101 | 92.5119 | 11.981 | 0.0000 |
| Qwen2.5-1.5B-Instruct | hidden LoRA | 0.159031 | 0.256681 | 92.6603 | 12.317 | 0.0000 |
| Qwen2.5-1.5B-Instruct | hidden + output | 0.158211 | 0.257514 | 92.6830 | 12.316 | 0.0000 |
| Qwen2.5-1.5B-Instruct | hidden + input + output | 0.157666 | 0.257683 | 92.6171 | 12.382 | 0.0000 |
| Qwen2.5-1.5B-Instruct | 等参数 hidden 对照 | 0.159111 | 0.256418 | 92.6596 | 12.243 | 0.0000 |

Qwen3-0.6B，both−none：public-dev answer CE 逐种子差值为 -0.002224 / +0.000725 / -0.000558；internal-dev CE 为 -0.001372 / -0.001992 / -0.002176（CE 负值表示改善）。

Qwen3-0.6B，both−hidden_budget：public-dev answer CE 逐种子差值为 -0.002723 / +0.001309 / -0.000062；internal-dev CE 为 -0.001174 / -0.001673 / -0.002017（CE 负值表示改善）。

Qwen2.5-1.5B-Instruct，both−none：public-dev answer CE 逐种子差值为 +0.001922 / +0.001111 / -0.000026；internal-dev CE 为 -0.001721 / -0.001155 / -0.001218（CE 负值表示改善）。

Qwen2.5-1.5B-Instruct，both−hidden_budget：public-dev answer CE 逐种子差值为 +0.000416 / +0.003402 / -0.000022；internal-dev CE 为 -0.001852 / -0.000983 / -0.001499（CE 负值表示改善）。

全部 24 个任务训练端点的生成截断率均为零，本轮观察到的波动不能归因于 256 token 上限截断。

**与前两轮证据合并看。**

| 证据 | 支持的结论 | 尚未支持的结论 |
| --- | --- | --- |
| 中文通用对话 SFT 的 test CE | 两模型、三种子均优于 hidden 和一个精确等参数 hidden 分配；CE 平均降低约 0.0054 / 0.0069 | 相同幅度的准确率提升或普遍生成改善 |
| Base checkpoint 的逐臂温度校准 | CE 增量在固定温度网格校准后保留，不能只由该标量校准解释 | 完整几何机制或所有准确率增益都显著 |
| 通用 SFT 后的中文 CMRC/C3 迁移 | 部分模型/任务有正向均值 | 跨种子、跨任务稳定超过两个主要对照 |
| 本轮 CMRC 任务内训练 | 直接控制训练任务与短答案目标；Qwen3 相对 hidden 三种子均正 | 两模型均稳定超过等参数对照 |

前轮迁移的 both−hidden / both−等预算均值：CMRC Qwen3 为 +0.3745 / +0.0669 pp，Qwen2.5 为 +0.8733 / +0.7620 pp；C3 Qwen3 为 +0.2998 / +0.3169 pp，Qwen2.5 为 −0.0856 / −0.0428 pp。C3 是候选答案似然选择准确率，不是自由生成。完整正负结果及不同不确定性区间见[迁移报告](../stacking_chinese_transfer_20260915/FINAL_INTERPRETATION_ZH.md)。旧 CE/IFEval 与校准证据见[前轮报告](../stacking_claim_checks_20260914/FINAL_INTERPRETATION_ZH.md)。

**为什么 CE 收益与生成收益会分离。**

已观测到：通用 SFT 数据主体为中文，先前 IFEval 几乎全为英文；但换成中文 CMRC/C3 后仍有种子反转，因此语言不匹配不是完整解释。新任务的参考答案 CE 本身也存在种子波动，不能假设旧对话 CE 收益已经迁移到新任务。前轮选定 280 条输出的独立重生成逐 token 一致，这批波动来自不同训练结果之间的差别，不能简单归为重复推理的随机采样。

从指标定义看，teacher forcing 下参考 token 概率的小幅提高不要求 argmax 改变；自由生成又会使用模型自己的前缀，EM/F1 对离散答案形式敏感。这些事实说明 CE 与生成指标不必同步，但本实验没有因果分解各因素的贡献。前轮“baseline 已包含 gold 字符串”的诊断只是探索性观察，不能等同于语义正确率或证明全部收益都来自表达形式。

本轮任务训练让短答案表现大幅提高，却仍未建立普遍的边界增量；这意味着单靠换语言、消除长输出或对齐任务目标，并不足以保证叠加收益。当前证据支持效应对任务、配置与训练种子有依赖，而不是认为正结果必然应当出现。

**建议论文采用的主张。**

> 在所检验的中文通用对话适配配置下，低成本词表边界仿射适配能在 hidden LoRA 上继续降低 held-out cross-entropy，并超过一个等参数预算的 hidden-LoRA 分配。该建模增量跨种子复现，Base 模型上的增量经逐臂标量温度校准后仍保留；其向下游任务生成指标的转化具有条件性，尚未建立跨模型与等预算对照都稳定成立的生成收益。

> In the tested Chinese dialogue adaptation configurations, vocabulary-boundary affine adapters provide small, reproducible held-out cross-entropy reductions beyond hidden-layer LoRA and one parameter-matched hidden-LoRA allocation. The Base-model gains persist after per-arm scalar-temperature calibration. Translation to downstream generation metrics remains task- and configuration-dependent; consistent gains across models and matched-budget controls are not established.

对当前 paper-2：引言与结论应把“额外收益”落实到已验证的目标和配置；生成实验作为适用边界完整报告。关于 output 更适合作为补充、placement 反转和 shared mergeable tied 的论述，应依赖各自对应的原实验。本轮缺 input-only、untied 对照和 tying 干预，不能作为这些机制的新增因果证据。本文档提供可审阅表述，未自动改写论文正文。

**固定协议与复用核验。**

官方 CMRC train 按篇章分组划分为 9,114 条训练、1,028 条 internal dev；最终使用官方 public dev 的 3,219 题、848 个规范化篇章簇。三划分无精确篇章交集，但不据此排除语义近重复或基础模型预训练污染。该 public dev 已在迁移轮看过，本轮是明确的顺序补充，不能称为全新盲测或官方隐藏 test。

两个模型 × 四臂 × seeds 42/43/44，共 24 次从原始聊天模型出发的新训练。每次 1 epoch / 570 steps、batch 8×2、LR 5e-5、hidden r8；双边额外参数分别 66,560 / 99,840，与 hidden-budget 精确匹配。该预算对照是预先确定的一种内部 rank 分配，并非所有内部容量分配的最优上界。训练和评测输入无截断，最终 checkpoint 固定，不用 internal/public dev 挑 checkpoint 或调参。

八组烟测验证共享 hidden 初始化、边界初始化配对、预算、保存重载张量逐位相等和同精度 logits 差≤1e−5；概率计算对 FP64 核验。早期烟测环境遗漏导致的失败及修正原样保存在 protocol_history/，全部正式训练在修正与清单冻结后运行。

最终审计重新核验基础权重与冻结输入的完整 SHA-256、软件版本、24 组训练步数/初始化/参数预算、可训练张量的 FP32 与有限值、24 份 internal-dev CE 和 24 份 CMRC 原始结果；逐题重算与官方整集评分入口均通过。原始模型参考复用自迁移轮，已核对数据/评分文件哈希、提示与概率/生成/评分函数 AST 相同及原始输出来源哈希。新训练没有复用旧 adapter。此前旧对话实验清单的派生 RESULTS.md 例外仍按原审计说明保留，本轮不掩盖该历史事项。

范围限制：两个 Qwen 模型、一个 CMRC 任务、三个种子、一种未逐臂调优的配置，且 public dev 先前已暴露。本轮完成后没有按结果追加任务、挑种子或继续 sweep。

可复核材料：[固定设计](DESIGN.md)、[完整结果](RESULTS.md)、[机器可读结果及全部区间](RESULTS.json)、[逐种子差值 CSV](paired_effects.csv)、[最终审计](FINAL_AUDIT.json)、[参考复用审计](REFERENCE_REUSE_AUDIT.json)、[数据/模型审计](DATA_AND_MODEL_AUDIT.json)、[图 PNG](figures/cmrc_task_sft.png) / [PDF](figures/cmrc_task_sft.pdf)。原始逐题概率、生成 token、文本和指标在 outputs/；适配器与训练记录在 checkpoints/。
