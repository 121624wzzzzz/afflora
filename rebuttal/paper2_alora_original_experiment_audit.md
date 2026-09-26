# A-LoRA 原论文实验设计与结果审计

本文件以正式投稿包 `/home/wz/projects/mypro/im_exp/paper/submission/2/main.pdf` 为准，只还原投稿论文已经设计和报告的实验，不直接撰写 rebuttal。核心问题是区分：论文原本想验证什么、表格实际比较了什么，以及结果能够支持什么。

## 1. 原论文的完整证据链

### 1.1 Base→Instruct 静态结构分析

- 对象：30 个 Base→Instruct 权重对，包含 Qwen、Llama、Gemma 和 DeepSeek；17 个 tied、13 个 untied。
- 指标：目标矩阵变化落在 base matrix 诱导的 effective affine subspace 中的能量比例 $R^2_{\mathrm{aff},K}$。
- 结果：Untied input embedding 为 3.6%，Untied LM head 为 33.0%，tied shared matrix 为 32.2%。
- 能支持的结论：在本文分析的 Base→Instruct residual 中，output/tied 比 untied input 更 affine-friendly；A-LoRA 应被定位为有结构适用条件的方法。
- 不能直接支持的结论：$R^2_{\mathrm{aff},K}$ 必然预测任意下游任务收益。

### 1.2 离线 equal-budget fitting probe

- 性质：在已经观察到的 $\Delta W$ 上做离线最优拟合，不是 adapter 下游训练。
- 预算换算：$r_{\mathrm{aff}}\approx r_W(V+d)/(2d)$。
- rank-1 Vocab-budget 下的中位 aff/W explained-gain ratio：Untied U 为 2.18×（12/13 胜），tied 为 3.00×（15/17 胜），Untied E 只有 0.34×（1/13 胜）。
- 能支持的结论：当 residual affine-friendly 时，把同一参数预算用于 hidden-space affine rank，离线表示匹配更有效。
- 不能替代：真实 SFT 优化、生成质量或下游准确率实验。

### 1.3 下游 SFT 的四条验证路径

论文实验章节明确设计了四条路径：

1. A-LoRA-only 与 Vocab-LoRA-only 的效果—参数量比较；
2. A-LoRA-only 与参数量相近的 single-layer hidden LoRA 比较；
3. standalone 小预算下 input 与 lm_head placement 比较；
4. 已有 hidden LoRA 时的增量 placement，以及 tied/mergeable 形式。

投稿时下游设置为 MiniMind 25k（24k train / 1k eval）、1 epoch、max length 1024、effective batch 16、learning rate $2\times10^{-4}$、cosine schedule 和 warmup 0.03。小模型关键 placement 使用 seeds 42/43/44，较大模型主要是单 seed 外推。

## 2. 原 Table 4 的参数效率结论

用户的记忆是正确的：原 Table 4 的主要结论就是 **A-LoRA 用显著更少的 vocabulary-boundary 参数，通常取得比 Vocab LoRA 更低的 validation loss**，而不是只证明参数公式与 $V$ 无关。

| 模型与配置 | A-LoRA 参数 / loss | Vocab LoRA 参数 / loss | A/V 参数比 | A−V loss |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B, r2 | 4,480 / 1.879 | 611,328 / 1.883 | 1/136.5 | -0.004 |
| Qwen2.5-0.5B, r4 | 8,064 / 1.852 | 1,222,656 / 1.816 | 1/151.6 | +0.036 |
| Qwen3-0.6B, r2 | 5,120 / 1.494 | 611,840 / 1.529 | 1/119.5 | -0.035 |
| Qwen3-0.6B, r4 | 9,216 / 1.469 | 1,223,680 / 1.495 | 1/132.8 | -0.026 |
| Qwen2.5-1.5B, A r16 / V r4 | 50,688 / 1.473 | 1,227,776 / 1.538 | 1/24.2 | -0.065 |
| Qwen3-1.7B, A r16 / V r4 | 67,584 / 0.981 | 1,231,872 / 1.145 | 1/18.2 | -0.164 |

按投稿表格本身，A-LoRA 在 6 组中的 5 组 loss 更低，使用约 18–152 倍更少的 adapter 参数；唯一反例是 Qwen2.5-0.5B r4。这是明确的 performance–parameter trade-off 证据。

### Table 4 的实际协议与正文标签不一致

- 它不是 equal-budget：前四行控制 rank，后两行是 A r16 对 Vocab r4。
- 它也不是完全 topology-matched：A-LoRA 是 unilateral `affine_input`，Vocab LoRA 是 bilateral `embed_tokens + lm_head`。
- 因而 18–152× 的倍率中约有 2× 来自单侧/双侧差异。即使保守地把 Vocab LoRA 也按单侧参数量计算，A-LoRA 仍约少 9–76× 参数，但目前没有对应的 input-only Vocab loss。
- 实验逻辑表写了 “parameter budget matched”，Method 又给出了 equal-budget rank conversion；这与 Table 4 的实际运行协议不一致。Reviewer 9sSY 对公平性和口径的批评是成立的。

因此，rebuttal 中应把 Table 4 准确称为 **rank-controlled / cross-rank performance–parameter trade-off comparison**，不能称为 equal-budget superiority。

### Table 4 的正式投稿记录

- 正式投稿 PDF 的 Section 6.1 / Table 4 完整报告六组结果；Appendix A.11 又将其中的代表性配置列为 run-level results，并说明实际配置。
- 六组参数量均与论文中的 A-LoRA/Vocab LoRA 参数公式和 `Param ratio` 完全吻合，正式投稿材料内部一致。
- 旧开发目录中部分 launcher/checkpoint 已被清理，不能据此反推正式 Table 4 有误。先前基于旧目录提出的 small-model provenance 怀疑应撤回。
- Table 4 本身已清楚报告参数比例，足以支持实际配置下的 performance--parameter trade-off；它不需要被重新解释为 equal-budget 实验。

## 3. 其余原始下游结果分别证明了什么

### 3.1 Vocabulary-boundary layer 本身有 standalone 适配价值

Qwen3-0.6B：

- input A-LoRA r16：34k 参数，loss 1.419；
- bilateral A-LoRA r16：67k 参数，loss 1.320；
- 最强的 parameter-comparable single-layer qkvo LoRA：41k 参数，loss 1.473。

因此，34k 的 input A-LoRA 比 41k single-layer baseline 低 0.054；这条结果支持 vocabulary boundary 不是“随便加少量参数”就能替代的普通内部位置。

### 3.2 Standalone 小预算下 input 更强

| 模型 | input | lm_head | input−lm_head |
|---|---:|---:|---:|
| Qwen2.5-0.5B | 1.816 | 1.907 | -0.091 |
| Qwen3-0.6B | 1.460 | 1.621 | -0.161 |
| Qwen2.5-1.5B | 1.502 | 1.608 | -0.106 |
| Qwen2.5-7B（untied） | 1.258 | 1.392 | -0.134 |

这条结果说明 standalone capacity 不等同于静态 residual fit：input 扰动会穿过整个 Transformer，因此即使 untied input 的静态 affine fit 较低，standalone input A-LoRA 仍可能更强。

### 3.3 已有 hidden LoRA 时，lm_head 是更有效的低成本补充

原主表中，`lm_head + hidden` 相对 `input + hidden` 在 0.5B、0.6B、1.5B、1.7B 和 7B 上均更低，差距约 0.003–0.009；0.5B/0.6B 的 seeds 42/43/44 六组比较方向全部一致。Qwen3-0.6B 上 mergeable tied + hidden 为 1.094，hidden baseline 为 1.101，接近 lm_head + hidden 的 1.093。

这部分的正确定位是 incremental supplement，而不是独立主方法与完整 hidden LoRA 的替代关系。

## 4. 未进入投稿正文、但原项目中存在的更强参数效率矩阵

归档 `canonical_alora_efficiency_20260525_231722` 还有一组更完整的旧结果：在相同 bilateral placement 下比较 A-LoRA r16 与 Vocab LoRA r2，覆盖 Qwen2.5-0.5/1.5/7B 和 Qwen3-0.6/1.7/8B，每个模型三个 seeds。

| 模型 | A-LoRA mean ± sample SD | Vocab LoRA mean ± sample SD | Vocab/A 参数倍率 | 逐 seed |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | 1.6377 ± 0.0006 | 1.8790 ± 0.0010 | 10.50× | 3/3 A 更低 |
| Qwen3-0.6B | 1.3250 ± 0.0000 | 1.5457 ± 0.0058 | 9.19× | 3/3 A 更低 |
| Qwen2.5-1.5B | 1.3253 ± 0.0023 | 1.5910 ± 0.0046 | 6.15× | 3/3 A 更低 |
| Qwen3-1.7B | 0.9117 ± 0.0025 | 1.1933 ± 0.0051 | 4.63× | 3/3 A 更低 |
| Qwen2.5-7B | 1.1183 ± 0.0031 | 1.4223 ± 0.0045 | 2.67× | 3/3 A 更低 |
| Qwen3-8B | 1.0103 ± 0.0015 | 1.2780 ± 0.0026 | 2.35× | 3/3 A 更低 |

这组结果在原项目里非常有价值：topology 一致、覆盖到 untied 7B/8B、18/18 seeds 方向一致，而且 A-LoRA 始终参数更少。它没有进入当前投稿正文；最可能的原因是它属于 cross-rank trade-off（A r16 vs Vocab r2），而后续正文转而突出 input-only placement 与更高的参数倍率。

这组归档结果没有进入正式投稿 Table 4，因此不需要在 rebuttal 中主动引入。针对 Reviewer 9sSY 的比较口径意见，正式 Table 4 已经足以支持论文所主张的 performance--parameter efficiency；额外的 corrected-pipeline 重跑可以增强统计证据，但不是回应“严格同预算”质疑的逻辑前提。

## 5. 当前最准确的论文定位

原论文的核心主张应该保留为：

> A-LoRA 在 vocabulary boundary 上提供了显著更好的效果—参数量折中：它用主要依赖 hidden dimension 而非 vocabulary size 的参数化，在多组原始实验中以远少于 Vocab LoRA 的参数达到相当或更低的 loss。

同时必须收窄两点：

1. 这是 performance–parameter trade-off，不是原 Table 4 已经证明了 equal-budget superiority；
2. A-LoRA 是 affine-friendly residual 下的条件性参数化，不是对所有 vocabulary matrices、模型和任务的无条件替代。

## 6. 对 rebuttal 的直接含义

- 不应把原论文弱化成“只证明参数量公式”；原 Table 4 的确已经报告了更少参数且 5/6 更低 loss。
- 回应 Reviewer 9sSY 时应承认标签和 topology 报告不够清楚，并重新列绝对参数量、rank、target modules 与 topology。
- 针对“并非严格同预算”的意见，应首先澄清 Table 4 的研究问题是效果—参数量折中：少 18--152 倍参数且 6 组中 5 组 loss 更低已经构成低参数方向上的 Pareto dominance。
- Equal-budget downstream 实验回答的是相同预算下的性能上限，是不同且可选的补充问题；不能把它写成 Table 4 参数效率结论成立的必要条件。
