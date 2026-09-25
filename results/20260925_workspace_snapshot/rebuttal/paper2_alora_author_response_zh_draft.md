# A-LoRA 作者回复中文稿（第一版）

## 共同说明：A-LoRA 的核心定位与比较口径

> A-LoRA 并非对所有 vocabulary-boundary matrices 都普遍更优的替代方法。本文的核心结论是：当目标矩阵具有 affine-friendly 的更新结构时，A-LoRA 能够以主要依赖隐藏维度 $d$、而不随词表大小 $V$ 线性增长的参数量，为大规模 vocabulary-boundary matrices 提供高效适配。对于单个 $V\times d$ 词表矩阵，直接 Vocab LoRA 的参数量为 $O(r(V+d))$，而 A-LoRA 的乘性部分仅为 $O(dr)$。在 Table 4 所报告的具体 rank 和 topology 下，A-LoRA 的实际 vocabulary-adapter 参数量约为对照的 $1/18$ 到 $1/152$，并在多数配置上取得相近或更低的 validation loss。由于这些比例同时受到 rank、单边/双边 topology 和目标模块数影响，修订稿将完整列出 target modules 与绝对参数量，而不会把全部倍率简单归因于参数化形式本身。
>
> 修订稿将把“显著的参数效率”与“条件性的下游收益”明确区分，并将结果逐项标为 rank-controlled、cross-rank parameter-efficiency 或 low-cost supplement。Table 4 不构成严格的 equal-budget 比较；只有在总可训练参数确实匹配时，我们才会使用 equal-budget 标签。我们也将同步修正正文实验逻辑表和方法部分中容易让人误以为 Table 4 已经 parameter-matched 的表述。论文的核心主张是 A-LoRA 的参数缩放性质和性能—参数折中，而不是 equal-budget superiority，也不是在所有模型和任务上获得无条件的性能优势。

## 共同说明：补充的多种子统计与任务评测

> 我们在严格隔离的 test set 上补充了 seeds 42/43/44 的配对实验，比较 hidden LoRA rank 8 与相同 hidden-LoRA 初始化加 bilateral input/lm_head A-LoRA rank 16。在 Qwen3-0.6B 上，跨三个 paired-seed deltas 的 mean 为 $\Delta\mathrm{CE}=-0.005219$，sample SD 为 $0.000315$，seed-level 95% $t$ interval 为 $[-0.006001,-0.004437]$，3/3 seeds 方向一致；对应 PPL 约由 3.369 降至 3.352。在 Qwen2.5-1.5B 上，mean $\Delta\mathrm{CE}=-0.007343$，sample SD 为 $0.000189$，同样为 3/3 seeds 改善。两组实验中，每个 seed 的 paired item-bootstrap 95% 置信区间均严格低于 0；该 item-level 区间与跨训练 seed 的统计将分开报告。
>
> 我们还补充了实际任务指标。在 Qwen2.5-1.5B 上，我们使用 seeds 45/46/47 比较 hidden LoRA rank 4 与 hidden LoRA rank 4 加 tied shared/mergeable A-LoRA rank 16。相对 4.616M-parameter hidden-LoRA baseline，A-LoRA 额外增加约 50.7K 参数（约 1.1%）；GSM8K 的 paired-seed delta mean 为 $+0.6065$ 个百分点，sample SD 为 $0.0758$，seed-level 95% $t$ interval 为 $[+0.4182,+0.7949]$，3/3 seeds 均为正向变化。修订稿将同时报告 CE、PPL、任务准确率、逐种子结果、标准差和置信区间；任务级结论仅限定在相应模型与配置内。

## Reviewer yP5t

感谢审稿人认可 A-LoRA 的 hidden-space affine parameterization、其在 tied embedding 场景中的实用价值，以及本文对 input/output/tied placement 的分析。我们将重点从以下四方面修改论文。

### 1. 关于 A-LoRA 的核心贡献与参数效率

A-LoRA 的关键优势是避免在词表维度上学习自由低秩因子。对于 $V\times d$ 的 vocabulary matrix，直接 Vocab LoRA 的参数量为 $O(r(V+d))$，而 A-LoRA 的乘性部分仅为 $O(dr)$。因此，当 $V\gg d$ 时，A-LoRA 的参数量不随词表规模线性增长。

原 Table 4 的主要目的正是展示这种性能—参数折中，而不是声称 A-LoRA 在任意预算和任务上都优于不受结构约束的 Vocab LoRA。在当前结果中，A-LoRA 使用约 $1/18$ 到 $1/152$ 的 vocabulary-adapter 参数，在多数配置上取得相近或更低的 validation loss。例如，在 Qwen3-0.6B 上，rank-2/4 A-LoRA 分别使用约 $1/120$ 和 $1/133$ 的参数，loss 从 Vocab LoRA 的 1.529/1.495 降至 1.494/1.469；在 1.5B/1.7B 模型上，A-LoRA rank 16 分别只使用 Vocab LoRA rank 4 约 $1/24$ 和 $1/18$ 的参数，同时取得更低的 loss。

修订稿将为每一项比较明确列出 rank、topology、绝对参数量和参数比例，并将其准确标记为 equal-rank 或 low-cost parameter-efficiency comparison，避免将该结果误解为无条件的性能优势。

### 2. 关于小幅 loss 改善的统计可靠性及 hidden LoRA 上的增量

我们同意，单次运行不足以判断 Table 7 中的小幅差异是否超过随机种子波动。为此，我们使用严格隔离的 test set，在完全相同的训练数据、训练步数和 hidden-LoRA 初始化下，补充了三个随机种子的配对实验。

在 Qwen3-0.6B 上，hidden LoRA 加入 A-LoRA 后：

$$
\Delta\mathrm{CE}=-0.005219\pm0.000315,
\qquad
95\%\ \mathrm{CI}=[-0.006001,-0.004437].
$$

三个种子均获得改善，每个种子的 paired bootstrap 95% 置信区间也均严格低于 0；对应 PPL 约由 3.369 降至 3.352。在 Qwen2.5-1.5B 上，三个种子的平均变化为 $\Delta\mathrm{CE}=-0.007343\pm0.000189$，同样为 3/3 seeds 改善，且每个种子的 paired bootstrap 置信区间均低于 0。

这些结果说明：A-LoRA 在 hidden LoRA 上带来的增量幅度虽然不大，但在上述配置中能够稳定超过种子波动。我们会将原文表述修改为“稳定的低成本增量”，而不将其描述为幅度很大的质量提升。同时，修订稿将对核心结果统一报告 mean、standard deviation、95% confidence interval 和逐种子方向。

### 3. 关于 perplexity 和实际任务指标

我们将同时报告 CE 和 PPL，以提高结果的可解释性；同时会明确说明，PPL 是 CE 的单调变换，而不是一项独立的质量证据。

我们还补充了实际任务评测。在 Qwen2.5-1.5B 上，将 mergeable A-LoRA rank 16 作为 hidden LoRA 的低成本补充，只增加约 50.7K 个可训练参数（约为 hidden LoRA 参数量的 1.1%）。在 GSM8K 上，三个新随机种子的平均提升为

$$
+0.6065\pm0.0758\ \text{个百分点},
\qquad
95\%\ \mathrm{CI}=[+0.4182,+0.7949],
$$

并且 3/3 seeds 均为正向变化。这表明，至少在代表性配置中，validation-loss 改善能够转化为可测量的任务收益。我们不会把该结果外推为所有任务上的普遍提升，而会将任务级结论严格限定在相应模型和配置中。

### 4. 关于 $R^2_{\mathrm{aff}}$、论文组织与复现性

我们同意 Table 20 中的 affine explained variance 在不同模型和矩阵之间并不均匀。这里需要澄清，$R^2_{\mathrm{aff}}$ 衡量的是一个已经观察到的 Base$\rightarrow$Instruct 权重差分与仿射形式的结构匹配程度；实际下游收益还受到目标数据、优化过程、adapter rank、placement 以及已有 hidden adaptation capacity 的共同影响。因此，$R^2_{\mathrm{aff}}$ 应被理解为结构适用性的诊断量，而不是下游性能提升的单变量预测器。修订稿将删除任何可能暗示该结构是普遍性质、或 $R^2_{\mathrm{aff}}$ 能直接预测任务收益的表述。

我们也接受审稿人关于论文组织和复现性的建议。修订稿将：

- 在首次使用前定义 $r_{\mathrm{aff}}$ 和 $r_W$，并调整 Eq. 4 的位置；
- 重组方法与结构分析的顺序，使参数化形式在相关讨论前得到完整定义；
- 补充数据划分、随机种子、训练轮数、optimizer、学习率、scheduler、batch size、gradient accumulation、LoRA/A-LoRA rank 与 scaling、checkpoint selection 和评测流程；
- 对核心多种子结果统一报告 mean、standard deviation 和 95% confidence interval。

通过上述修改，论文的结论将更加集中：A-LoRA 的主要价值在于，在适用的结构条件下，以一到两个数量级更少的 vocabulary-boundary 参数获得有竞争力的适配效果。

## Reviewer 9sSY

感谢审稿人准确概括本文的研究价值，特别是肯定 vocabulary-boundary placement 的重要性、A-LoRA 将参数规模从词表维度 $V$ 转移到隐藏维度 $d$ 的优势，以及区分 standalone 与 hidden-LoRA-conditioned 两种适配场景的必要性。

### 1. 关于 affine structure 的适用范围

我们同意，Base$\rightarrow$Instruct 权重差分中观察到的仿射结构是一种经验规律，而不是所有模型、任务和训练范式都满足的普遍性质。本文希望表达的更准确结论是：

> 当目标 vocabulary-boundary matrix 的更新具有足够明显的 affine-friendly structure 时，A-LoRA 提供了一种参数量主要依赖隐藏维度、而不随词表大小线性增长的高效参数化。

修订稿将把 affine-friendliness 明确写为方法的适用条件，并相应收窄摘要、引言和结论中的跨模型表述。对 30 个 Base$\rightarrow$Instruct 模型对的分析用于说明这一结构在现有模型中反复出现，而不再被表述为所有下游适配任务的普遍保证。

### 2. 关于 A-LoRA 与 Vocab LoRA 的比较口径

感谢审稿人建议进一步区分不同的比较口径。我们首先澄清，Table 4 的研究问题并不是“在严格相同参数预算下，哪一种参数化具有更高的性能上限”，而是“达到相当或更好效果时，哪一种 vocabulary-boundary adapter 所需的参数更少”。因此，该表在 `Param ratio` 一列中有意明确报告双方的实际参数比例，而没有将它们描述为相同预算。

Table 4 给出的结果形成了直接的性能—参数量优势：A-LoRA 仅使用 Vocab LoRA 约 $1/18$--$1/152$ 的 vocabulary-adapter 参数，却在 6 组配置中的 5 组取得更低的 validation loss。例如，在 Qwen3-0.6B 的 rank-controlled 比较中，rank-2 A-LoRA 使用 5,120 个参数，而 Vocab LoRA 使用 611,840 个参数，loss 为 1.494 vs. 1.529；rank-4 时分别使用 9,216 vs. 1,223,680 个参数，loss 为 1.469 vs. 1.495。类似地，在 Qwen2.5-1.5B 和 Qwen3-1.7B 上，A-LoRA 分别使用 50,688 和 67,584 个参数，而 Vocab LoRA 使用 1,227,776 和 1,231,872 个参数，A-LoRA 的 loss 仍分别更低 0.065 和 0.164。换言之，其中五组配置形成了低参数方向上的严格 Pareto dominance；严格 equal-budget 比较回答的是另一个有价值、但不同的问题，并不是本文参数效率结论成立的必要条件。

我们会在修订稿中把几类证据标注得更加明确：Table 4 的前四组是 rank-controlled performance--parameter comparisons，后两组是 cross-rank performance--parameter comparisons；Table 7 讨论的是 A-LoRA 在 hidden LoRA 之上的 low-cost supplementation；严格预算匹配只在参数确实匹配的分析中使用该标签。我们也会将 Table 3 中可能造成误解的简写 “parameter budget matched” 改为 “performance--parameter trade-off under actual trainable-parameter costs”，并在表中继续报告 rank、target modules、绝对参数量和 `Param ratio`。这一文字修正不会改变 Table 4 已经支持的核心结论：A-LoRA 能够以少一到两个数量级的 vocabulary-boundary 参数获得相当或更好的适配效果。

### 3. 关于统计证据和实际任务质量

我们补充的三种子配对实验表明，在 Qwen3-0.6B 上，hidden LoRA 加入 A-LoRA 后的变化为

$$
\Delta\mathrm{CE}=-0.005219\pm0.000315,
\qquad
95\%\ \mathrm{CI}=[-0.006001,-0.004437],
$$

3/3 seeds 均获得改善，对应 PPL 约由 3.369 降至 3.352。在 Qwen2.5-1.5B 上，三个种子的平均变化为 $-0.007343\pm0.000189$，同样为 3/3 seeds 改善。

对于实际任务指标，Qwen2.5-1.5B 上的 mergeable A-LoRA rank 16 仅为 hidden LoRA 增加约 1.1% 的可训练参数，却在 GSM8K 上获得 $+0.6065$ 个百分点的三种子平均提升，95% CI 为 $[+0.4182,+0.7949]$，且三个种子方向一致。修订稿将同时报告逐种子结果、均值、标准差和置信区间。

与此同时，我们会避免把上述结果外推为所有任务均会提高。本文更核心且更稳定的结论，是 A-LoRA 在 vocabulary boundary 上的显著参数效率；具体任务增益仍取决于模型、任务和 placement。

### 4. 关于实验覆盖范围

我们同意，当前训练实验主要验证 Qwen 系列上的 instruction tuning，因此不足以支持对代码生成、多语言迁移、词表扩展、领域继续预训练或长上下文适配的普遍结论。修订稿将明确区分：

- 跨模型权重分析揭示的结构现象；
- 当前在 Qwen instruction-tuning 设置下获得的下游验证。

我们将删除或收窄超出这一范围的泛化表述，并将其他模型家族和非 instruction-tuning 场景列为后续验证方向。本文当前希望建立的是一种新的 vocabulary-boundary 参数化及其参数缩放优势，而不是声称已经覆盖所有可能的应用场景。

### 5. 关于其他轻量 vocabulary-boundary baselines

本文选择 Vocab LoRA 作为主要直接基线，是因为它与 A-LoRA 都对完整 vocabulary matrix 建模，两者的主要区别是：更新直接定义在词表空间，还是由隐藏空间的共享低秩变换诱导。Trainable-token 和 partial-embedding 方法通常还要求预先选择需要训练的 token，针对的是局部词表编辑；token-wise output bias 主要调整 token prior。它们与本文研究的 dense shared transformation 并不完全等价。

我们将在相关工作和实验说明中进一步澄清这些方法的目标、参数化和适用场景差异，避免将当前比较解释为对所有 vocabulary adaptation 方法的全面覆盖。

## Reviewer dBPd

感谢审稿人认可本文对 vocabulary-boundary placement 的统一分析、A-LoRA 的参数效率，以及不同 adaptation regime 下 placement 选择的差异。审稿人指出的两个问题有助于我们进一步明确方法的数学形式和适用边界。

### 1. 关于 output-side $\beta$ 项

审稿人的判断是正确的。对于 output-only A-LoRA，令 $\gamma=\alpha/r$，其 logits 为

$$
z=M_0(I+\gamma PQ)h+(\beta^\top h)\mathbf 1_V.
$$

记 $c=\beta^\top h$，则标准 softmax cross-entropy 满足

$$
\begin{aligned}
\mathcal L(z+c\mathbf 1_V,y)
&=-(z_y+c)+\log\sum_v\exp(z_v+c)\\
&=-z_y+\log\sum_v\exp(z_v)
=\mathcal L(z,y).
\end{aligned}
$$

因此，$\beta$ 在纯 output 路径上只给所有 vocabulary logits 加上相同标量，不改变 softmax 分布、预测结果或标准语言模型损失，也不提供有效的 output-side 建模能力。我们将在修订稿中：

- 从 output-only A-LoRA 以及 decoupled topology 的独立 output 分支中移除 $\beta$；
- 将 output-only A-LoRA 的参数量由 $d+2dr$ 修正为

  $$
  P_{\mathrm{aff}}^{\mathrm{out}}=2dr;
  $$

- 修正文中和附录中将该项描述为 output-side bias change 的表述。

这一修正不会改变标准 softmax 目标下已有 output-only 实验的预测分布和评测结果，只会使其名义参数量进一步减少 $d$。因此，原稿对 output-only A-LoRA 参数效率的统计是保守的。

$\beta$ 在 input-side 的作用不同。对于输入 token $x$，

$$
e_x=M_0[x,:](I+\gamma PQ)+\beta,
$$

该平移会进入后续 Transformer 层并改变隐藏状态，因此不会被 softmax shift invariance 消除，应予以保留。对于 tied/shared A-LoRA，同一个 $\beta$ 同时出现在 input 和 output 路径中：其 output 路径上的直接贡献仍会抵消，但可以通过 input 路径影响后续隐藏表示。因此，input-only 和 tied/shared 配置仍保留 $\beta$，其参数量仍为 $d+2dr$。

### 2. 关于 A-LoRA 的条件适用性

我们同意，A-LoRA 不应被表述为适用于所有 vocabulary-boundary matrices 的通用替代方法。本文的结构分析本意正是提供一个适用性诊断：A-LoRA 更适合目标更新能够较好地由

$$
M_0A+\mathbf 1_Vb^\top
$$

描述的 affine-friendly matrix。现有分析表明，output LM head 和 tied/shared matrix 通常更符合这一结构，而 untied input embedding 不应仅因具有相同的 $V\times d$ 形状就被默认对称处理。

修订稿将在摘要、方法和结论中明确这一适用前提，并将 $R^2_{\mathrm{aff}}$ 表述为结构匹配程度的诊断量，而不是下游性能提升的充分保证。这一收窄不改变 A-LoRA 的主要效率贡献：对于 affine-friendly 的 vocabulary-boundary matrix，A-LoRA 能够以显著更低、且不随词表大小线性增长的参数成本提供适配能力。
