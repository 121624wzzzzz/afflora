# Qwen 共享预算与 untied 单侧消融：最终记录

共享实验仅包含 tied 的 Qwen2.5-1.5B、Qwen3-0.6B；untied 的7B、8B共享组已按用户要求撤销。24次新正式训练、8次短训练、34068条新输出均通过审计。单侧消融复用历史训练，经重新核验，不计为新增独立重复。

## Tied：共享 rank16 / rank32

额外 E/U 参数：共享r16为33d，共享r32为65d；独立E/U各r16为65d。前者减少的是额外边界参数，hidden LoRA参数不变。共享输出使用转置，保持有效权重绑定；两档alpha/rank均为8。

|模型|任务|H|等预算 H|独立 E/U|共享16|共享32|
|---|---|---:|---:|---:|---:|---:|
|qwen25_15b_base|cluener|59.002|59.054|62.247|61.723|62.428|
|qwen25_15b_base|wikisql|70.475|70.801|72.786|72.135|72.201|
|qwen3_06b_base|cluener|60.975|61.144|62.034|62.146|62.446|
|qwen3_06b_base|wikisql|70.964|71.224|72.396|72.298|72.070|

两档共享方案在全部四个模型×任务组合上均高于H及等预算H的均值；相对独立双侧并非统一更优。共享32在两个CLUENER组合略高于独立双侧，在两个WikiSQL组合较低。共享16在三个组合均值低于独立双侧，一个较高。三个种子及固定共同学习率不足以支持普遍优越性或等效性结论。

## Untied：分别移除 E 或 U

Qwen2.5-7B、Qwen3-8B各跑两任务、三种子。比较H / H+E / H+U / H+E+U，同时保留无H的E-only / U-only / E+U。48个单侧历史训练重新验证：没有另一侧参数，只有预期参数发生更新，优化器白名单、冻结基座、rank与保存张量均通过。

单侧完整表：[VERIFIED_UNTIED_ZH.md](VERIFIED_UNTIED_ZH.md)。范围检查：[UNTIED_SINGLE_SIDE_SCOPE_AUDIT.json](UNTIED_SINGLE_SIDE_SCOPE_AUDIT.json)。

这类移除组件的消融预算不同，可以说明各组件贡献，但不能独立证明双侧的同预算优势。Qwen3-8B WikiSQL中H+U与双侧三种子均值相同；不应宣称双侧总比单侧强。

审计：[study/FINAL_AUDIT.json](study/FINAL_AUDIT.json)。原始配对结果：[study/RESULTS.json](study/RESULTS.json)。
