# 边界普通 LoRA 对照：阶段结果
仅纳入通过独立审计的结果；未完成种子组的均值不是最终结论。调参只读 dev，不列入此表。

|阶段|模型|任务|方法/位置/rank/bias|n|均值|复用|
|---|---|---|---|---:|---:|---:|
|confirmation|qwen3_4b_base|wikisql|affine/shared/30/True|5|82.3633|0|
|confirmation|qwen3_4b_base|wikisql|vocab/shared/1/False|5|82.4609|0|
|confirmation|qwen3_4b_base|wikisql|affine/shared/16/True|5|82.0508|0|
|confirmation|qwen3_4b_base|wikisql|vocab/shared/2/False|5|82.6562|0|
|confirmation|qwen3_4b_base|wikisql|none/none/0/False|5|81.4258|0|
