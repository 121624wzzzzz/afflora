# 边界普通 LoRA 对照：阶段结果
仅纳入通过独立审计的结果；未完成种子组的均值不是最终结论。调参只读 dev，不列入此表。

|阶段|模型|任务|方法/位置/rank/bias|n|均值|复用|
|---|---|---|---|---:|---:|---:|
|confirmation|qwen3_06b_base|clinc150|affine/shared/16/True|5|83.4800|0|
|confirmation|llama32_1b_base|clinc150|vocab/shared/1/False|5|83.7467|0|
|confirmation|llama32_1b_base|clinc150|affine/shared/16/True|5|84.2578|0|
|confirmation|llama32_1b_base|clinc150|none/none/0/False|5|83.2133|0|
|confirmation|qwen3_06b_base|clinc150|none/none/0/False|5|82.8133|0|
|confirmation|qwen3_06b_base|clinc150|vocab/shared/1/False|5|82.0267|0|
