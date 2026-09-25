# 边界普通 LoRA 对照：阶段结果
仅纳入通过独立审计的结果；未完成种子组的均值不是最终结论。调参只读 dev，不列入此表。

|阶段|模型|任务|方法/位置/rank/bias|n|均值|复用|
|---|---|---|---|---:|---:|---:|
|confirmation|llama32_1b_base|clinc150|none/none/0/False|5|83.0667|0|
|confirmation|qwen25_15b_base|clinc150|none/none/0/False|5|84.2000|0|
|confirmation|llama32_3b_base|emotion|affine/shared/16/True|5|73.9000|0|
|confirmation|llama32_3b_base|clinc150|none/none/0/False|5|91.0800|0|
|confirmation|llama32_3b_base|emotion|none/none/0/False|5|73.3200|0|
|confirmation|llama32_1b_base|clinc150|affine/shared/16/True|5|83.9289|0|
|confirmation|llama32_1b_base|emotion|vocab/shared/1/False|4|74.1000|0|
|confirmation|qwen25_15b_base|emotion|none/none/0/False|5|72.1500|0|
|confirmation|qwen25_15b_base|emotion|vocab/shared/1/False|5|71.5100|0|
|confirmation|qwen25_15b_base|clinc150|affine/shared/16/True|5|84.9511|0|
|confirmation|qwen25_15b_base|clinc150|vocab/shared/1/False|5|84.1511|0|
|confirmation|llama32_3b_base|clinc150|affine/shared/16/True|5|91.0978|0|
|confirmation|qwen25_15b_base|emotion|affine/shared/16/True|5|71.8900|0|
|confirmation|llama32_1b_base|clinc150|vocab/shared/1/False|5|83.0711|0|
|confirmation|llama32_1b_base|emotion|affine/shared/16/True|5|77.7200|0|
|confirmation|llama32_3b_base|clinc150|vocab/shared/1/False|5|91.1600|0|
|confirmation|llama32_1b_base|emotion|none/none/0/False|5|73.7400|0|
|confirmation|llama32_3b_base|emotion|vocab/shared/1/False|5|72.8800|0|
