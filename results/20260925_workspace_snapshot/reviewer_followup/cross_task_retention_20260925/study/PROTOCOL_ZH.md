# 跨任务能力保留 / 遗忘评测（2026-09-25）

## 本轮问题
不再对更多下游任务做训练。固定已完成CLINC150、WikiSQL训练出的adapter，直接评测非目标任务，检查目标任务收益是否以其他能力下降为代价。与此前“在新任务重新训练”的泛化实验严格区分。

## 固定模型、来源和方法
Qwen3-0.6B、Llama3.2-1B pretrained tied base。源任务CLINC150使用clinc150_generalization_20260925完整五seed确认，WikiSQL使用tied_model_expansion_20260923完整五seed确认，复用路径逐一解析到实际checkpoint并核对源审计和hash。源任务只依据其原dev选择配置，不根据保留评测挑选adapter。每模型每源任务H、H+shared普通r1、H+shared aLoRA r16各5seed，共60checkpoint；每模型原始base评测一次，共62个评测作业。没有训练、梯度、adapter合并或新LR搜索。

早先做过SciQ/ANLI任务探索；它们不是本研究从未看过的保密基准。本轮checkpoint没有在这两项任务上训练。本地输入文本规范化去重审计检查源训练文本与保留数据是否精确重合，不等于证明语义或预训练污染不存在。

## 非目标任务与评分
SciQ完整1000条test，4个答案；ANLI R1完整1000条test，3类关系。使用统一multiple-choice prompt和原生BOS/EOS。候选字符串为单token的空格加字母，逐样本验证prompt+候选的token拼接一致，不截断输入。

所有选项作循环轮换（SciQ4次、ANLI3次），预测各候选字母的下一token log probability。每轮先在候选集合softmax，映射回语义选项后对各轮概率求均值，再argmax；并列取原选项索引最小。主指标为轮换集成准确率。另报告逐轮准确率、正确语义选项的平均概率以及候选总概率质量。后者帮助发现模型更想输出训练任务格式的情况。

这是“给定答案集合”的任务能力保留测量，不等同于自由生成合规性、开放式回答质量或全部能力。不能把候选归一化掩盖的输出偏好变化称为完全无遗忘。评测float32、无采样、不训练。每条存原始候选logprobs，可独立复算。

## 技术复用与审计
准备时核验源模型SHA、源冻结代码、初始与最终adapter文件SHA及原逐run审计、训练顺序/参数元数据与源test输出hash。目标任务成绩只从审计通过的源结果引用，不重复拼接不同seed。每次评测检查模型冻结、无梯度、adapter载入tensor digest相同；独立复算轮换映射和准确率，预测前先用两例比较HF完整forward和仅最后token计算的logits，及base/adapter disabled一致性。输出与元数据独立目录，源目录不修改。

## 分析
每个模型×源任务×保留任务给出base、H、普通边界、aLoRA分数、相对base下降幅度，同时报告原目标任务分数。
8个条件的aLoRA−普通为Holm8族，aLoRA−H为另Holm8族；五seed配对t区间。对base差值亦给五seed区间，不能把共同SFT下降全归于aLoRA。主要关注A−H/A−普通是否额外下降。无显著差异不是等效证明，正负结果全部保留。

## 资源
不抢占正在运行的训练/base队列；等待generalization_repaired_20260925的FINAL_AUDIT与BASE_AUDIT通过后，仅使用空闲GPU，每卡一个评测worker。中途失败保留并报警，不按效果停止。全部结果保存本项目目录；不新增模型或adapter权重。
