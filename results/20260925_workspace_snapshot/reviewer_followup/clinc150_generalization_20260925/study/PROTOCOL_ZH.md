# CLINC150 新数据集泛化验证（2026-09-25）

冻结在查看任何本任务模型指标前。CLINC150原始官方域内150类；OOS不纳入，本实验不声称开放集检测能力。本地既有Banking77结果不是新发现；本轮检验另一个多领域意图数据集，不能外推任意任务泛化或证明预训练无污染。

固定模型Qwen3-0.6B、Llama3.2-1B pretrained tied base，不根据本轮效果增删模型。输入列出150个语义标签，输出精确JSON字符串；相同prompt、greedy解码、nativeEOS/BOS、masked token loss。主指标完整test准确率，宏F1与合法输出率辅助；解析失败算错，不约束解码掩盖格式错误。

官方train中固定hash分层选择2048条（每类13条，hash固定的98类额外1条）；dev每类固定10条共1500；test保留完整官方域内test。跨split规范化输入去重采用test>dev>train优先级，排除清单保留，绝不基于模型错误筛样本。只用train确定标签列表并检查150类。训练64步batch32，无基于test早停。

H固定r8 alpha16 dropout.05 LR2e-4。主要对照H+shared aLoRA r16（E bias）和H+shared普通词表LoRA r1，另H-only及原始base。报告真实边界及总参数，不称严格等参。

两种方法边界LR倍率[1/64,1/16,1/4,1]，每档dev-only三个种子807100..807102（共48训练）。按dev准确率均值选倍率，平局距1近再小倍率，全部dev审计通过后冻结配置。确认五个新种子907100..907104，每模型两边界配置及H-only共30训练；6smoke覆盖全部训练架构。共84训练作业，另2个独立原始base评测。Base成绩不改变方案。

跨两个模型的aLoRA−普通LoRA为Holm2比较族；aLoRA−H另Holm2。报告所有seed和正负结果，不按test选择rank/模型/任务。新任务不复用旧任务指标。技术smoke失败才修实现，性能差不取消。最终审计逐条token解码及独立评分，初始/最终adapter、数据split/hash、模型权重hash、冻结基座与重载检查均保留。

训练产物存/home/wz/experiment_artifacts/clinc150_generalization_20260925。容量保护40GiB产物盘/5GiB项目盘。现有任务原始目录和结果不修改。
