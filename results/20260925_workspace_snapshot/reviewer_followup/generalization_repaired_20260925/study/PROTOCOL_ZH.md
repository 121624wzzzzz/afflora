# 泛化评测修复与扩展（2026-09-25）

## 状态与边界
CLINC150 已查看过旧 test；本轮属于事后修复与新seed/新尺寸重复，不声称全新盲测。Emotion 为新增数据集，协议在本轮模型评测前固定。两者都是英文分类，不能外推所有任务、语言或开放集能力。保留旧目录、旧严格评分、正负结果，不按本轮结果更换任务。

## 评分修复
Prompt和训练目标仍要求JSON字符串，保持CLINC旧prompt逐字一致。主指标为准确内容标签：允许完整JSON字符串、精确裸标签、带Intent:/Emotion:前缀的精确标签或所有非空行重复同一精确标签。解释性文字、不同标签并列、子串/模糊匹配、同义词、大小写猜测全部拒绝。不接收gold作为解析条件。另完整报告原严格JSON准确率、严格格式合法率、内容合法率、宏F1与输出截断率。

历史重评分仅对现有所有confirmation和base输出统一重算，明确标为posthoc。逐条核对原预测SHA/token解码、旧严格正确性保持及独立regex解析+sklearn指标；不能根据重评分改写旧选参/显著性结论。

## 数据与模型
四个固定pretrained tied base：Qwen3-0.6B、Qwen2.5-1.5B、Llama3.2-1B、Llama3.2-3B。新数据由官方dair-ai/emotion固定revision的split parquet获得，六类情绪，不使用unsplit语料。CLINC精确保留旧2048train/1500dev/4500test和150语义标签，不纳入OOS。Emotion按test>dev>train优先级规范化文本去重，官方dev/test全部保留去重后数据，train固定hash取2048条；不按模型效果筛选，保留类别计数与排除记录。

重验模型文件SHA、原生BOS/EOS和完整token编码、split输入无交叉、评分器正确/错误/非法案例。无截断训练样本。原始base也使用同一prompt与两套评分；0格式分不解释为0分类能力。

## 公平搜索
H固定r8/alpha16/dropout.05/LR2e-4，batch32、2048例/64步。方法H+shared aLoRA r16（E bias）与H+shared普通词表LoRA r1。报告边界和总参数，非严格等参。四模型×两任务，每方法搜索7档边界倍率[1/256,1/64,1/16,1/4,1,4,16]×3个dev-only种子，方法间预算一致，总336次。H学习率未独立优化，结论限定于此训练预算。

Dev seed CLINC1107100..1107102，Emotion1108100..1108102；每架构按dev内容准确率均值选LR，平局倍率距1最近，再较小。完整dev审计后冻结选择，test不参与。即便再次选中网格边缘，也完成并报告本协议，不根据test追加直到获胜。

确认CLINC1207100..1207104、Emotion1208100..1208104：两边界配置加同seed H-only，8个模型任务×15=120正式训练。24个smoke覆盖所有架构/任务；共480训练作业。另8个原始base模型任务评测。训练全部新跑，不混用旧选择或旧test均值。

## 统计和审计
8个模型任务的aLoRA−普通LoRA为Holm8一族，aLoRA−H为另一Holm8族，配对5seed t区间，必须配对完整才统计。严格评分/宏F1为辅助诊断，不择优替代主指标。所有架构以相同seed的H初始化、数据次序配对。每run检查基座冻结、零残差、mask、scope与参数数、shared两侧调用、保存重载、最终hash、逐条token与独立解析指标；包含正负结果。

初始/最终adapter与预测保存到/home/wz/experiment_artifacts/generalization_repaired_20260925，预计adapter约29GiB，产物盘留40GiB/项目盘留5GiB保护。旧产物只读，单独FINAL_AUDIT训练审计、BASE_AUDIT基础模型审计、历史重评分AUDIT。
