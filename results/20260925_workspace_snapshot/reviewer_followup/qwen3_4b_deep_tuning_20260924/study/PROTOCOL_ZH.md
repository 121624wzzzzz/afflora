# Qwen3-4B 深入边界调参（2026-09-24）

本轮为看到此前 test 差异后发起的诊断性后续实验，不能称为未接触过测试任务的独立发现。旧结果保留，新代码与产物独立目录。

## 固定范围
仅 Qwen3-4B pretrained tied base，CLUENER 与 WikiSQL。H 固定 rank8/alpha16/dropout0.05、LR2e-4；训练2048例/64步/batch32。复用已审计的训练和评分实现、原数据及 token，但重新校验模型文件哈希、逐条 native tokenizer 编码、split disjoint。所有训练从头运行，不复用指标。

边界四配置：shared aLoRA r16/r30（带E bias），shared普通词表LoRA r1/r2。d=2560,V=151936，边界参数量分别84480/156160/154496/308992；r30比普通r1多1.077%，明确为近等参而非严格等参。scale固定8，H配置不变。

## Dev选择与确认
每架构每任务搜索边界倍率[1/64,1/16,1/4,1,2,4]，即LR[3.125e-6,1.25e-5,5e-5,2e-4,4e-4,8e-4]。CLUENER dev种子306100..306102，WikiSQL307100..307102。双方方法族各2个rank，拥有相同候选数与dev种子数。总144个dev-only训练，训练和独立审计不读取test。

每个固定rank单独按三个种子的dev primary均值选LR，平局优先倍率距1近，再较小倍率。四配置都报告，不在test上挑rank获胜者。全部dev审计完成后写SELECTION_FROZEN，再启动确认。

确认种子CLUENER406100..406104，WikiSQL407100..407104，每架构5次共40次，另同seed H-only10次。10个新smoke覆盖全部架构/任务。总204次，其中194正式训练。确认test均为已使用过的原test，新seed仅检验训练随机性，不是新数据集证据。

主比较为各任务aLoRA r16/r30分别对普通r1，共4项一族，配对5seed t区间及Holm；aLoRA对H另4项一族。普通r2为容量/优化诊断，完整公布但不替代主基线。记录各候选dev均值及逐seed分数，比较低LR边界和选参稳定性。任何追加训练时长/H LR/bias消融均需新协议，不能按本轮test反复改候选或停止负向配置。

## 审计与存储
每次检查零残差、目标mask、冻结基座、trainable参数白名单与计数、shared实际调用、同seed H初始化、训练顺序、保存重载、预测token、独立评分和官方SQL执行。保留初始及最终adapter。产物盘至少留40GiB，项目盘至少留5GiB；故障阻断对应架构，不能按效果差取消。
