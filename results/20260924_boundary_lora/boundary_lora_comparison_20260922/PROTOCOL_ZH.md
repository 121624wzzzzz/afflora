# 边界 aLoRA 与直接词表 LoRA：2026-09-22 冻结方案

用户已授权启动全部类别。此轮按阶段执行，不依据测试集收益正负改变配置或追加种子。

## 范围

1. Qwen2.5-7B、Qwen3-8B、Llama3.1-8B（untied）：H、H+E、H+U、H+E+U。同位置比较 aLoRA r16 与直接词表 LoRA r1，五个配对训练种子。untied 禁止共享边界参数。
2. Qwen3-4B、Qwen2.5-1.5B、Qwen3-0.6B（tied）：H、共享 aLoRA r8/r16/r32、真正共享词表更新的普通 LoRA r1，五种子；原三种子实验合格才复用，补齐另外两个。共享模块只注册一份，E/U 使用同一有效矩阵及其转置。
3. 预算曲线限定在双侧或共享组：普通 LoRA r1/r2，aLoRA r8/r16/r32；r2及untied的额外预算点三种子。Qwen untied已有单侧r8保留三种子。避免所有位置与所有rank无目的笛卡尔扩张。
4. bias 消融：各含E的aLoRA r16配置增加无bias版，三种子。普通LoRA保持标准无bias；不额外增加未决定的普通LoRA+bias实验。
5. 公平调参：Qwen2.5-7B、Qwen3-8B、Qwen3-4B、Llama3.1-8B的双侧/共享主配置；aLoRA r16及普通LoRA r1各三个边界学习率倍率[0.25,1,4]×两个dev专用种子。H学习率固定2e-4。按dev主指标跨种子平均选倍率，完全相同时优先接近1，再较小倍率。选择写入SELECTION_FROZEN后，独立的五个训练种子确认，每种方法相同搜索预算。
6. 样本稳健性：Qwen3-0.6B，两任务，把现有2048训练样本按固定seed92817分成互不重叠的两个1024子集，每份H/共享aLoRA16/共享普通LoRA1各三种子。固定32步，与全数据64步结果分列。此处是训练子集敏感性，不是新任务、新测试数据或独立数据总体。

所有任务均为CLUENER与WikiSQL。两任务此前已依据结果筛选，因此本轮不能支持“未见任务普遍有效”；原有负结果仍保留。新任务未指定，不擅自把同一任务新种子称为新下游任务。Gemma/Qwen3.5不扩展。Llama3.2-3B是原建议中的条件后续项，本轮不展开。

## 固定训练与评测

H固定rank8/alpha16/dropout0.05，q/k/v/o/up/down/gate投影；边界dropout0，scale8（alpha=8r）。普通embedding LoRA：词表因子零初始化，特征因子标准正态；输出LoRA：词表因子零、特征因子Kaiming，与embedding/linear常用初始化方向对应。共享普通LoRA采用embedding初始化，U严格复用转置。记录初始化，不把不同参数化的相同LR解释为相同有效更新强度。

LR2e-4，AdamW，WD0，梯度裁剪1，batch32，2048训练样本/64步，3%warmup/cosine；训练仅target token含native EOS计loss。Qwen EOS151643；Llama BOS128000/EOS128001。dev/test生成使用fp32基座、贪婪解码和原始任务评分；WikiSQL另用官方执行器复核。Llama使用旧共同2e-4、1024问题test协议，不混入后续4e-4/2048问题test结果。

每次运行检查零更新、HF shifted loss、选择token loss、参数白名单、冻结基座hash、优化器dtype、两侧是否实际调用、独立参数存储、checkpoint重载与逐条评分。按模型seed检查H初始化一致。短训练通过独立审计后才放行同架构正式训练。

参数量均为实际可训练数量。H不变，不填充无效参数、不删H参数凑等参；整数rank不能精确对齐时明确差额，报告参数-效果曲线。

## 种子和统计

共同LR使用CLUENER6100..6104、WikiSQL7100..7104；探索性三种子为各自前三个。dev调参为起始seed+100,+101；确认seed+200..+204；子集seed+300..+302。旧合格结果复用不算新独立重复。调参后的确认仍使用既有test，仅训练随机性独立。

主共同LR比较族：三个untied模型×三个位置×两任务，加三个tied模型×共享位置×两任务，共24个 aLoRA16−普通LoRA1 配对差异；另24个 aLoRA16−H 为叠加收益族。报告全部五种子均值、配对差值、95% t区间及族内Holm校正p值；n=5时区间不稳，不只依赖显著性。调参确认16组method配置组成8个配对比较单独校正。预算曲线/bias/subset属探索性，完整公开，禁止择优隐藏。

## 执行与复用

FORMAL_JOBS/TUNING_JOBS/CONFIRMATION_TEMPLATES/SMOKE_JOBS是完整预设列表。REUSE_AUDIT核对历史checkpoint冻结/重载证据、参数量、adapter hash、test输入与逐条评分，WikiSQL官方复核；失配条件重新训练。新运行保存全部spec/checkpoint/训练序列/输出/审计。

优先完成Qwen untied普通r1直接对照。各架构短训练门控；故障阻断相应架构并保留日志，不按效果停止。GPU按空闲显存调度，不终止其他任务。剩余磁盘低于20GiB停止发新任务并等待处理，不删除历史结果。每阶段报告已审计结果，未完成种子均值注明临时。
