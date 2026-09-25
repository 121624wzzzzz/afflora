# Llama 1B / 3B / 8B：两个核心任务的尺寸扩展

研究 CLUENER 和 WikiSQL 上叠加 aLoRA 是否跨尺寸保持收益。旧结果：3B/8B CLUENER 三种子均正；WikiSQL 3B均值正、8B共同LR旧结果略负。8B后来独立调参的五种子正均值另列，不混入本轮固定LR矩阵。

|型号|本轮新增|复用（先核验再独立重评分）|
|---|---|---|
|Llama-3.2-1B Base|两任务×普通/严格等参/叠加×3种子＝18次训练，2项Base|无|
|Llama-3.2-3B Base|两任务×新严格等参×3种子＝6次训练|普通/叠加12次训练、2项Base|
|Llama-3.1-8B Base|无重复训练|三组18次训练、2项Base|

合计24次新正式训练、2项新Base、8次技术试跑；历史复用30次训练、4项Base。复用不是新的独立重复。所有原结果保留，包括旧3B多1024参数的对照。

固定共同LR2e-4、2048样本、64步；采用与旧Llama一致的数据、原生BOS/EOS和评分。此阶段检验共同方案下的尺寸外推，不是各方法独立调参后的最优比较。三个型号全部严格等参，预算分别为5,769,216 / 12,356,608 / 21,237,760个活跃可训练参数。为实现精确预算，1B增加15Q+16K+3MLP-down的rank；3B增加28Q+4K+1MLP-down；8B沿用30Q+4K。每处rank8→9，共有rank8初始化与缩放2保留。3B的对照位置也变了，不能把变化仅归因于减去1024参数。

当前状态见 [LAUNCH_STATE.json](LAUNCH_STATE.json)，训练开始后见 [study/STATE.json](study/STATE.json)。完整块结果会写入 [study/RESULTS.md](study/RESULTS.md)，最终全矩阵核验写入 `study/FINAL_AUDIT.json`。启动控制器完成下载门禁、准备和复用审计、技术试跑、正式训练与最终审计。最多4个GPU任务，不终止其他负载，磁盘余量不足60GiB时暂停新准入。

模型从ModelScope公开分发下载，逐文件与官方Meta固定提交身份核对。1B官方版本为 `4e20de362430cd3b72f300e6b0f18e50e7166e08`。完整实验设计见 [study/PROTOCOL.md](study/PROTOCOL.md)。不同版本/预训练/权重绑定限制纯尺寸因果解释，公开数据不能排除预训练暴露。

空间清理：依据用户授权删除三个已停止扩展的Gemma模型的12个原始权重文件，释放46.03GiB；保留配置、分词器、历史实验输出、adapter和审计证据。精确清单见 [GEMMA_WEIGHT_CLEANUP.json](GEMMA_WEIGHT_CLEANUP.json)。再次执行Gemma推理需要重新下载已记录身份的原始权重。
