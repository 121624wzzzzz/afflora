# 小尺寸 tied 模型完整配列（2026-09-23）

用户授权扩充全部本地小尺寸tied模型。实际config已核对，范围为Qwen2.5-0.5B/1.5B/3B、Qwen3-0.6B/1.7B/4B、Llama3.2-1B/3B，8个pretrained base。无Gemma、Qwen3.5或untied强制绑定。每个模型覆盖CLUENER与WikiSQL。

## 固定共同LR架构块

H保持rank8/alpha16/dropout0.05及同一target modules。共同LR2e-4，训练2048例/64步，batch32；评测、EOS/BOS、loss mask、官方SQL复核沿用已审计实现。

- H-only、H+共享aLoRA r8/r16/r32、H+共享普通词表LoRA r1：各5种子。
- H+共享普通LoRA r2、H+共享aLoRA r16去bias：各3种子探索性消融。
- 共同LR种子CLUENER6100..6104、WikiSQL7100..7104，3种子取前三个。
- 真正共享一个有效词表更新，输出用转置；输入/输出基座实际weight指针、两侧调用、独立参数计数逐次检查。
- 实际边界预算：普通共享LoRA r(V+d)，共享aLoRA (2r+1)d，去bias为2rd。H不改，不把整数rank不同的预算称为严格等参。

## 公平搜索与确认

主比较为共享普通LoRA r1与共享aLoRA r16。双方各倍率[1/16,1/4,1,4]×2个dev-only种子，H LR固定2e-4，仅边界LR变化。CLUENER106100/106101、WikiSQL107100/107101。按dev primary平均选倍率，平局优先距1更近，再较小倍率；不依据test选配置。

确认种子206100..206104/207100..207104，每模型任务双方各5次及同种子H-only5次。选择先冻结后评test。Qwen3-0.6B/Qwen2.5-1.5B已有完全一致的四候选搜索和确认，核验后复用，不能记作新的独立重复；Qwen3-4B旧三候选/旧seed协议不替代本轮四候选确认。

主统计：全部16模型任务的aLoRA−普通LoRA、aLoRA−H各为独立预设比较族；五种子配对差、95% t区间与族内Holm校正。共同LR与调参确认分开。只看均值领先不足以宣称稳定优势，非显著不等于等效。任务已用过测试集，本轮是模型配列扩展，不是完全未接触过的任务确认。

## 复用、存储和运行

逐一核对模型文件SHA、真实tied config、原生tokenizer重编码、数据split/ID无交叉。复用训练需匹配参数量、seed、LR、训练时长/样本顺序、基座冻结、保存重载、最终adapter SHA、输出token和逐条评分；WikiSQL另用官方执行器复核。技术smoke只继承同架构、同输入协议且证据完整的组；新模型所有架构先短测再正式训练。旧初始张量缺失的情况不宣称已重新校验那些文件，保留其原审计和初始化hash。

新增checkpoint和预测文件放在/home/wz/experiment_artifacts/tied_model_expansion_20260923，通过本study内checkpoints/evaluations符号链接访问。保留初始及最终adapter，不删旧模型或实验数据。产物盘低于40GiB、项目盘低于5GiB时停止派发新任务。清理文件仅限经过逐份哈希确认的重复内容，独立保存清理记录。

精确新增/复用数量、磁盘上界见PLAN_COUNTS.json。每次完成独立审计后更新RESULTS_ZH.md及配对统计；失败阻断对应架构，不因测试效果差而取消配置。
