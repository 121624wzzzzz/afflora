# CMRC 任务内 SFT：叠加收益的直接对照

本实验在中文通用 SFT 的迁移结果出现种子波动后启动。它是明确的顺序补充实验，不是未看过原任务结果的预注册。固定问题：在阅读理解任务本身训练时，双边边界适配能否在 hidden LoRA 上增加答案生成收益，并超过等参数预算对照？本轮固定为 CMRC，不按结果扩展任务或挑选种子。

使用与前轮相同、已按官方完整 SHA-256 验证的 Qwen3-0.6B post-trained 和 Qwen2.5-1.5B-Instruct。各臂均从原始聊天模型独立初始化，**不继续训练上一轮通用对话 adapters**。实验臂 none(hidden-only)、output(hidden+output)、both(hidden+input+output)、hidden_budget(等双边额外参数的 hidden rank 分配)。全部运行 seeds 42/43/44。沿用已经检查的 paired initialization 和精确额外参数预算：66,560 / 99,840。

官方 CMRC train 是唯一监督训练来源。按规范化篇章内容哈希分组，取 SHA256("20260915:"+context_hash) 最小的 240 组为 internal dev，其余为 train；原文件顺序保留。若训练篇章与最终 public dev 精确重叠，则从训练候选中剔除并记录。train/internal-dev/public-dev 三者的篇章精确重叠检查保存。最终仍评测完整官方 public dev 3,219 题；该集合已被上一轮迁移评测使用，明确承认之前的评测暴露，不称为官方隐藏 test 或全新盲测。原始数据版本、拆分计数与哈希在 DATA_AND_MODEL_AUDIT.json 中固定。

训练 user 提示与前轮 CMRC 评测逐字相同；assistant 目标为官方训练集唯一答案。使用原生 no-thinking 模板，监督 assistant 答案和原生 im_end，排除提供的空 thinking 前缀。最大长度 2,048，仅作为安全上限；训练前逐条确认没有截断。prompt_text 及评分逻辑沿用前轮固定实现。

固定训练预算：1 epoch、batch 8 × accumulation 2、LR 5e-5、cosine schedule、warmup ratio 0.03、max gradient norm 1、hidden rank 8 / alpha 16 / dropout 0.05；边界 rank 16 / alpha 128 / dropout 0，input 含 bias、output 无 bias，沿用原方法。基础权重 BF16、可训练 adapter 参数 FP32、BF16 AMP。所有臂一致，无按 public-dev 结果调参。只保存并评测最后一步，不用 internal dev 挑最佳 checkpoint。internal dev CE 用于训练检查和报告。该固定配置不代表每个方法已被单独充分调优。

正式训练前，两个模型 × 四臂各做两步烟测，验证训练参数数目、共享 hidden 初始化、输入/输出边界的初值配对、保存重载的 adapter 张量逐位一致与 plain/AMP logits 差≤1e-5。烟测生成与数值评分仅用 internal dev 首 8 题。全部通过后冻结正式训练/评分清单。旧源码只读拷贝，旧训练与迁移结果保持不变。

生成评测完全沿用前轮：greedy、max_new_tokens=256、repetition_penalty=1、原模型 EOS 加 im_end、原生 no-thinking、最多 16 条且按 padded prompt tokens≤8,192 组成批次、源顺序、不裁剪上下文、官方 CMRC v5-special 的 Python 3 兼容评分逻辑。保存原始 token/text、EM、F1、(EM+F1)/2、截断率。答案 CE/top-1 使用只含答案内容的概率，逐题先等权平均参考答案，再平均题目；FP32 连续 CE 对 FP64 在烟测核验。相同模型跨臂共用相同批次规则。辅助 CE 与原聊天 token 加权 CE 的口径不同，不混算。

主要效应有 4 个：两个模型各自 both−none、both−hidden_budget，在 (EM+F1)/2 上的百分点差。报告三个训练种子的配对均值、名义 95% t 区间；另以三个已拟合模型为条件做 10,000 次篇章簇配对 bootstrap，随机种子 20260915。另报 Bonferroni family=4 的同时区间，区分种子与题目两类不确定性。EM/F1、answer CE/top-1、output 臂及训练成本为辅助分析。不得只报告正向种子或某个对照。前轮未做 SFT 的两个参考可在完全相同推理定义及数据哈希核验后复用，明确标注来源。

结论范围限于两个模型、一个中文阅读理解任务、一种固定训练配置和三种子。正结果不能证明普遍生成改善；负结果也必须与先前 CE 和迁移结果并列保留。当前补充不验证 tying 的因果效应，也不能以缺少 input-only 的臂布局建立完整 input/output 交互机制。
