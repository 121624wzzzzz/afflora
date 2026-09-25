# 视觉 token 与架构差异排查

2026-09-20。只读检查现有输入、输出和配置，未运行新训练，未改动冻结协议。

四个Qwen3.5尺寸各检查7,924条缓存样本、3,124,241个prompt/target token，五类视觉特殊token（248053至248057）出现次数均为0。各型号config的图像/视频/视觉边界ID一致；2B tokenizer还定义248055为vision_pad。六组2B WikiSQL高LR运行（3方法×2开发种子）的6,144条原始生成token序列也均未出现这五类token。具体文件哈希和计数见VISUAL_TOKEN_INPUT_CHECK.json、VISUAL_TOKEN_OUTPUT_CHECK.json。

训练器加载Qwen3_5ForCausalLM的text_config，只使用文本骨干、embedding及lm_head；视觉编码器/MTP权重不进入所用模型。CHECKPOINT_RUNTIME_AUDIT记录320个2B文本张量核验通过，技术探针中的原生VLM纯文本路径与独立文本路径logits最大差为0。这些检查不支持“误输入图像token”或“生成视觉特殊token导致低分”的解释。

这不等于排除所有多模态影响：联合预训练可能改变文本表示；248,320词表行仍参与完整softmax，特殊token即使没有被选中也可有概率质量。本次未测这部分质量，不能给出零影响结论；扩大的词表也不能全部归为视觉token。

更直接的混杂与候选解释：

- 旧WikiSQL/ANLI五种子协议使用共同LR2e-4；本轮搜索至8e-4。旧设置稳定与新设置高LR失稳不是相同条件下的代际比较。
- 本地配置显示Qwen3-1.7B为28层标准Qwen3文本骨干；Qwen3.5-2B为24层，其中18层Gated DeltaNet、6层完整门控注意力。两者hidden_size均2048，但计算结构不同。沿用同一边界缩放、同一LR与扩容规则不保证相同优化行为。
- 2B严重失稳出现在8e-4第二个开发种子；4e-4相同初始化和顺序的同方法得到81.4453%，2e-4两个种子均值80.5176%。现有证据首先支持配置/种子敏感性，没有隔离视觉预训练、词表、混合骨干各自的因果贡献。

更有辨别力的后续实验是先在共同训练设置下比较Qwen3与Qwen3.5，并在3.5内分别控制边界学习率、E/U位置和hidden扩容位置。若专门测试视觉特殊token的softmax影响，应先测概率质量；直接删词表会改变模型及评分，不能作为无需说明的修复。本次没有执行这些新实验。

官方来源：https://huggingface.co/Qwen/Qwen3.5-2B-Base 、https://huggingface.co/Qwen/Qwen3-1.7B-Base 。本地模型配置与冻结训练器为当前实验实际设置依据。
