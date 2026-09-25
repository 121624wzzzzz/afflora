# Output-only equal-budget 独立审计

审计日期：2026-07-28

## 结论

未发现会使当前 held-out CE 结论失效的实现 bug 或明显不公平点。
当前证据支持以下限定结论：

> 在 Qwen2.5-1.5B Base、corrected SFT、共同 hidden-LoRA r8、输出侧近等
> 训练参数预算、dev CE 选参和 BF16 联合训练这一精确定义的协议下，
> direct output Vocab-LoRA 的 held-out CE 稳定优于 output-only A-LoRA。

这个结果不能外推为 Vocab-LoRA 在所有模型、任务或优化方式下都优于
A-LoRA，也不能单凭 IFEval 结果断言任一方法的全局最优行为能力。

实验协议见 [DESIGN.md](DESIGN.md)，dev 选参记录见
[final_selection.json](final_selection.json)，独立 seed 确认见
[confirmation_summary.md](confirmation_summary.md)。

## 关键证据

### 1. 实际 forward 路径和输出侧参数

Output-only A-LoRA 的实际前向为：

```text
logits = W [h + 16 · U(Dh)]
D: [50, 1536]
U: [1536, 50]
```

- 仅包裹 `lm_head`，不改变 input embedding 路径。
- `D` 使用 Kaiming 初始化，`U` 初始化为零，因此初始 residual 严格为零。
- 无 boundary bias，boundary dropout 为零。
- 输出侧原始训练参数数为 `2 × 1536 × 50 = 153,600`。

实现依据见
[src/affine_vocab_lora/adapter.py](../../src/affine_vocab_lora/adapter.py)。

Direct output Vocab-LoRA 的实际前向为：

```text
logits = Wh + 32 · B(Ah)
A: [1, 1536]
B: [151936, 1]
```

- PEFT target 中存在 `lm_head`，不存在 `embed_tokens`。
- checkpoint 中存在一组 `lm_head.lora_A/lora_B`，不存在
  `embed_tokens.lora_*`。
- `A` 使用 PEFT 默认 Kaiming 初始化，`B` 初始化为零，因此初始
  residual 同样严格为零。
- 无 bias；训练时 wrapper 将 `lm_head` 的 LoRA dropout 替换为
  `Identity`。
- 输出侧原始训练参数数为 `1536 + 151936 = 153,472`。

实现和断言依据见
[train_corrected_sft_output_vocab.py](../train_corrected_sft_output_vocab.py)。

两种方法输出侧仅差 128 个训练参数，即 A-LoRA 多 `0.0834%`。加上相同的
hidden-LoRA 后，总训练参数分别为：

| 方法 | hidden-LoRA | 输出侧 | 总训练参数 |
|---|---:|---:|---:|
| output-only A-LoRA r50 | 9,232,384 | 153,600 | 9,385,984 |
| direct output Vocab-LoRA r1 | 9,232,384 | 153,472 | 9,385,856 |

Smoke run 的初始 loss 两边均为 `2.246`，验证了两种 boundary adapter 都从
相同的零函数开始；对应日志为
[A-LoRA smoke log](logs/smoke_aff_output_r50.log) 和
[Vocab-LoRA smoke log](logs/smoke_vocab_output_r1.log)。

### 2. 初始化、训练和 optimizer 配对

- seed 42--45 中，每一对 A-LoRA/Vocab-LoRA 的共同 hidden-LoRA
  初始化 SHA256 均完全一致。
- seed 42 的全部 9 对 dev 搜索候选也逐配置使用相同 hidden-LoRA
  初始化。
- Vocab-LoRA 的额外 RNG 隔离只作用于 output boundary 的特定形状；
  hidden-LoRA 初始化不被额外 `lm_head` target 消耗的随机数扰动。
- 两边共同 hidden-LoRA 均为 r8、alpha16、dropout0.05，覆盖全部 28 层的
  `q/k/v/o/up/down/gate`。
- 最终入选配置的 hidden 和 boundary LR 均为 `2e-4`，使用同一 cosine
  scheduler、warmup ratio `0.03`、global max grad norm `1`、一轮
  `1,424` optimizer steps。
- 所有可训练参数及 Adam 一、二阶状态均为 FP32；最终模型两边均为
  394 个可训练张量。

初始化隔离代码见
[train_corrected_sft.py](../../corrected_sft_experiment/train_corrected_sft.py)
和
[train_corrected_sft_isolated_vocab.py](../train_corrected_sft_isolated_vocab.py)。
训练器及 optimizer 分组逻辑见
[train_affine_vocab_lora.py](../../scripts/train_affine_vocab_lora.py)。

### 3. dev 超参搜索预算对称

两种方法都完成了完全相同的 9 个候选点：

```text
(scale, boundary-LR multiplier)
(1,1), (2,1), (4,1),
(8,0.5), (8,1), (8,2),
(16,1), (16,2), (32,1)
```

- A-LoRA 由 dev CE 选择 `scale=16, boundary-LR=1`。
- Vocab-LoRA 由 dev CE 选择 `scale=32, boundary-LR=1`。
- test 和 IFEval 都未参与选择。
- seed42 test report 的生成时间晚于 final selection。
- A-LoRA 的 scale32 已比 scale16 变差，最优点得到括定。
- Vocab-LoRA 到 scale32 仍在改善；固定停止规则对 Vocab-LoRA 更保守，
  不会人为制造其 CE 优势。

完整候选及停止规则见
[final_selection.md](final_selection.md)。

### 4. 数据、held-out 报告和 checkpoint

- train/dev/test 分别为 `22,780 / 1,000 / 1,000` 条，三者 record ID
  两两无交集。
- dev/test 的逐样本 record ID、source 信息和 supervised-token 数在所有
  方法、seed 间完全配对。
- 使用当前 tokenizer 和 corrected data pipeline 重新 tokenization 后，
  dev/test 的逐样本 token 数与报告逐条一致，总数分别为
  `267,252 / 266,223`。
- 16 份 dev/test 报告的 `avg_ce` 均可由逐样本 `nll_sum/token_count`
  精确重算。
- 8 个最终 checkpoint 的方法、seed、rank、alpha、target、参数形状、
  trainable count 和输出路径均通过 fail-closed checkpoint validator。
- 当前 corrected data pipeline SHA256 在所有 run 中一致，loss mask 均为
  `assistant_content_plus_im_end`。

评测加载顺序为“base → A-LoRA（如有）→ PEFT hidden/output LoRA”，与训练
拓扑一致；held-out CE 实现见
[evaluate_corrected_sft.py](../../corrected_sft_experiment/evaluate_corrected_sft.py)。

独立 seed 43--45 的 test 结果为：

| Seed | A-LoRA CE | Vocab-LoRA CE | A − V |
|---:|---:|---:|---:|
| 43 | 1.122098913 | 1.019337400 | +0.102761513 |
| 44 | 1.122133085 | 1.017254804 | +0.104878281 |
| 45 | 1.122227789 | 1.019543274 | +0.102684515 |

seed-level 平均差为 `+0.103441437`，paired-t 95% CI 为
`[+0.100348837, +0.106534036]`；正值表示 Vocab-LoRA CE 更低。

### 5. IFEval checkpoint 和生成协议

- IFEval 只使用冻结后的 dev-selected 配置，不再进行选择。
- 每个 seed 都明确映射到 A-LoRA s16 和 Vocab-LoRA s32 checkpoint。
- 使用相同的 541 个 Google IFEval prompt、native chat template、左 padding、
  greedy decoding、batch size 8、`max_new_tokens=512`。
- 每种模型均按完全相同的八段边界分片。
- 已生成的 response 与 official strict/loose score artifact 均按 canonical
  prompt、checkpoint 路径、seed、row order 和 scorer 输出逐行校验。

生成实现见 [evaluate_ifeval.py](../evaluate_ifeval.py)，严格 artifact 校验见
[validate_output_only_ifeval.py](../validate_output_only_ifeval.py)。

## 风险边界

### 中等：这不是固定 hidden adapter 的纯边界表达能力实验

两边只配对了 hidden-LoRA 的初始化，之后 hidden 和 boundary 是联合训练的。
所有最终 run 的日志采样点中，global grad norm 都高于 `1`，因此全局梯度裁剪
始终活跃。不同 boundary 参数化会通过总梯度范数改变共同 hidden-LoRA 的实际
更新。

这不破坏“相同端到端训练协议”的比较，但若要把差异完全归因于输出层函数类，
应补一个固定或冻结同一 hidden-LoRA 的 boundary-only 实验，或补一个不触发
global clipping 的诊断。

### 中等：IFEval 的 estimand 是“CE 选参后的行为迁移”

超参依据 dev CE 而不是 IFEval 选择。因此：

- 若 A-LoRA 的 IFEval 更好，可以表述为它在 CE-selected 配置下表现出更好的
  约束遵循迁移；
- 不能据此断言 A-LoRA 的 IFEval 全局最优配置一定优于 Vocab-LoRA；
- 同样不能使用 IFEval 反向修改当前配置后，再把结果当作 untouched
  confirmation。

### 中低：混合精度 forward 的量化位置不同

A-LoRA 在 hidden 空间形成 FP32 residual 后，会在进入 BF16 base `lm_head`
前转换回 hidden dtype；direct Vocab-LoRA 在 FP32 vocab 空间形成 residual，
与 base logits 相加后再转换回输出 dtype。

这是两种实际实现路径的一部分，并非 checkpoint 加载错误，但会限制“纯精确
线性代数表达能力”解释。若需要排除该因素，可补一个小规模 FP32/head-only
diagnostic。

### 低：等训练参数预算不等于等 checkpoint 存储预算

由于 Qwen2.5 的 input embedding 与 `lm_head` tied，PEFT 自动在 Vocab-LoRA
adapter 中额外保存了一份冻结的 `lm_head.base_layer.weight`：

- Vocab-LoRA `adapter_model.safetensors` 约 481 MB；
- A-LoRA hidden adapter 与 affine adapter 合计约 37 MB。

这份冻结权重的 SHA256 与原始 base embedding 完全一致，不在 optimizer 中，
加载后权重绑定仍保持一致，因此不影响当前 forward 或训练参数公平性。但论文
或报告中应使用“等训练参数预算”，不能写成“等 checkpoint 存储预算”。
PEFT 的 `ensure_weight_tying=false` 警告也意味着未来 merge/export 应单独验证。

### 低：Vocab boundary dropout 的训练时 override 未写回 PEFT config

训练时 `lm_head` LoRA dropout 已由 wrapper 替换为 `Identity`，但保存的
`adapter_config.json` 仍记录全局 hidden-LoRA dropout `0.05`。当前所有
held-out/IFEval 评测都显式进入 `eval()`，dropout 不生效，结论不受影响。
若未来恢复训练，必须仍通过 output-only wrapper，否则 boundary dropout 会
错误恢复为 `0.05`。

### 低：训练器内置 final-eval monkeypatch 实际未生效

corrected trainer 修改的是模块全局 `Trainer`，但训练实际实例化的是此前已定义
的 `AffineLearningRateTrainer`，所以承诺的训练结束后 `final_eval` 文件没有
生成。

本实验的 dev/test 均由训练完成后的独立 evaluator 生成，选参读取的是经过严格
校验的 `dev_report.json`，因此该辅助代码问题不影响本实验结果。

### 低：实现 provenance 仍可加强

run metadata 固化了 corrected data-pipeline SHA256，但没有同时固化 trainer、
A-LoRA adapter 和 output-only wrapper 的代码 SHA256，也没有保存逐 batch
record-ID trace。现有文件时间、初始化哈希、日志断言、checkpoint 结构和逐样本
报告相互一致，未显示污染；正式归档时仍建议补齐上述 provenance。

### 低：原始参数近等不代表函数空间自由度严格相等

A-LoRA r50 的因子分解存在约 `r²` 的 gauge redundancy；direct Vocab-LoRA r1
也有一维尺度冗余。两边按标准 LoRA 约定匹配的是原始可训练参数数，而不是严格
相等的函数空间自由度。当前原始预算差仅 128，且 A-LoRA 的原始参数略多，因此
这不会解释 Vocab-LoRA 的大幅 CE 优势，但报告中应明确预算定义。

## 最终判定

当前可安全报告：

1. output-only equal-trainable-budget 对照没有 input/output 混入、bias、dropout、
   checkpoint 错配或 hidden 初始化不配对等致命问题；
2. direct Vocab-LoRA 的约 `0.103` held-out CE 优势在独立 seed 43--45 上稳定，
   不是单 seed 或 test 选参造成的；
3. 该优势更可能来自 direct token-space residual 的结构能力或优化路径，而不是
   A-LoRA 代码没有参与 forward；
4. IFEval 应作为 CE-selected 配置的跨指标迁移证据单独解释，不能与 held-out CE
   结论混为同一 estimand。
