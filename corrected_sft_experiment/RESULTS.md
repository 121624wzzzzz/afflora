# 修复后多轮 SFT 实验结果

## 修复范围

本实验只修复数据与评测流程，不修改 AffLoRA 参数化、训练精度策略或 merge 实现：

- 保留原始多轮顺序；
- 使用模型原生 chat template；
- 只对 assistant 内容和 `<|im_end|>` 计算 loss；
- 丢弃不完整或角色顺序异常的对话；
- dev/test 按首个 user prompt 隔离，杜绝精确上下文重叠；
- test 只在配置确定后用于最终评测。

## 数据审计

数据清洗后保留 24,780 条合法对话，拒绝 220 条异常数据：154 条以未回答的 user 结尾，66 条角色没有严格交替。确定性划分为 22,780 train / 1,000 dev / 1,000 test。dev 和 test 的首个 user prompt 均为 1,000 个互不重复的 singleton prompt，并按 assistant 轮次数与全量数据分布进行分层。

train/dev/test 两两之间的首个 user prompt、完整 assistant context 和完整 conversation 精确重叠均为 0。完整来源 SHA256、轮次直方图和生成文件 SHA256 见 `data/manifest.json`。

## 正式实验矩阵

| 模型 | seed | baseline | treatment |
|---|---:|---|---|
| Qwen3-0.6B-Base | 42/43/44 | hidden LoRA r8 | hidden LoRA r8 + input/lm_head AffLoRA r16, s1=8 |
| Qwen2.5-1.5B-Base | 42/43/44 | hidden LoRA r8 | hidden LoRA r8 + input/lm_head AffLoRA r16, s1=8 |

统一设置：1 epoch，effective batch 16，learning rate `2e-4`，cosine schedule，warmup `0.03`，base BF16，trainable master weights FP32，`max_seq_len=1024`。

为保证配对比较，seed 在模型和适配器创建前设置；AffLoRA 使用隔离的确定性 RNG stream，随后恢复全局 RNG，因此同模型、同 seed 的 baseline/treatment 具有完全相同的 hidden-LoRA 初始化。正式日志已逐组记录并核对 `hidden_lora_init_sha256`。

## 结果

### Qwen3-0.6B，seed 42/43/44

| seed | test hidden | test + AffLoRA | test Δ CE | 95% paired bootstrap CI |
|---:|---:|---:|---:|---:|
| 42 | 1.215027 | 1.209675 | **-0.005352** | [-0.005886, -0.004823] |
| 43 | 1.214631 | 1.209771 | **-0.004860** | [-0.005359, -0.004342] |
| 44 | 1.214628 | 1.209182 | **-0.005446** | [-0.005968, -0.004933] |
| mean | 1.214762 | 1.209543 | **-0.005219** | — |

test 共 1,000 条对话、270,223 个 supervised token。三个 seed 的 Δ CE 样本标准差为 0.000315，基于三个配对 seed 的 95% t 区间为 [-0.006001, -0.004437]。每个 seed 的 10,000 次 paired bootstrap 区间均严格低于 0，treatment 更优概率均为 1.0。三个 seed 的 test PPL 分别为 3.370385→3.352396、3.369049→3.352716、3.369040→3.350742。

运行目录：`outputs/formal/qwen3_06b_{hidden,afflora}_sd{42,43,44}`。逐样本报告和 bootstrap 结果保存在相应目录及 `outputs/formal/qwen3_06b_sd{42,43,44}_test_comparison.json`。

### Qwen2.5-1.5B，seed 42/43/44

| seed | dev hidden | dev + AffLoRA | test hidden | test + AffLoRA | test Δ CE | test 95% CI |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 1.113355 | 1.105618 | 1.132984 | 1.125557 | **-0.007427** | [-0.007941, -0.006924] |
| 43 | 1.113148 | 1.105974 | 1.132497 | 1.125370 | **-0.007127** | [-0.007638, -0.006627] |
| 44 | 1.112989 | 1.105095 | 1.132312 | 1.124837 | **-0.007476** | [-0.007993, -0.006962] |
| mean | 1.113164 | 1.105563 | 1.132598 | 1.125255 | **-0.007343** | — |

三组 test delta 的样本标准差为 0.000189，每个 seed 的 10,000 次 paired bootstrap 区间都严格低于 0，treatment 更优概率均为 1.0。三个 seed 的 test PPL 分别为 3.104908→3.081934、3.103397→3.081357、3.102823→3.079714。

运行目录：`outputs/formal/qwen25_15b_{hidden,afflora}_sd{42,43,44}`；汇总见 `outputs/formal/qwen25_15b_multiseed_test_summary.json`。

## Frozen-base 参考

统一的 corrected test 上，Qwen3-0.6B frozen base CE 为 1.875968，Qwen2.5-1.5B frozen base CE 为 1.818539。该数值仅用于显示 SFT 适配收益，不参与 baseline/treatment 的 paired claim。

## 当前结论

修复多轮结构、chat template、assistant-only loss、数据格式和 split 泄漏后，`hidden LoRA + AffLoRA` 仍稳定优于相同 hidden-LoRA 初始化的 baseline。Qwen3 和 Qwen2.5 各三个 seed 的独立 test 结果方向全部一致，且所有 paired bootstrap 95% 区间均不跨 0。
