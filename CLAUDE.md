# CLAUDE.md

本文件为 Claude Code（claude.ai/code）提供项目指导。

## 项目：AffLoRA

本项目测试 **AffLoRA**——在词表大小的层（`embed_tokens` 和 `lm_head`）上应用低秩仿射适配器，这些层在标准 LoRA 的 SFT/post-training 中通常被跳过。核心洞察（来自 `task6_base_instruct_full_vocab` 分析）：这些层中 base→instruct 的变化满足隐维度的仿射关系：`W' = W (I + s1·A·B) + s2·b`。

主要任务是 `sft_t2t_mini_25k`（24k 训练 + 1k 验证），在 Qwen2.5/Qwen3 base 模型上测试。验证三条主张：(1a) AffLoRA + hidden LoRA 优于纯 hidden LoRA；(1b) 在相同参数量下，AffLoRA 优于 emb/lm_head 上的 vocab-dim LoRA；(2) 仅 AffLoRA 优于冻结 base；(3) 在相同参数预算下，AffLoRA 优于单层 LoRA。

## 环境配置

```bash
source /home/wz/projects/mypro/im_exp/set
cd /home/wz/projects/mypro/im_exp/lora
export PYTHONPATH=/home/wz/projects/mypro/im_exp/lora/src:${PYTHONPATH:-}
```

Qwen base 模型位于 `/home/wz/projects/mypro/im_exp/models/`。输出目录为 `outputs/affine_vocab/`，按 `{任务}/{模型}/{变体}_{超参}` 组织。

## 常用命令

**冒烟测试**（2 步，Qwen3-0.6B，验证工具链和精度）：
```bash
bash scripts/sh/run_affine_vocab_smoke.sh
```

**语法检查**（验证所有 Python 脚本可解析）：
```bash
bash scripts/sh/run_syntax_check.sh
```

**主训练命令**（单 GPU，推荐超参）：
```bash
python scripts/train_affine_vocab_lora.py \
  --model-path /home/wz/projects/mypro/im_exp/models/Qwen2.5-1.5B-Base \
  --train-data data/sft_t2t_mini_25k/train.jsonl \
  --eval-data data/sft_t2t_mini_25k/eval.jsonl --eval-samples 1000 --eval-steps 250 \
  --output-dir outputs/affine_vocab/<run_name> \
  --variant affine_input_lm_head_plus_hidden_lora \
  --hidden-lora-rank 8 --hidden-lora-alpha 16 \
  --affine-rank 16 --affine-alpha 128 \
  --max-seq-len 1024 --per-device-train-batch-size 8 --gradient-accumulation-steps 2 \
  --learning-rate 2e-4 --lr-scheduler-type cosine --warmup-ratio 0.03 \
  --num-train-epochs 1 --save-strategy no \
  --master-dtype fp32 --base-dtype auto --bf16 --seed 42
```

**冻结 base 评估**（Claim 2 的零训练参考基线）：
```bash
python scripts/eval_base_loss.py \
  --model-path /home/wz/projects/mypro/im_exp/models/Qwen2.5-1.5B-Base \
  --eval-data data/sft_t2t_mini_25k/eval.jsonl \
  --report-file outputs/affine_vocab/sft_t2t_mini/claim2_base/qwen25_1_5b.json
```

## 架构

### 核心库（`src/affine_vocab_lora/adapter.py`）

整个 AffLoRA 机制集中在一个文件中：

- **`AffineVocabConfig`** — 包含所有超参的数据类：`rank`、`alpha`（LoRA 风格的缩放系数 = alpha/rank）、`bias_scale`、`use_input`/`use_lm_head` 开关、用于 tied-embedding 模型的 `tie_input_lm_head_adapters`。

- **`LowRankAffineMap`** — 核心模块。逐行仿射变换 `x → x + s1·up(down(x)) + s2·bias`，其中 down/up 是通过秩瓶颈的线性投影。`down` 用 Kaiming 均匀初始化，`up` 用零初始化（使映射初始等价于恒等变换）。

- **`AffineEmbedding`** — 包装冻结的 `nn.Embedding`，在查表后应用 `LowRankAffineMap`。

- **`AffineLMHead`** — 包装冻结的 `lm_head`（`nn.Linear`），在线性投影之前对隐藏状态应用 `LowRankAffineMap`。

- **`TiedTransposeAffineLMHead`** — 用于 tied 模式，与 `AffineEmbedding` 共享仿射适配器，在输出端用转置形式保证与合并后 embedding 等价。

- **`apply_affine_vocab_adapters(model, cfg)`** — 主入口。冻结整个 base 模型，然后根据配置包装 `embed_tokens` 和/或 `lm_head`。处理 tied-embedding 共享和可合并转置输出。

- **`save_affine_vocab_adapter()` / `load_affine_vocab_adapter()`** — 通过 safetensors + JSON 配置进行序列化。

### 训练脚本（`scripts/train_affine_vocab_lora.py`）

使用 HuggingFace `Trainer` + PEFT 进行单 GPU 训练。关键设计决策：

1. **精度**：可训练参数必须为 fp32，使用 `--master-dtype fp32`。base 模型可以保持 bf16（`--base-dtype auto --bf16`）。bf16 主权重 + bf16 AdamW 会在典型学习率下静默丢弃 lr=2e-4 的更新。脚本在开始时记录可训练参数 dtype，在结束时记录 Adam 状态 dtype——两者都应显示 fp32。

2. **变体**：通过 `--variant` 控制 8 种训练变体，详见下方「变体定义」章节。

3. **数据格式**：支持 JSON/JSONL 文件，包含 `conversations`（role/content 字典列表）或简单的 `question`/`answer`（或 `query`/`response` 等）字段。使用 prompt 模板包装问题；labels 将 prompt 前缀部分用 -100 掩盖。

4. **仿射适配器检查点**：`SaveAffineAdapterCallback` 为仿射适配器镜像 Trainer 的 PEFT 检查点保存，因为 Trainer 的 `save_model` 只序列化 PEFT 权重。

### 辅助脚本

- `scripts/eval_base_loss.py` — 计算冻结 base 模型的 eval loss（Claim 2 参考基线）。
- `scripts/eval_merge_equivalence.py` — 验证 tied 仿射适配器与合并后 embedding 的等价性。`--quick` 做单文本 logit 差异比对；不加此参数则在验证集上运行完整的 PPL + 生成对比。
- `data/prepare_ultrachat_100k.py` — 准备 UltraChat 100k 数据集（一次性脚本，与产出的数据一起保留）。

## 变体定义

### 9 个主变体（`--variant`）

按"是否训练 hidden LoRA"和"AffLoRA 作用位置"两个维度交叉组合：

| # | Variant | hidden LoRA | input AffLoRA | lm_head AffLoRA | emb/lm_head vocab-dim LoRA | 服务主张 |
|---|---------|:---:|:---:|:---:|:---:|---|
| 1 | `full_finetune` | — | — | — | — | 天花板参考 |
| 2 | `hidden_lora` | ✓ | | | | Claim 1a 基线 |
| 3 | `affine_input` | | ✓ | | | Claim 2 / 3 |
| 4 | `affine_lm_head` | | | ✓ | | 消融 |
| 5 | `affine_input_lm_head` | | ✓ | ✓ | | Claim 2 |
| 6 | `affine_input_plus_hidden_lora` | ✓ | ✓ | | | Claim 1a |
| 7 | `affine_lm_head_plus_hidden_lora` | ✓ | | ✓ | | Claim 1a |
| 8 | `affine_input_lm_head_plus_hidden_lora` | ✓ | ✓ | ✓ | | **Claim 1a 主处理组** |
| 9 | `hidden_lora` + `--include-emb-lmh-lora-rank N` | ✓ | | | ✓ | **Claim 1b 控制组** |

> 变体 9 通过 `--variant hidden_lora --include-emb-lmh-lora-rank N` 调用。它在 emb/lm_head 上使用词表维 LoRA（参数量 = vocab_size × N × 2），与等参数预算下的 AffLoRA（变体 8）对比，验证 hidden 维仿射是否比词表维 LoRA 更参数高效。常用 N=1。

### 子变体 flag

以下 flag 在 8 个主变体之上叠加，用于控制实验和消融：

| Flag | 作用 | 适用场景 |
|------|------|---------|
| `--tie-affine-input-lm-head-adapters` | input/lm_head 共享 AffLoRA，输出侧用转置形式保证可 merge | tied-embedding 模型，需同时启用 input + lm_head affine + lm_head 偏置 |
| `--hidden-lora-layers-to-transform "0"` | LoRA 只作用于指定 decoder 层 | Claim 3 控制组 |
| `--no-affine-input-bias` | 去掉 input 侧偏置 b | 消融 |
| `--affine-lm-head-bias` | 启用 lm_head 侧偏置 b | tied 时必须开 |

### Merge 与部署

本项目使用的 Qwen 模型均为 tied embedding（`embed_tokens.weight` 和 `lm_head.weight` 是同一块内存）。AffLoRA 有两种 adapter 放置模式：

#### 默认模式：独立适配（不可 merge）

不做特殊设置时，input 和 lm_head 各自有独立的 `LowRankAffineMap`。两个 affine 参数不同，无法合并到单一 embedding 矩阵。这是主实验使用的模式，适合纯 loss 研究。

#### Tied 模式：共享适配（可 merge ✅）

`--tie-affine-input-lm-head-adapters`

input 和 lm_head 共享同一个 `LowRankAffineMap`，输出侧用 `TiedTransposeAffineLMHead` 计算转置形式：`h·M^T·W^T + h·b^T`，与合并后 `W' = WM + b` 的 tied 输出完全等价。

等价性已通过 `eval_merge_equivalence.py --quick` 验证（max logit diff ≈ 3e-5，fp32 精度噪声级别）。

> 合并方法：对 `AffineEmbedding` 的 base weight 应用 `affine()` 得到 `W'`，直接替换 `embed_tokens.weight` 和 `lm_head.weight`。

## 关键设计规则

- **永远不要在没有 `--master-dtype fp32` 的情况下训练**。bf16 主权重在典型学习率下会静默破坏训练。
- **模型路径**：主要模型位于 `/home/wz/projects/mypro/im_exp/models/`。
- **Tied embeddings**：本项目使用的所有 Qwen 模型都具有 tied `embed_tokens`/`lm_head` 权重。`tie_input_lm_head_adapters` 启用共享仿射适配器，使其可以合并回单一 embedding 矩阵。
- **数据格式**：训练数据应包含 `conversations`（`{role, content}` 字典列表，含 "user"/"assistant" 角色）或平铺的 `question`/`answer`（或 `query`/`response`、`instruction`/`output`、`problem`/`solution` 等）。
- **仅以 eval_loss 评估**：这是纯粹的语言建模 loss 研究。当前工作流中没有基于生成的评估。

## 输出目录结构

`outputs/affine_vocab/` 按 `{任务}/{模型}/{变体}_{超参}` 组织：

```
outputs/affine_vocab/
├── sft_t2t_mini/          ← 主任务（Claim 1a/1b/2/3）
│   ├── Qwen2.5-0.5B/  Qwen3-0.6B/
│   ├── Qwen2.5-1.5B/  Qwen3-1.7B/
│   ├── Qwen2.5-3B/    Qwen3-4B/
│   ├── Qwen2.5-7B/    Qwen3-8B/
│   └── claim2_base/       ← frozen base 基线
├── metamathqa/            ← MetaMathQA + GSM8K/MATH 评测
│   ├── Qwen2.5-1.5B/
│   └── Qwen3-8B/
├── ultrachat/             ← UltraChat 100k
│   ├── Qwen2.5-1.5B/
│   └── Qwen3-1.7B/
└── minimind/              ← MiniMind random 100k
    ├── Qwen2.5-1.5B/
    └── Qwen3-1.7B/
```

命名规则：`{variant}_ar{rank}_s1{scale}_hr{hidden_rank}_sd{seed}`，例如：
`affine_input_lm_head_plus_hidden_lora_ar16_s18_hr8_sd42`

## 结果与实验追踪

- `docs/RESULTS_SO_FAR.md` — 权威结果账本，按 Claim 组织所有证据。
- `docs/AFFINE_VOCAB_MAIN_EXPERIMENT.md` — 实验设计及三条主张。
- `docs/EXPERIMENT_SETTINGS.md` — 完整路径、模型表、超参、变体描述。
- `docs/CLEANUP_MANIFEST.md` — 删除/保留内容及原因。
- `docs/archive/` — 旧 phase 结果文档存档。
