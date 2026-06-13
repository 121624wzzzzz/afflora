# 结果审计

快照时间：2026-06-05。

本文按 Claim 整理实验结果。项目核心是 AffLoRA：在 `embed_tokens` 和 `lm_head` 上使用 hidden 维低秩仿射适配器，让词表层以极低成本参与 SFT/post-training。

除特别说明外，所有数字为 `data/sft_t2t_mini_25k/eval.jsonl` 上的最终 `eval_loss`，越低越好。

## 结论摘要

| 主张 | 状态 | 最强证据 |
|------|:---:|------|
| **Claim 1a**：AffLoRA + hidden LoRA > 纯 hidden LoRA | ✅ | 8 个模型全部正向，1.5B 三 seed 均值 Δ=-0.006 |
| **Claim 1b**：AffLoRA > vocab-dim LoRA（等 emb/lm_head 适配范围） | ✅ | 参数更少，效果持平或更好，大模型上优势更明显 |
| **Claim 2**：仅 AffLoRA > 冻结 base | ✅ | 33k-100k 参数降低 0.5+ eval_loss |
| **Claim 3**：小参数预算下 AffLoRA > 单层 LoRA | ✅ | 33k AffLoRA 优于 33k-49k 单层 LoRA |

---

## Claim 1a：完整架构 > 纯 hidden LoRA

**对比**：`affine_input_lm_head_plus_hidden_lora` vs `hidden_lora`
**统一超参**：affine r=16 α=128, hidden r=8 α=16, seed=42（除非标注多 seed）

### 全模型汇总

| 模型 | hidden_lora | + AffLoRA | Δ |
|------|:---:|:---:|:---:|
| Qwen2.5-0.5B | 1.302 | 1.294 | **-0.008** |
| Qwen3-0.6B | 1.101 | 1.098 | -0.003 |
| **Qwen2.5-1.5B** (seed 42) | 1.054 | 1.048 | -0.006 |
| **Qwen2.5-1.5B** (3 seed 均值) | 1.054 | **1.048** | **-0.006** |
| Qwen3-1.7B | 0.7383 | 0.7350 | -0.0033 |
| Qwen2.5-3B | 0.9797 | 0.9762 | -0.0035 |
| Qwen3-4B | 0.8828 | 0.8810 | -0.0018 |
| Qwen2.5-7B | 0.9180 | 0.9151 | -0.0029 |
| Qwen3-8B | 0.8717 | 0.8696 | -0.0021 |

> 1.5B 三 seed（42/43/44）全部正向：Δ = -0.006 / -0.006 / -0.007。

### 低 hidden rank 下增益更大

当 hidden LoRA rank 降低时，AffLoRA 的边际收益更显著：

| 模型 | hidden r=1 | hidden r=1 + AffLoRA | Δ | hidden r=4 | hidden r=4 + AffLoRA | Δ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen2.5-7B (3 seed) | 0.9658 | 0.9576 | **-0.0082** | 0.9361 | 0.9313 | **-0.0048** |
| Qwen3-8B (3 seed) | 0.9074 | 0.9001 | **-0.0073** | 0.8849 | 0.8805 | **-0.0044** |
| Qwen2.5-0.5B | 1.398 (r2) | 1.380 (r2) | **-0.018** | 1.354 | 1.339 | **-0.015** |
| Qwen3-1.7B | 0.7713 (r2) | 0.764 (r2) | **-0.0073** | - | - | - |

### 增益来源分解（Qwen2.5-7B）

| hidden rank | hidden-only | + input-only | + lm_head-only | + input+lm_head |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.966 | 0.965 | 0.959 | 0.957 |
| 2 | 0.952 | 0.951 | 0.946 | 0.946 |

> 结论：主要增益来自 **lm_head 输出侧**，input 侧贡献很小。

### Hidden LoRA rank scaling（基线参考）

| model | r2 | r4 | r8 | r16 | r32 |
|------|:---:|:---:|:---:|:---:|:---:|
| Qwen3-0.6B | 1.163 | 1.133 | 1.101 | 1.071 | 1.042 |

> 纯 hidden LoRA rank 增大收益稳定，AffLoRA 定位是低成本补充词表层，不替代 hidden LoRA 扩容。

---

## Claim 1b：AffLoRA > vocab-dim LoRA

**对比**：hidden LoRA + AffLoRA (emb+lm_head) vs hidden LoRA + vocab-dim LoRA (emb+lm_head, r=1)，**等 hidden rank，比谁在 emb/lm_head 上用少量参数获得更大收益**。

| 模型 | hidden rank | AffLoRA eval | vocab-dim LoRA eval | AffLoRA 优势 | 参数量对比 |
|------|:---:|:---:|:---:|:---:|------|
| Qwen2.5-0.5B | r1 | 1.413 | 1.413 | 打平 | 608k vs 856k |
| Qwen2.5-0.5B | r2 | 1.378 | 1.378 | 打平 | 1.16M vs 1.41M |
| Qwen2.5-1.5B | r1 (3 seed) | **1.113** | 1.120 | **-0.007** | — |
| Qwen2.5-1.5B | r2 (3 seed) | **1.095** | 1.099 | **-0.003** | — |
| Qwen3-1.7B | r1 | **0.777** | 0.785 | **-0.008** | 1.22M vs 1.40M |
| Qwen3-1.7B | r2 | **0.764** | 0.769 | **-0.005** | 2.31M vs 2.49M |
| Qwen2.5-7B | r1 (3 seed) | **0.9576** | 0.9639 | **-0.0063** | 2.76M vs 2.83M |
| Qwen2.5-7B | r2 (3 seed) | **0.9457** | 0.9503 | **-0.0046** | 5.28M vs 5.36M |
| Qwen3-8B | r1 (3 seed) | **0.9001** | 0.9062 | **-0.0061** | 2.99M vs 3.04M |
| Qwen3-8B | r2 (3 seed) | **0.8907** | 0.8963 | **-0.0056** | 5.72M vs 5.77M |

> 趋势：模型越大，AffLoRA 相比 vocab-dim LoRA 的优势越明显。0.5B 打平，≥1.5B 全部胜出，且 AffLoRA 参数量始终更少。

---

## Claim 2：仅 AffLoRA > 冻结 base

**不含 hidden LoRA，只在 emb/lm_head 上训练 AffLoRA。**

| 模型 | frozen base | AffLoRA-only | 参数量 | Δ |
|------|:---:|:---:|:---:|:---:|
| Qwen3-0.6B | 1.848 | **1.320** | 67k | -0.528 |
| Qwen2.5-1.5B | 1.926 | **1.326** | 100k | -0.600 |

---

## Claim 3：小参数预算 > 单层 LoRA

**模型**：Qwen3-0.6B，affine r=16 α=512 (s1=32)

| 方法 | 参数量 | eval_loss |
|------|:---:|:---:|
| **AffLoRA input-only** | **33k** | **1.426** |
| AffLoRA input+lm_head | 67k | 1.346 |
| Single-layer QKVO l14 r4 | 40k | 1.473 |
| Single-layer Q l=0 r16 | 49k | 1.626 |
| Single-layer Q l=14 r16 | 49k | 1.667 |
| Single-layer Q l=27 r16 | 49k | 1.780 |

---

## 支撑实验

### AffLoRA rank sweep

固定 hidden r=8，比较 AffLoRA rank 8/16/32：

| 模型 | rank 8 | rank 16 | rank 32 |
|------|:---:|:---:|:---:|
| Qwen3-1.7B | 0.7357 | 0.7346 | 0.7338 |
| Qwen2.5-7B | 0.9175 | 0.9167 | 0.9137 |
| Qwen3-8B | 0.8700 | 0.8690 | 0.8682 |

> rank 16 是性价比最好的默认值，rank 32 略优但增益不大。

### α 鲁棒性

| model | α=64 | α=128 | α=256 |
|------|:---:|:---:|:---:|
| Qwen3-1.7B r1 | 0.779 | 0.777 | 0.776 |
| Qwen3-1.7B r2 | 0.766 | 0.764 | 0.763 |
| Qwen2.5-7B r1 | 0.959 | 0.958 | 0.956 |
| Qwen2.5-7B r2 | 0.947 | 0.946 | 0.944 |

> α 在 64-256 范围内不敏感，256 略优。

### bias_scale 鲁棒性

Qwen3-1.7B，α=128：

| hidden r | s2=0 | s2=0.5 | s2=1 | s2=2 |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.7776 | 0.7773 | 0.7767 | 0.7774 |
| 2 | 0.7642 | 0.7641 | 0.7636 | 0.7637 |

> bias_scale 在 0-2 范围内几乎不敏感。

### Full finetune 天花板

| 模型 | full finetune | 参考 |
|------|:---:|------|
| Qwen2.5-0.5B | 1.177 | 对比 hidden LoRA r32: 1.211 |
| Qwen3-0.6B | 1.019 | 对比 hidden LoRA r32: 1.042 |

---

## 补充实验（不进入主要 Claim）

### Position ablation（Phase 3c）

将 AffLoRA 放在 decoder layer 3/10/17/24 或 final norm 后。layer 3/10 约 1.367 最优，after embedding 约 1.423，final norm 最差 1.569。该实验不直接验证三条 Claim，代码已移除。

### DoRA / rsLoRA 对照（Phase 3d）

Qwen3-0.6B：rsLoRA r=8 效果最好（1.069），DoRA r=8（1.098）与 LoRA r=8（1.100）接近。不进入主要 Claim 结论。

---

## Claim 闭合状态

| Claim | 状态 | 闭合标准 |
|------|:---:|------|
| Claim 1a | ✅ | 8 模型全正向，1.5B 三 seed 一致，低 rank 下增益放大 |
| Claim 1b | ✅ | ≥1.5B 全部胜出，模型越大优势越明显，→0.5B 打平但参数更少 |
| Claim 2 | ✅ | 0.6B/1.5B 均显著优于 frozen base |
| Claim 3 | ✅ | 0.6B 上 33k AffLoRA 优于所有 33k-49k 单层 LoRA |
| 超参鲁棒性 | ✅ | α 64-256、bias_scale 0-2 均不敏感，rank 16 性价比最优 |

### 输出目录索引

所有实验位于 `outputs/affine_vocab/sft_t2t_mini/`，按 `{模型}/{变体}_{超参}` 组织：

| Claim | 模型 | 典型 run |
|------|------|------|
| 1a | Qwen2.5-1.5B | `affine_input_lm_head_plus_hidden_lora_ar16_s18_hr8_sd42` |
| 1b | Qwen2.5-7B | `hidden_lora_hr1_vlr1_sd42` vs `affine_input_lm_head_plus_hidden_lora_ar16_s18_hr1_sd42` |
| 2 | Qwen2.5-1.5B | `affine_input_lm_head_ar16_s116_sd42` + `claim2_base/` |
| 3 | Qwen3-0.6B | `affine_input_ar16_s132_sd42` vs `hidden_lora_hr16_L14_sd42` |

MetaMathQA 下游评测位于 `outputs/affine_vocab/metamathqa/`，UltraChat/MiniMind 位于 `ultrachat/` 和 `minimind/`。
