# AffLoRA 实验

**2026-09-20，Qwen3.5扩至四个尺寸，统一数值设置实验进行中：** [0.8B/2B/4B/9B运行入口](reviewer_followup/qwen35_fixed_multiscale_20260920/README.md)。每尺寸两任务、普通/严格等参/叠加三组，同等六学习率搜索、两个开发种子和五个确认种子，合计456项运行、16项主要比较。四个尺寸均已通过权重身份、适配器公式和原生计算路径核验；完整效果尚未产出。

[数值诊断](reviewer_followup/qwen35_numerics_20260920/REPORT_ZH.md)通过仅改变FLA归一化执行配置，精确复现了旧0.8B一项运行的全部16个首批微损失；旧进程未记录其实际配置，且不能据此认定最终效果涨跌的原因。固定执行配置和PyTorch数值策略后，0.8B的等参LoRA与叠加组均通过独立进程、不同GPU八步的损失/梯度/参数逐位一致检查。新矩阵采用共同设置并添加独立进程首批参考门禁；旧4B完整结果及旧0.8B/2B/9B的41项完成运行原样保留，全部重训、不拼接旧结果。下方4B分数属于旧设置的历史证据，其最大首批差异尚未直接复现。

2026-09-19：[Qwen3.5-4B Base 混合架构五种子实验](reviewer_followup/qwen35_transfer_20260919/FINAL_INTERPRETATION_ZH.md) 已完成最终审计。模型通过官方魔搭下载并对照固定官方版本逐文件校验；空间足够，未删除Gemma。72次开发训练、30次确认训练、2项Base及10项技术试跑共114项运行，86,908条输出通过审计。WikiSQL的Base/普通H/严格等参H/H+E+U为43.311/84.238/84.395/83.408%；TREC50为62.60/89.16/89.16/88.68%。叠加相对等参分别−0.986/−0.480pp，本轮没有复现额外收益。

三组均以相同六学习率、两个开发种子独立选参，再用五个新训练种子确认；等参H与叠加严格同为16,398,848参数。四项校正种子区间均跨零，不能证明普遍无效；WikiSQL固定五模型的两个校正表格bootstrap区间低于零。Qwen3.5是Qwen家族内的新混合架构，不计作新增独立家族。原生48个FP32张量保留，冻结原权重和实际共享初始化均核验；但14个阶段/任务/种子组中7组首批损失存在小数值差异，最大0.00858930，已做零更新重放但来源未定位，不宣称加速BF16训练逐位确定或已定位下降机制。完整负结果、次要指标及限制见报告。

2026-09-19：[Gemma-2-9B Base 跨模型族实验](reviewer_followup/gemma_transfer_20260918/FINAL_INTERPRETATION_ZH.md) 已完成：72次开发训练、30次五种子确认训练、2项Base及10项技术试跑，114项运行与86,908条输出通过最终审计。普通/严格等参/叠加的 WikiSQL 均值为83.350/84.072/83.809%，TREC为89.36/89.16/87.76%；Base分别19.727/27.80%。两任务叠加相对等参为−0.264/−1.400pp，四项校正种子区间均跨零；固定五个模型时，TREC两项校正数据重采样区间均低于零。本轮未新增稳定叠加优势，保留为模型/任务边界。

Gemma两对照与叠加采用相同六学习率、各两个开发种子，独立选择后五种子确认；严格等参组与叠加均为27,241,984参数。TREC最高学习率的一个开发种子发生单类别崩塌，完整保留；选中配置的确认种子没有该现象，不能据此确定下降机制。共享公开基准已被其他模型族评估。首版缓存评分技术预检失败已独立封存；正式搜索前改为原生完整前向，保持阈值并精确复现十项试跑训练轨迹。详细设置、负结果及审计范围见报告。

2026-09-18：[Llama-8B WikiSQL 公平学习率搜索与五种子验证](reviewer_followup/llama_lr_20260918/FINAL_INTERPRETATION_ZH.md) 已完成最终审计。三种方法各 6 候选×2 开发种子，独立新表格验证；全部选中 LR=4e-4，H+E+U 的 E/U 比例为 1。普通/严格等参/叠加均值为 82.813/82.617/83.271%，叠加差 +0.459/+0.654 pp，校正种子区间均跨零。相对原 LR 的调参改善为叠加 +1.543、等参 +2.139 pp，两项校正区间均为正。原 LR 在本轮新种子和验证集上已出现 +1.250 pp 的叠加均值，不能把旧负结果转正全部归于调参。69项作业、88,160条输出、两个零更新梯度探针和表格 bootstrap 完成；完整限制与跨模型尺度诊断见报告。

2026-09-18：[Llama 跨家族三种子扩展](reviewer_followup/llama_transfer_20260918/FINAL_INTERPRETATION_ZH.md) 已完成最终审计：Llama-3.2-3B / Llama-3.1-8B Base，36次新训练、4项Base与8次短程试跑，56,652条输出全部核对。CLUENER相对预算LoRA平均 +2.203 / +0.942 F1，两款模型三个种子均同时高于普通与预算对照；WikiSQL为 +0.651 / −0.293 pp，分别两正一负／两负一正。四条件中三组均值正、一组负，八项未校正及Bonferroni-8种子区间均跨零。8B严格等参；3B预算对照多1,024参数（总训练预算约0.00829%）。支持指定任务的跨家族收益方向，保留8B WikiSQL未复现额外收益的条件，不支持普遍优势。

2026-09-18：[14小时模型、任务与架构实验报告](reviewer_followup/model_architecture_14h_20260917/FINAL_INTERPRETATION_ZH.md) 已完成最终审计：377次新训练、18组Base、32次短程检查，518,836条新输出记录；另有104条核验后的历史运行复用。八个Qwen Base覆盖核心主对照16/16、九架构三种子14/16、新任务三种子9/16；125个计划运行因时间预算未启动，untied额外五种子块未齐。7B/8B两个untied型号的核心九架构三种子均完整。

本轮支持边界层的独立低参数适配能力，以及依赖模型/任务的叠加收益。核心两任务主对照均值全正，但含历史复用和任务选择；完整块28项叠加比较的本轮校正区间均跨零。新任务9个完整条件中5组均值高于两对照、3组低于两者、1组混合，全部校正区间跨零。8B WikiSQL无内部LoRA的双侧适配以266,240参数取得67.708%执行正确率，Base为46.582%，这项校正区间为正；没有新的standalone等预算内部LoRA对照，不支持普遍叠加优势或同预算最优。分侧负例、回答/拒答分解、两次排队修订和全部遗漏见报告。

2026-09-17：[Qwen2.5 Base 3B / 7B 五种子尺寸扩展](reviewer_followup/model_scaling_20260917/FINAL_INTERPRETATION_ZH.md) 已完成。CLUENER 相对预算 LoRA 的平均增益为 +1.675 / +1.247 F1，WikiSQL 为 +1.152 / +0.684 pp，四组均值都高于两个对照。仅 3B WikiSQL 相对严格等参的校正区间高于零；其余七项跨零，尚无新增同时通过两个对照的条件。7B WikiSQL 第五种子相对普通少对 1 题、与预算组持平。3B 严格等参数，7B 预算对照多 512 个参数。90,432 条新输出及 37,872 条历史锚点输出已复核；任务因此前正结果而选择，尺寸趋势不作单一机制归因。

2026-09-17：[Banking77 与清洗版 E2E 五种子扩展](reviewer_followup/downstream_nlg_intent_20260917/FINAL_INTERPRETATION_ZH.md) 已完成。四组任务/模型的叠加均值均高于普通和严格等参数 LoRA；1.5B 两任务五种子均正向，但本轮八个预定校正区间均跨零，尚未新增强确认性条件。178,208 条输出完成复核，生成词句匹配与属性错误分开解释。

2026-09-17：[ANLI R1 与 WikiSQL 五种子扩展](reviewer_followup/downstream_transfer_20260917/FINAL_INTERPRETATION_ZH.md) 已完成。WikiSQL 两个 Base 模型相对等参数 LoRA 分别提高 +1.43 / +2.21 pp，五种子均正向，本轮八项比较校正区间均高于零但下界接近零；ANLI 未发现稳定叠加收益。104,960 条输出已复核，完整设置和限制见报告。

## 当前下游任务复核入口（2026-09-16）

新增 [工具调用与实体抽取实验](reviewer_followup/posttraining_tasks_grounded_20260916/FINAL_INTERPRETATION_ZH.md) 已完成两个官方 Base、五种子及保留集评测，131,442 条开发/保留集输出通过复核。Qwen2.5-1.5B Base 的实体抽取叠加收益为 +3.39 F1，相对严格等参数 LoRA 为 +3.33，八项比较校正后的种子区间均高于零；其他三组任务/模型均值为正但区间跨零。这是固定 2,048 条训练、单一共同学习率下的任务适配证据。

SciQ 任务适配使用 [修复版协议与运行入口](reviewer_followup/sciq_repaired_protocol_20260916/README.md)。该流程固定 checkpoint、提示、推理模式和生成预算，分别报告候选准确率、生成答案准确率、格式、停止及截断，并加入参数量接近的单层内部 LoRA 对照。结果与完成状态以该目录的 `RESULTS.json`、`FINAL_AUDIT.json` 和 `COMPLETION.json` 为准。

历史纯 loss 实验继续保留，但 loss 改善不能直接替代答案质量或通用后训练能力的证据。代码中的实验臂 `base` 有时表示“未适配当前 checkpoint”，并不保证该 checkpoint 是预训练 Base；必须核对官方模型身份和训练阶段。Base 与后训练模型的原生接口、thinking 和非 thinking 结果须分别报告。

本目录用于验证 **AffLoRA**：基于 `${AFFLORA_ANALYSIS_ROOT:-$REPO_ROOT/../../get_useful/ijcai_clean/results/task6_base_instruct_full_vocab}` 中 base→instruct 全词表仿射关系的分析结果，让 SFT / post-training 中通常被跳过的 `embed_tokens` 和 `lm_head` 以很小参数量参与训练。

AffLoRA 的定位不是替代 transformer block 上的普通 LoRA，而是补上后训练里词表层难以高效训练的问题。对于 tied 模型，`lm_head` 和 embedding 权重绑定，但仍可在输入侧与输出侧 hidden state 上分别放置适配；对于非 tied LLM，则对应适配 `embed_tokens` 和独立的 `lm_head` / `llm_head`。

核心文档：

```text
docs/AFFINE_VOCAB_MAIN_EXPERIMENT.md   # 实验设计与三条主张
docs/EXPERIMENT_SETTINGS.md            # 路径、模型、超参、运行方式
docs/RESULTS_SO_FAR.md                 # 当前结果审计
docs/CLEANUP_MANIFEST.md               # 清理与归档记录
```

## 历史 loss 工作流与研究假设

原始 loss 工作流使用 `sft_t2t_mini_25k`，单卡训练，hidden LoRA r=8，AffLoRA rank=16 / alpha=128。以下保留原研究假设；它们不是当前下游任务中已普遍成立的结论，实际证据与限制以上方审计报告为准：

1. 完整架构有效：`hidden_lora + affine_emb + affine_lm_head` 优于纯 `hidden_lora`。
2. AffLoRA 模块本身有效：只训练 emb/lm_head 侧 AffLoRA 也明显优于 frozen base。
3. 参数效率更高：同等小参数预算下，AffLoRA 优于 vocab 维 LoRA 或少量层 LoRA。

旧的 math generation 评测、LoRMA 论文设置、GLUE/E2E 对照已经从原 loss 工作流移除。本轮清洗版 E2E 是上方独立冻结协议下的新实验，不复用旧 E2E 分数。历史数据和结果只作为审计/背景保留；当前任务内容评测使用上面的修复版入口。

## 目录结构

```text
data/      主任务 sft_t2t_mini_25k，以及 UltraChat / MiniMind 扩展数据
docs/      实验设计、运行设置、结果审计、清理记录
outputs/   训练和评估产物
scripts/   主训练、base eval、phase launcher
src/       affine_vocab_lora / AffLoRA 核心实现
```

## 环境

```bash
cd /path/to/im_exp/lora
export REPO_ROOT="$PWD"
export MODEL_ROOT="${MODEL_ROOT:-$REPO_ROOT/../models}"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
# Optional: source "$REPO_ROOT/../set" if your machine uses that env file
```

Qwen base 模型默认位于 `$REPO_ROOT/../models`，也可通过 `MODEL_ROOT=/path/to/models` 覆盖。

## Smoke

```bash
bash scripts/sh/run_affine_vocab_smoke.sh
```

该命令在 Qwen3-0.6B 上跑 2 step，用于验证工具链和精度打印。训练日志中应看到 trainable 参数与 Adam state 为 fp32，避免 bf16 master 权重导致小更新被舍入。

## 主训练示例

```bash
python scripts/train_affine_vocab_lora.py \
  --model-path ${MODEL_ROOT}/Qwen2.5-1.5B-Base \
  --train-data data/sft_t2t_mini_25k/train.jsonl \
  --eval-data data/sft_t2t_mini_25k/eval.jsonl --eval-samples 1000 --eval-steps 250 \
  --output-dir outputs/affine_vocab/headline/qwen25_1_5b/affine_input_lm_head_plus_hidden_lora \
  --variant affine_input_lm_head_plus_hidden_lora \
  --hidden-lora-rank 8 --hidden-lora-alpha 16 \
  --affine-rank 16 --affine-alpha 128 \
  --max-seq-len 1024 --per-device-train-batch-size 8 --gradient-accumulation-steps 2 \
  --learning-rate 2e-4 --lr-scheduler-type cosine --warmup-ratio 0.03 \
  --num-train-epochs 1 --save-strategy no \
  --master-dtype fp32 --base-dtype auto --bf16 --seed 42
```

常用控制组：

```bash
# Claim 2：frozen base 参照
python scripts/eval_base_loss.py \
  --model-path ${MODEL_ROOT}/Qwen2.5-1.5B-Base \
  --eval-data data/sft_t2t_mini_25k/eval.jsonl \
  --report-file outputs/affine_vocab/claim2_base/qwen25_1_5b.json

# Claim 1b：emb/lm_head 上的 vocab 维 LoRA
bash scripts/run_phase4b_claim1b.sh

# Claim 1a：1.5B 多 seed 稳定性
bash scripts/run_phase4c_multiseed_15b.sh
```

## 当前结果

截至 2026-05-19，三条主张均已有支持性证据：

- Claim 1a：Qwen2.5-1.5B 三 seed 上 `combined_inlmh` 平均 `eval_loss` 1.048，baseline 1.054。
- Claim 1b：1.5B 上 100k AffLoRA 略优于 307k vocab-dim LoRA r=1。
- Claim 2：AffLoRA-only 相比 frozen base 降低约 0.5+ `eval_loss`。
- Claim 3：33k AffLoRA-only 明显优于 33k-49k single-layer LoRA。

完整表格见 `docs/RESULTS_SO_FAR.md`。
