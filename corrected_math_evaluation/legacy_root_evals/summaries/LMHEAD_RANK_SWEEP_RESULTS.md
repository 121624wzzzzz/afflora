# hidden LoRA + lm_head-only AffLoRA rank sweep

日期：2026-07-02 至 2026-07-03

## 目的

这次 sweep 用来确认一个具体问题：`hidden LoRA + lm_head-only AffLoRA` 是否真的“基本没用”，以及之前不同 rank / 小样本探索结果为什么和 ar16/hr8 的 full eval 结论不一致。

首轮只评测已有 checkpoint；随后补训缺失的 hr8/ar1、hr8/ar2 checkpoint，并按同一口径评测。所有新跑结果统一使用：

- MATH：`data/math/test.jsonl` 全部 5,000 题，clean 排除 5 条已确认 MetaMathQA train 等价/重复题；
- GSM8K：`data/gsm8k/test.jsonl` 全部 1,319 题；
- shards：8；
- batch size per shard：64；
- max new tokens：512；
- 解码：greedy；
- 输出目录：`corrected_math_evaluation/outputs/lmhead_rank_sweep/`；
- 日志目录：`corrected_math_evaluation/logs/lmhead_rank_sweep/`。

## 健康检查

- 所有 merged JSON 均覆盖完整 dataset index：
  - MATH：0..4999，共 5,000 条；
  - GSM8K：0..1318，共 1,319 条。
- 日志中未发现 `Traceback`、`OOM`、`RuntimeError`、`nan`。
- 新补的 hr8/ar1、hr8/ar2 均完成 2,469/2,469 个训练 step；两者
  `train_loss=0.138`、内部 `eval_loss=0.1391`，训练参数与 Adam 状态均为
  FP32，冻结 base 参数为 BF16。
- `git diff --check -- corrected_math_evaluation` 通过。

## 主结果

delta 按 matching hidden rank 计算：hr4 的 lm_head rank 与 `hidden_lora_hr4` 比，hr8 的 lm_head rank 与 `hidden_lora_hr8` 比。

| hidden rank | lm_head rank | MATH clean | Δ MATH | GSM8K | Δ GSM8K |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | - | 2480/4995 = 49.6496% | +0.0000 pp | 1136/1319 = 86.1259% | +0.0000 pp |
| 4 | 1 | 2515/4995 = 50.3504% | +0.7007 pp | 1137/1319 = 86.2017% | +0.0758 pp |
| 4 | 2 | 2507/4995 = 50.1902% | +0.5405 pp | 1134/1319 = 85.9742% | -0.1516 pp |
| 4 | 4 | 2484/4995 = 49.7297% | +0.0801 pp | 1135/1319 = 86.0500% | -0.0758 pp |
| 4 | 8 | 2476/4995 = 49.5696% | -0.0801 pp | 1132/1319 = 85.8226% | -0.3033 pp |
| 4 | 16 | 2500/4995 = 50.0501% | +0.4004 pp | 1131/1319 = 85.7468% | -0.3791 pp |
| 8 | - | 2489/4995 = 49.8298% | +0.0000 pp | 1137/1319 = 86.2017% | +0.0000 pp |
| 8 | 1 | 2494/4995 = 49.9299% | +0.1001 pp | 1136/1319 = 86.1259% | -0.0758 pp |
| 8 | 2 | 2493/4995 = 49.9099% | +0.0801 pp | 1135/1319 = 86.0500% | -0.1516 pp |
| 8 | 4 | 2496/4995 = 49.9700% | +0.1401 pp | 1131/1319 = 85.7468% | -0.4549 pp |
| 8 | 8 | 2514/4995 = 50.3303% | +0.5005 pp | 1135/1319 = 86.0500% | -0.1516 pp |
| 8 | 16 | 2491/4995 = 49.8699% | +0.0400 pp | 1133/1319 = 85.8984% | -0.3033 pp |

## 结论

这次 sweep 推翻了“hidden + lm_head-only 一概没用”的强表述。

更准确的结论是：

1. `hidden + lm_head-only` 有明显 rank 敏感性。只看 ar16/hr8 会低估这个方向。
2. MATH 上确实存在有效点：
   - hr4/ar1：+0.7007 pp；
   - hr4/ar2：+0.5405 pp；
   - hr8/ar8：+0.5005 pp。
3. 但 GSM8K 上收益不稳：
   - hr4/ar1 略高 +0.0758 pp；
   - 其他 lm_head rank 基本持平或下降，最大下降到 -0.4549 pp。
4. 因此它不是稳定双 benchmark 提升；更像是对 MATH 有帮助、但 rank 选择敏感，并可能牺牲 GSM8K。
5. 之前 200-sample exploratory eval 与 full eval 不一致是合理的：rank 敏感 + 小样本噪声都会造成观察差异。
6. 补齐 hr8/ar1、hr8/ar2 后，hr8 sweep 的低 rank 也没有形成稳定提升：
   MATH 分别只增加 +0.1001、+0.0801 pp，GSM8K 分别下降 -0.0758、
   -0.1516 pp。hr4/ar1 的较好表现不能仅凭 affine rank=1 外推到 hr8。

## 与 ar16/hr8 单点结论的关系

之前我基于 `affine_lm_head_plus_hidden_lora_ar16_s18_hr8_sd42` 说“基本没用”，这个说法只对 ar16/hr8 单点成立：

- MATH clean：49.8699%，比 hidden_hr8 只高 +0.0400 pp；
- GSM8K：85.8984%，比 hidden_hr8 低 -0.3033 pp。

但 sweep 后不能把这个结论外推到所有 rank。尤其是 `ar8/hr8` 和 `ar1/hr4` 在 MATH 上都明显强于对应 hidden baseline。

## 当前建议

如果继续做这个方向，我建议优先保留两个候选：

- `lmhead_ar1_hr4`：MATH +0.7007 pp，GSM8K +0.0758 pp，是当前最干净的双测集非负点；
- `lmhead_ar8_hr8`：MATH +0.5005 pp，GSM8K -0.1516 pp，适合作为高 hidden-rank 下的候选。

后续若要形成强结论，需要补 seed；单 seed 下 0.5–0.7 pp 的 MATH 提升值得关注，但还不应直接写成稳定显著提升。
