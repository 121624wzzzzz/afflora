from settings import *


def main():
    result=read(HERE/'RESULTS.json');audit=read(HERE/'FINAL_AUDIT.json');assert audit['status']=='passed'
    names={'qwen3_06b_base':'Qwen3-0.6B-Base','qwen25_15b_base':'Qwen2.5-1.5B Base',
           'qwen3_06b_chat':'Qwen3-0.6B 后训练／非 thinking','qwen25_15b_chat':'Qwen2.5-1.5B-Instruct'}
    arms={'input':'仅输入 A-LoRA','output':'仅输出 A-LoRA','hidden_r8':'全层内部 LoRA r8（较大预算）','small_q':'中间层 q_proj LoRA（小预算）'}
    base_table=['| checkpoint | 方法 | 参数 | 候选准确率 | 生成答案准确率 | 完整结束且答对 | 截断率 |',
                '|---|---|---:|---:|---:|---:|---:|']
    secondary=['| checkpoint | 方法 | 候选准确率 | 生成答案准确率 | 严格单字母正确率 |',
               '|---|---|---:|---:|---:|']
    decomposition=['| checkpoint | 方法 | 答案 token CE | EOS token CE | 严格单字母正确率 |',
                   '|---|---|---:|---:|---:|']
    for model,out in result['models'].items():
        reference=out['reference'];records=[('未适配当前 checkpoint',0,reference['primary'],reference['generation'])]
        records += [(arms[arm],v['parameters'],v['primary'],v['generation']) for arm,v in out['arms'].items()]
        for label,parameters,primary,gen in records:
            if model in BASE_MODELS:
                base_table.append(f"| {names[model]} | {label} | {parameters:,} | {primary['candidate_accuracy']:.3f}% | {gen['answer_accuracy']:.3f}% | {gen['completed_answer_accuracy']:.3f}% | {gen['length_cap_rate']:.3f}% |")
                decomposition.append(f"| {names[model]} | {label} | {primary['answer_ce']:.4f} | {primary['eos_ce']:.4f} | {gen['strict_accuracy']:.3f}% |")
            else:secondary.append(f"| {names[model]} | {label} | {primary['candidate_accuracy']:.3f}% | {gen['answer_accuracy']:.3f}% | {gen['strict_accuracy']:.3f}% |")
    contrasts=['| 模型 | 方法 | 指标 | 相对未适配 Δ pp | 校正种子区间 | 题目区间 |',
               '|---|---|---|---:|---|---|']
    for row in result['primary_contrasts']:
        lo,hi=row['corrected_seed_ci95_pp'];ql,qh=row['question_bootstrap95_pp']
        contrasts.append(f"| {names[row['model']]} | {arms[row['arm']]} | {row['metric']} | {row['mean_delta_pp']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | [{ql:+.3f}, {qh:+.3f}] |")
    efficiency=['| 模型 | 指标 | 输入 A-LoRA − 小预算内部 LoRA pp | 校正种子区间 | 题目区间 |',
                '|---|---|---:|---|---|']
    for row in result['efficiency_contrasts']:
        lo,hi=row['corrected_seed_ci95_pp'];ql,qh=row['question_bootstrap95_pp']
        efficiency.append(f"| {names[row['model']]} | {row['metric']} | {row['mean_delta_pp']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | [{ql:+.3f}, {qh:+.3f}] |")
    selections=read(HERE/'SELECTION.json')['selected']
    eos_shares=[]
    for model in BASE_MODELS:
        before=result['models'][model]['reference']['primary'];after=result['models'][model]['arms']['input']['primary']
        answer_drop=before['answer_ce']-after['answer_ce'];eos_drop=before['eos_ce']-after['eos_ce']
        eos_shares.append(100*eos_drop/(answer_drop+eos_drop))
    text=f'''# 修复版 SciQ 任务适配结果

本轮完成 54 个历史 checkpoint 的统一重新评分与实际生成，以及 10 个新小预算对照的五种子评测。新对照先使用验证集在三个学习率中选择，再固定五种子训练；原模型参数始终冻结。本轮数据已被多次查看，属于事后修复和补充对照，不能作为未触碰测试集的新确认研究。

**修复后的结论：** 输入 A-LoRA 的候选准确率点估计仍提高 1.58 / 1.90 pp。Qwen3 的校正种子区间与题目区间均为正；Qwen2.5 在扩展至 16 个主比较后，校正种子区间变为 [-0.082, +3.890] pp，题目区间仍为正，因此不能再说它通过本轮全部主比较标准。这个变化来自比较族扩大，不是原始准确率被改写。

参数量接近的对照带来不同结果：Qwen3 输入 A-LoRA 相对小预算内部 LoRA 为 -0.461 pp，校正区间跨零，没有显示优势，也未证明等效或更差；Qwen2.5 为 +1.563 pp，效率比较族的校正种子区间和题目区间均为正。后者只支持相对这个预定位置和预算控制的优势，不代表超过所有内部 LoRA 配置。输出侧仍没有可靠的候选准确率提升；较大预算内部 LoRA 的绝对准确率最高。

## 已修复的比较与评分设置

- 明确 checkpoint 身份与实验臂：`base` 实验臂只表示未适配，不能把后训练 checkpoint 误称为预训练 Base。
- 每个 adapter 与同一 checkpoint、同一输入 token、同一推理模式、同一解码规则的参考比较。Base 与后训练版本分表；thinking 诊断仅作背景。
- 全部真实 greedy 生成统一允许 128 token；答案内容、严格单字母格式、完整结束和截断分别报告。解析器不接收 gold，用固定规则拒绝含糊答案和明显字母/选项冲突。
- 保留原候选四选一准确率作为共同内容指标；它测固定答案位置上的选项选择。生成解析覆盖率、失败文本和路线保留，不能据此反复调解析规则。
- 新增预先固定中间层 q_proj 的小预算内部 LoRA，单独检验输入 A-LoRA 的参数预算比较。没有把较大预算 r8 当作同预算对照。

## 两个预训练 Base：主实验

每个适配方法为五种子均值，未适配参考为固定 greedy；每次测试的有效题数均为 998，另两道歧义题沿用旧排除规则，原始输出全部保留。

{chr(10).join(base_table)}

生成内容正确不必然意味着输出完整结束；两者同时报告。严格格式正确率不作为内容准确率的替代。当前解析器仅覆盖明确的答案表达，不评价后续解释的真实性，也不保证识别所有语义等价表达。

## 相对未适配起点的效果

主比较族为两个 Base × 四种方法 × 两类内容指标，共 16 个比较。种子区间使用该族的 Bonferroni 校正；题目区间使用 10,000 次配对题目 bootstrap，名义 95%，以已拟合的五个模型为条件。二者回答不同不确定性问题，不能互相替换。

{chr(10).join(contrasts)}

“没有通过正收益区间”不等于严格无效或非劣性成立。事后修复的区间也不能消除此前查看基准造成的研究选择偏差。

## 小预算比较

Qwen3 的两种方法均为 33,792 个参数；Qwen2.5 的输入 A-LoRA 为 50,688 个参数，小预算内部 LoRA 为 49,152 个，少约 3%。内部位置固定为第 14 层（从 0 开始）的 q_proj，Qwen3 rank11/alpha22，Qwen2.5 rank16/alpha32。dropout0.05，无偏置。它只代表这个预定控制，不代表所有内部层或所有低参数方法。

两个新对照的验证集选中学习率分别为 {selections['qwen3_06b_base']['lr']:g} / {selections['qwen25_15b_base']['lr']:g}。数据、训练顺序种子、步数、优化器和三学习率搜索预算与原 Base 适配保持一致；方法的参数化和缩放按各自固定定义使用。

以下四个效率比较构成单独的族，使用族大小 4 校正；不能把两个比较族合称一个覆盖所有结论的总体错误率控制。

{chr(10).join(efficiency)}

## 后训练 checkpoint：次要分析

同样重新生成，保持它们原先的非 thinking 输入协议。这些结果用于验证旧任务评测，不能与 Base 表直接构成仅改变训练阶段的因果比较。

{chr(10).join(secondary)}

官方 Qwen3 thinking 诊断的 83.87% 来自更长预算和另一种采样策略，并且没有训练本轮 adapter；不能与本表适配结果混成配对效果。没有在本轮验证 adapter 的 thinking 保留或其他能力保留。

## 监督目标与损失分解

训练仍然是每题“答案字母 + 原生 EOS”两个监督 token 的任务适配。标准 causal shifted loss 检查和相同 logits 上的监督掩码恒等检查通过；未因更新评测而重训已核验的 A-LoRA checkpoint。它不是完整开放域指令 SFT。

{chr(10).join(decomposition)}

对于两个 Base 的输入 A-LoRA，等权两 token CE 的下降中，EOS 项分别占 {eos_shares[0]:.2f}% / {eos_shares[1]:.2f}%。这解释了为什么本轮任务的 loss 可以大幅下降而选项准确率只小幅提高：大部分 loss 改善来自答案后停止行为。这个比例仅适用于当前两个监督 token 的 SciQ 实验，不能外推到论文原长回答 SFT 的 loss。答案项 CE 的变化也涉及置信度和输出接口，并非纯知识指标。

## 审计与复用

重新核验 {audit['source_sealed_files_verified']} 个既有封存文件。复算 {audit['predictions_recomputed']:,} 条候选预测，重新解码 {audit['generations_redecoded']:,} 条真实生成，54 个旧 checkpoint 的候选预测翻转为零。验证集选择、adapter 哈希、优化器白名单、零残差初始化、冻结参数与评估不改变权重均通过。

新目录包含 20 次训练记录：第一轮两个冒烟尝试（其中 Qwen3 数值容差检查失败），修正检查后的两个成功冒烟，六个验证调参和十个正式种子。第一次失败及独立重载诊断完整保留；修正的是浮点误差检查方式，训练目标和优化设置未改变。详情见 PREFLIGHT_NUMERICS.md。

原研究及其数字未覆盖。新的结果使用 RESULTS.json，审计见 FINAL_AUDIT.json，解析失败见 INVALID_ANSWERS.json，协议见 DESIGN.md，来源见 REUSE_AUDIT.json 与 CHECKPOINT_REUSE.json。

## 结论范围

本轮可用于评估固定设置下的单侧适配和指定小预算对照。通用后训练、知识注入、跨任务、能力保留以及叠加内部 LoRA 后的增量收益仍是独立主张，不能由本表推出。高基线和小增益应如实报告，不能通过换差提示或选择有利模式制造空间。
'''
    (HERE/'FINAL_INTERPRETATION_ZH.md').write_text(text)
    print('Final report written.')


if __name__=='__main__':main()
