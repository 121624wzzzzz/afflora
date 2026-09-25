from common import HERE,read

NAMES={'qwen3_06b_chat':'Qwen3-0.6B','qwen25_15b_chat':'Qwen2.5-1.5B-Instruct'}
CONTROLS={'none':'普通 hidden LoRA','hidden_budget':'等参数预算 hidden LoRA'}

def main():
    r=read(HERE/'RESULTS.json');audit=read(HERE/'FINAL_AUDIT.json');assert audit['status']=='passed'
    successful=[NAMES[m] for m,v in r['model_confirmations'].items() if v]
    missing=[NAMES[m] for m,v in r['model_confirmations'].items() if not v]
    lines=['# A-LoRA 叠加收益：五个新训练种子的确认结果','',
        '固定的30组新训练和完整评测已全部完成，并通过最终审计。主要结论只使用新种子1001–1005；旧42/43/44未混入分析。','',
        ('两个模型均达到预先固定的概率收益确认标准。' if r['joint_two_model_confirmation'] else '未达到跨两个模型的联合概率收益确认标准。')]
    if successful:lines+=['达到模型内两个对照均通过校正区间标准的模型：'+ '、'.join(successful)+'。']
    if missing:lines+=['尚未达到该标准的模型：'+'、'.join(missing)+'。未通过不等于证明效应为零，应同时看效应大小与区间。']
    lines+=['','主要指标为每题所有**不同参考答案字符串加结束符**的概率之和，再对3219题等权平均。下表差值为 both−control，概率提升单位为百分点；不是生成准确率百分点。',
        '', '| 模型 | 对照 | 平均概率差 pp | 四对比校正区间 pp | 通过 |','|---|---|---:|---|---|']
    for x in r['primary_contrasts']:
        v=x['seed_bonferroni_family4'];lines.append(f'| {NAMES[x["model"]]} | {CONTROLS[x["control"]]} | {100*v["mean"]:+.4f} | [{100*v["ci"][0]:+.4f}, {100*v["ci"][1]:+.4f}] | {"是" if x["confirmed"] else "否"} |')
    lines+=['','配对 t 区间基于5个训练种子、df=4，主要4对比使用 Bonferroni 校正。每个模型必须同时优于普通 LoRA 和等预算 LoRA。逐种子值和名义区间见 RESULTS.md / RESULTS.json。小样本 t 推断依赖种子效应分布假设。',
        '', '## 概率与生成效果分别判断','',
        '以下为预先保留的支持指标。NLL 越低越好，EM/F1/AVG 越高越好。所有括号为未作多重校正的名义95%种子区间，不能与主要确认判据混用。',
        '', '| 模型 / 对照 | 集合 NLL 差 | EM 差 pp | F1 差 pp | AVG 差 pp |','|---|---|---|---|---|']
    for model in NAMES:
        for control in CONTROLS:
            values=[]
            for metric in ['answer_set_nll','em','f1','avg']:
                x=next(x for x in r['contrasts'] if x['model']==model and x['control']==control and x['split']=='public' and x['metric']==metric)
                v=x['seed_nominal95'];k=1 if metric=='answer_set_nll' else 100
                values.append(f'{k*v["mean"]:+.5f} [{k*v["ci"][0]:+.5f}, {k*v["ci"][1]:+.5f}]')
            lines.append('| '+NAMES[model]+' / '+CONTROLS[control]+' | '+' | '.join(values)+' |')
    lines+=['', '**生成与内容 CE 的负面结果必须保留：**Qwen2.5 相对普通 hidden LoRA 的 EM、AVG 在5个新种子上全部下降，均值分别−0.25474、−0.11249个百分点；上表名义区间均在0以下。这些支持指标未做多重比较校正，但方向一致的退化不能被主要概率指标的成功掩盖。相对等预算对照的 EM/AVG 均值也为负，区间跨0。',
        '', 'Qwen2.5 内容 macro CE 对两个对照分别增加0.00157183、0.00169114，5个种子均变差；相应未校正区间分别[0.00017268, 0.00297099]、[0.00035049, 0.00303179]。内容 micro CE 对普通LoRA微降、对等预算对照微升，两条区间均跨0。Qwen3 的集合NLL和内容micro CE则在5个种子上都改善。',
        '', '因此，本轮支持的是**平均完整参考答案概率的增量**，不能笼统改写成所有概率建模指标或生成质量均改善。四个主要校正区间中有三个下界非常接近0，5种子区间仍较宽；这是一项有明确范围的小幅收益证据。',
        '', '## 为什么概率收益没有稳定转化为生成收益','',
        '对全部逐题输出的描述性分解显示，约96.9%–97.4%的比较中EM判定没有改变。Qwen2.5对普通LoRA的总体概率差为+0.51412个百分点，其中双方都EM正确的部分贡献+0.51523个百分点，双方都EM错误的部分贡献−0.01963个百分点；对等预算对照的对应数值为+0.55404、+0.54007、−0.01228个百分点。',
        '', '按五次训练逐题合计，Qwen2.5对普通LoRA有228次从错变对、269次从对变错；对等预算对照为225次与267次。主要概率收益集中于原本已经答对的问题，同时出现少量净准确率退化。参考答案概率提高并不保证greedy解码改变为正确答案，因而两类指标可以不同向。',
        '', '此处“次”包含同一道题在不同训练种子下的重复比较，不是独立题目数。分组是结果的描述性分解，不是训练机制的因果证明，也不进入四个主要检验。完整分解见 GENERATION_PROBABILITY_LINK.json / .md。']
    tradeoffs=[x for x in r['contrasts'] if x['split']=='public' and x['metric']=='answer_set_nll' and x['seed_nominal95']['mean']>0]
    if tradeoffs:lines+=['','存在集合 NLL 均值变差的比较：'+ '；'.join(NAMES[x['model']]+' 对 '+CONTROLS[x['control']] for x in tradeoffs)+'。因此不能笼统说所有概率指标都改善。']
    else:lines+=['','四个比较的集合 NLL 均值均向改善方向变化；是否超出种子不确定性需按各自区间判断。']
    lines+=['','内容 micro/macro CE、EOS、first/rest token 指标及内部验证集结果全部保存在 RESULTS.json / paired_effects.csv。篇章簇 bootstrap 使用848簇、10000次重采样，以5个已拟合模型为条件；它不替代训练种子的区间。',
        '', '## 有效性检查','',
        '- 30组均从经哈希验证的原始聊天模型重新训练570步；数据9114 train / 1028 internal dev，没有续训或复用旧 adapter。',
        '- 10个模型×种子配对组共享 hidden 初始化；双边 A-LoRA 与一个预定的 hidden rank 分配对照精确匹配参数，额外参数分别66560/99840。',
        '- 所有可训练权重有限且为FP32；冻结源码、数据、模型文件与结果哈希再次核验。',
        f'- 逐参考概率独立重算：公开集共{audit["unique_public_reference_evaluations"]}个唯一答案评测，原始重复参考按映射复用，去重集合概率使用独立 exp/sum/log 核对。',
        f'- {audit["public_question_evaluations"]}条生成结果重新解码和官方兼容评分；达到256 token上限的输出共{audit["generation_cap_hits"]}条。',
        f'- {audit["numerical_canonical_exact_replays"]}个固定案例的 canonical 重放逐位一致；改变batch/填充后的最大token NLL差 {audit["max_shape_token_nll_difference"]:.8g}，概率差 {audit["max_shape_probability_difference"]:.8g}。',
        f'- FP32 CE与FP64对照最大差 {audit["max_fp32_ce_vs_fp64_difference"]:.8g}；所需位置投影与完整logits最大token NLL差 {audit["max_full_logits_token_nll_difference"]:.8g}。',
        '', '6条达到长度上限的输出全部来自Qwen3的两个问题，Qwen2.5没有此类输出。仅作描述性敏感性检查，排除任一配对臂达到上限的案例后，Qwen3对两个对照的AVG均值差为+0.19889、+0.22638个百分点，与全量+0.19906、+0.22644基本一致；Qwen2.5的下降不涉及长度截断。正式指标仍保留全部样本，详见 GENERATION_CAP_DIAGNOSTIC.json。',
        '', '首次两步烟测有一个形状检查略超最初NLL阈值，失败和调查记录完整保留。在任何正式训练前加入canonical逐位重放检查，并固定形状诊断容差NLL≤5e-4、概率≤1e-4；全部6臂重跑通过后才冻结正式协议。详见 DESIGN.md 与 protocol_history。有限案例的数值检查不是整个测试集误差上界。',
        '', '## 论文主张的范围','',
        '这轮只确认两个模型、一个任务、一个固定训练配置和一个预定等预算分配上的结果。CMRC public dev已经用于前轮探索；新种子是独立训练重复，但评测集并非未看过的新测试集。',
        '', '完整参考概率不包含全部语义等价答案，也不直接表示 greedy 生成正确率或概率校准。训练仍为BF16 AMP/FP32 adapter；本轮推理改为FP32 eager，不能将旧BF16与本轮的绝对差值直接解释成方法或种子的效果。',
        '', '论文应分别陈述通过校正检验的概率收益和实际EM/F1证据，保留未通过与负向结果。当前实验不支持 input-only、权重共享的因果机制、所有内部rank分配或跨任务普遍生成收益的结论。',
        '', '产物：RESULTS.json、paired_effects.csv、FINAL_AUDIT.json，以及 figures/new_seed_confirmation.png / .pdf。原始逐题概率、生成token和文本在 outputs/。']
    (HERE/'FINAL_INTERPRETATION_ZH.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
