"""Post-output reporting only; does not change the frozen Llama analysis.

Created after the first two seeds were partly observed. Secondary summaries and
valid-query contribution counts are descriptive, not new inferential tests.
"""
import csv, hashlib, json
from datetime import datetime
from pathlib import Path
from statistics import mean

HERE=Path(__file__).resolve().parent
R=HERE if HERE.name=='llama_transfer_20260918' else HERE/'llama_transfer_20260918'
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')

def main():
    audit=read(R/'FINAL_AUDIT.json');assert audit['status']=='passed'
    analysis=read(R/'TEST_ANALYSIS.json');assert len(analysis['conditions'])==4 and all(c['complete'] for c in analysis['conditions'])
    names={'llama32_3b_base':'Llama-3.2-3B Base','llama31_8b_base':'Llama-3.1-8B Base'}
    metrics={'cluener':['primary','span_precision','span_recall','schema_valid','strict_json','capped_pct'],
             'wikisql':['primary','lf_correct_pct','query_valid_pct','strict_json_pct','capped_pct']}
    rows=[];secondary=[];sql=[]
    for c in analysis['conditions']:
        task,model=c['task'],c['model'];store={}
        for arm in ['base','hidden','hidden_budget','hidden_both']:
            for seed in ([None] if arm=='base' else c['seeds']):
                name=f'{task}_{model}_'+('base' if seed is None else f'{arm}_s{seed}')
                a=read(R/'audits'/f'{name}.json');assert a['status']=='passed'
                for split in ['dev','test']:
                    s=read(R/'evaluations'/name/split/'SUMMARY.json')
                    rows.append(dict(task=task,model=model,arm=arm,seed=seed,split=split,primary=s['primary'],summary=str(R/'evaluations'/name/split/'SUMMARY.json')))
                    store[arm,seed,split]=s
            for split in ['dev','test']:
                ss=[store[arm,seed,split] for seed in ([None] if arm=='base' else c['seeds'])]
                secondary.append(dict(task=task,model=model,arm=arm,split=split,means={k:mean(s[k] for s in ss) for k in metrics[task] if k in ss[0]}))
        if task=='wikisql':
            for seed in c['seeds']:
                for control in ['hidden','hidden_budget']:
                    def responses(arm):
                        p=R/'evaluations'/f'{task}_{model}_{arm}_s{seed}'/'test/responses.jsonl'
                        return [json.loads(line) for line in p.read_text().splitlines()]
                    treat,ctrl=responses('hidden_both'),responses(control)
                    assert [r['id'] for r in treat]==[r['id'] for r in ctrl]
                    both=sum(int(x['content_correct'])-int(y['content_correct']) for x,y in zip(treat,ctrl) if x['query_valid'] and y['query_valid'])
                    other=sum(int(x['content_correct'])-int(y['content_correct']) for x,y in zip(treat,ctrl) if not (x['query_valid'] and y['query_valid']))
                    sql.append(dict(model=model,seed=seed,control=control,n=len(treat),both_valid_delta_pp=100*both/len(treat),other_delta_pp=100*other/len(treat),total_delta_pp=100*(both+other)/len(treat)))
    with (R/'ALL_FORMAL_RESULTS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(R/'SECONDARY_READOUT.json',{'at':datetime.now().astimezone().isoformat(),'scope':'Post-output descriptive summaries of predefined secondary metrics; SQL valid-query partition is post hoc and not causal','metrics':secondary,'sql_validity_partition':sql})
    positive=sum(all(x['mean']>0 for x in c['comparisons']) for c in analysis['conditions'])
    neg=sum(all(x['mean']<0 for x in c['comparisons']) for c in analysis['conditions'])
    corrected=[(c,x) for c in analysis['conditions'] for x in c['comparisons'] if x['bonferroni8_95_ci'][0]>0]
    lines=['# Llama 跨家族叠加实验：最终结果','',
           f"本轮已完成 36 次正式训练、4 项 Base 评测和 8 次技术短程试跑。四个模型／任务条件均有完整的三个预定种子；{audit['responses_checked']:,} 条输出记录通过独立审计。没有复用旧的 Llama 训练结果。",'',
           f'四个条件中，叠加组相对普通与预算 LoRA 的均值均正的有 {positive} 个，均负的有 {neg} 个；其余为持平或混合。八项预定比较中，校正区间完全高于零的有 {len(corrected)} 项。完整逐种子结果和区间见 TEST_ANALYSIS.json / RESULTS.md。','',
           'CLUENER 在两款 Llama 上均为三个种子同时高于普通与预算 LoRA，支持这个实体抽取任务的跨家族收益方向。WikiSQL 的 3B 是两正一负、均值为正；8B 相对预算是两负一正、均值为负，未复现整体额外收益。结合既有 Qwen 结果，当前更清晰的主张是指定任务上的条件性叠加收益，不能写成跨模型、跨任务普遍增益。本轮八个未校正的描述性种子区间也均跨零；统计不确定性不只是来自多重比较校正。','',
           '|任务|模型|Base|普通 LoRA|预算 LoRA|叠加 aLoRA|Δ普通|Δ预算|',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for c in analysis['conditions']:
        m=c['means'];d=c['comparisons']
        lines.append(f"|{c['task']}|{names[c['model']]}|{m['base']:.3f}|{m['hidden']:.3f}|{m['hidden_budget']:.3f}|{m['hidden_both']:.3f}|{d[0]['mean']:+.3f}|{d[1]['mean']:+.3f}|")
    lines+=['','分数按 0–100 计：CLUENER 为 span micro-F1，WikiSQL 为官方执行正确率。均值只使用预定的三个种子；Base 每条件评测一次。','',
            '## 每个配对种子与不确定性','',
            '|任务／模型|对照|三个差值|均值|Bonferroni-8 95% 种子 t 区间|',
            '|---|---|---|---:|---|']
    for c in analysis['conditions']:
        for x in c['comparisons']:
            raw=', '.join(f'{v:+.4f}' for v in x['differences']);lo,hi=x['bonferroni8_95_ci']
            lines.append(f"|{c['task']} / {names[c['model']]}|{x['control']}|{raw}|{x['mean']:+.4f}|[{lo:.4f}, {hi:.4f}]|")
    lines+=['','固定八项比较、df=2；没有依据结果改种子、缩小比较族、切换主要指标或选择 checkpoint。正向均值或三个正号本身不等同于通过校正检验；跨零也不能证明无效。这些区间描述固定数据与设置下的训练种子变异，不覆盖所有数据抽样、模型或超参数不确定性。此前 Qwen 的三／五种子统计族分别保留，不合并成新的确认性检验。','',
            '## 参数预算与设置','',
            '|模型|普通内部 LoRA|预算 LoRA|内部 LoRA＋双侧 aLoRA|对照多出的参数|',
            '|---|---:|---:|---:|---:|']
    for model,b in read(R/'BUDGET_PLAN.json').items():lines.append(f"|{names[model]}|{b['hidden']:,}|{b['hidden_budget']:,}|{b['hidden_both']:,}|{b['budget_excess']:,}|")
    lines+=['','8B 严格等参；3B 对照多 1,024 个参数，占叠加组训练参数的约 0.00829%，占新增边界参数预算的约 0.513%。沿用与 Qwen 相同的 q/k 增 rank 规则，最小化非负差额，新增参数均参与实际计算。q/k 单 rank 成本在 3B 为 6,144／4,096，最大公约数 2,048；边界预算 199,680 不是其整数倍，因此该统一规则的最小超额为 1,024。8B 可以精确匹配。共同 rank-8 初始化和 alpha/r=2 的缩放保持一致，实际初始权重与样本顺序已逐组检查。','',
            '允许预算对照略多是合理的；该比较至少没有给对照更少的参数预算，但不意味着更多参数必然带来更高分。这里比较的是固定学习率、训练步数和结构的完整适配方案，尚不能把差异归因为纯放置位置或证明双方各自充分调参后的最优差异。','',
            '两个官方 Base 权重、配置、分词器分别与 Meta 固定版本的 git/LFS 身份核验；3B 实际 tied、8B untied。版本、架构与预训练差异使它们不是纯尺寸实验。任务样本、原始提示和评分器从封存 Qwen 实验逐文件验证后复制；Llama 使用自己的 BOS 128000 和 EOS 128001，不套用 Qwen token，也不使用聊天模板。所有样本通过长度检查，无截断或结果相关筛选。','',
            '每任务 2,048 条训练、一轮 64 步、有效 batch32、LR2e-4，H rank8/alpha16/dropout0.05，E/U rank16/alpha128/dropout0，E 有 bias、U 无 bias。原权重冻结；adapter 与优化器 FP32，训练 BF16，评测 FP32 greedy。只评测最终 checkpoint。无按当前结果调参。完整细节和冻结身份见 PROTOCOL.md、CODE_FROZEN.json、DATA_FROZEN.json。','',
            '## 格式与内容辅助指标','',
            '|任务／模型|方案|开发集主分数|测试集主分数|格式有效率|辅助内容指标|',
            '|---|---|---:|---:|---:|---|']
    for c in analysis['conditions']:
        for arm in ['base','hidden','hidden_budget','hidden_both']:
            get=lambda split:next(x['means'] for x in secondary if (x['task'],x['model'],x['arm'],x['split'])==(c['task'],c['model'],arm,split))
            dev,test=get('dev'),get('test');fmt=test.get('schema_valid',test.get('query_valid_pct'))
            aux=f"LF={test['lf_correct_pct']:.3f}" if c['task']=='wikisql' else 'span micro-F1 已作为主指标'
            lines.append(f"|{c['task']} / {names[c['model']]}|{arm}|{dev['primary']:.3f}|{test['primary']:.3f}|{fmt:.3f}|{aux}|")
    lines+=['','WikiSQL 格式有效率为 query_valid，不仅是 JSON 可解析。Base 的低分包含输出格式、停止行为及任务协议适应，不表示缺乏底层知识。额外收益的核心证据是相对两个已训练 LoRA 对照的比较。8B CLUENER 开发集叠加均值略低于普通 LoRA；8B WikiSQL 开发集叠加均值高于两对照，而测试集均值下降。这些相反方向均保留，不能据此确认过拟合或优化冲突的具体机制。','',
            '以下为事后描述性分区：固定全测试集分母，将正确题数差分为双方查询均有效的样本与其余样本。它不改主要指标，也不作因果归因或新的显著性检验。','',
            '|模型|对照|均有效样本贡献 pp|其余样本贡献 pp|总差 pp|',
            '|---|---|---:|---:|---:|']
    for model in names:
        for control in ['hidden','hidden_budget']:
            ss=[x for x in sql if x['model']==model and x['control']==control]
            lines.append(f"|{names[model]}|{control}|{mean(x['both_valid_delta_pp'] for x in ss):+.4f}|{mean(x['other_delta_pp'] for x in ss):+.4f}|{mean(x['total_delta_pp'] for x in ss):+.4f}|")
    lines+=['','## 审计与范围','',
            f"- 正式运行 40/40、短程检查 8/8；输出 {audit['responses_checked']:,} 条，官方 SQL 有效预测复核 {audit['official_sql_checks']:,} 条；分词重新编码 {audit['tokens_reencoded']:,} 条；12 组三方案共同初始化与训练顺序核验。",
            '- 原权重训练前后摘要一致、梯度不存在；所有指定适配组确实更新；实际可训练参数数量及范围、FP32 优化器状态和保存后破坏／重载等价检查通过。',
            '- 所有四个条件和三个种子完整保留。任务因此前 Qwen 正结果而选择，因此这是指定任务的模型家族迁移验证，不能外推任意任务。公共数据预训练暴露仍不能排除。',
            '- 本轮没有新增 standalone 或 E-only/U-only 对照；不能据此判断 Llama 的最佳侧或独立边界适配效率。原 Qwen 独立侧别实验保持独立。',
            '- 初始模型、数据、拟合、评分和主统计代码均在输出前冻结。本文和 SECONDARY_READOUT.json 为输出后报告；查询有效性分区是事后描述分析。完整原始预测、评分和训练记录均保留。','',
            '主要可支持的论点应依据上表限定到实际正向条件，并保留其他条件的持平／负向结果及统计不确定性。']
    (R/'FINAL_INTERPRETATION_ZH.md').write_text('\n'.join(lines)+'\n')
    script=Path(__file__);write(R/'REPORTING_PROVENANCE.json',{'at':datetime.now().astimezone().isoformat(),'script':str(script),'sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'post_output':True,'changes_frozen_primary_analysis':False})
    print('Wrote final report; jointly positive means',positive,'jointly negative',neg,'corrected positive comparisons',len(corrected))
if __name__=='__main__':main()
