"""Posthoc presentation of frozen analysis and descriptive validity partitions."""
import csv,json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from safetensors import safe_open

ROOT=Path(__file__).resolve().parent/'llama_lr_20260918'
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s]
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
LABELS={'tuned_hidden':'调参 H','tuned_budget':'调参等参 H','tuned_heu':'调参 H+E+U','anchor_budget':'原设置等参 H','anchor_heu':'原设置 H+E+U'}
EN={'tuned_hidden':'Tuned H','tuned_budget':'Tuned budget H','tuned_heu':'Tuned H+E+U','anchor_budget':'Original budget H','anchor_heu':'Original H+E+U'}

def main():
    assert read(ROOT/'FINAL_AUDIT.json')['status']=='passed' and not (ROOT/'SEAL.json').exists()
    a=read(ROOT/'ANALYSIS.json');sel=read(ROOT/'SELECTION.json');gold=rows(ROOT/'data/wikisql_confirm.jsonl')
    pairs={};secondary={}
    embedding_rms=read(ROOT/'BOUNDARY_WEIGHT_SCALES_POSTHOC.json')['models']['llama31_8b_base']['values']['model.embed_tokens.weight']['native_token_frequency_weighted_rms']
    for role,rs in a['results'].items():
        values=[]
        for r in rs:
            s=read(ROOT/'evaluations'/r['run']/'confirm/SUMMARY.json')
            value={'seed':r['seed'],**{k:s[k] for k in ['primary','lf_correct_pct','query_valid_pct','strict_json_pct','capped_pct','distinct_parameter_execution_correct_pct','empty_prediction_pct','duplicate_condition_column_pct']}}
            with safe_open(str(ROOT/'checkpoints'/r['run']/'adapter.safetensors'),framework='pt',device='cpu') as f:
                if 'boundary_input.bias' in f.keys():
                    bias_rms=float(f.get_tensor('boundary_input.bias').double().square().mean().sqrt())
                    value.update(input_bias_rms=bias_rms,input_bias_rms_over_non_special_training_embedding_rms=bias_rms/embedding_rms)
            values.append(value)
        secondary[role]=values
    write(ROOT/'SECONDARY_METRICS_POSTHOC.json',{'description':'Descriptive readout of metrics already calculated by the frozen scorer; no selection or extra significance tests.','roles':secondary})
    for c in a['comparisons']:
        key=c['a']+' minus '+c['b'];per_seed=[]
        for x,y in zip(a['results'][c['a']],a['results'][c['b']]):
            xs=rows(ROOT/'evaluations'/x['run']/'confirm/responses.jsonl');ys=rows(ROOT/'evaluations'/y['run']/'confirm/responses.jsonl')
            assert [r['id'] for r in xs]==[r['id'] for r in ys]==[r['id'] for r in gold]
            wins=sum(u['content_correct'] and not v['content_correct'] for u,v in zip(xs,ys))
            losses=sum(v['content_correct'] and not u['content_correct'] for u,v in zip(xs,ys))
            bv=sum(int(u['content_correct'])-int(v['content_correct']) for u,v in zip(xs,ys) if u['query_valid'] and v['query_valid'])
            other=wins-losses-bv
            empty=sum(int(u['content_correct'])-int(v['content_correct']) for u,v,g in zip(xs,ys,gold) if g['gold_execution']==[])
            assert 100*(wins-losses)/len(xs)==x['primary']-y['primary']
            per_seed.append({'seed':x['seed'],'recovered_questions':wins,'regressed_questions':losses,
                             'net_questions':wins-losses,'both_valid_net_pp':100*bv/len(xs),'other_validity_net_pp':100*other/len(xs),
                             'empty_gold_net_pp':100*empty/len(xs),'nonempty_gold_net_pp':100*(wins-losses-empty)/len(xs)})
        pairs[key]=per_seed
    write(ROOT/'DESCRIPTIVE_PAIRED_CASES_POSTHOC.json',{'description':'Fixed-all-confirmation-denominator validity and empty-gold partitions; descriptive, not causal decompositions.','empty_gold_examples':sum(g['gold_execution']==[] for g in gold),'pairs':pairs})
    with (ROOT/'ALL_RESULTS.csv').open('w') as f:
        fields=['stage','role','arm','candidate','seed','hidden_lr','boundary_lr_ratio','primary','lf_correct_pct','query_valid_pct','run']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for arm,cs in sel['scores'].items():
            for c in cs:
                for seed,value in zip([7200,7201],c['seed_scores']):
                    name=f'search_{arm}_{c["id"]}_s{seed}';s=read(ROOT/'evaluations'/name/'dev/SUMMARY.json')
                    writer.writerow(dict(stage='search',role='',arm=arm,candidate=c['id'],seed=seed,hidden_lr=c['lr'],boundary_lr_ratio=c['boundary_lr_ratio'],primary=value,lf_correct_pct=s['lf_correct_pct'],query_valid_pct=s['query_valid_pct'],run=name))
        for role,rs in a['results'].items():
            for r in rs:
                s=read(ROOT/'checkpoints'/r['run']/'spec.json')
                writer.writerow(dict(stage='confirmation',role=role,arm=s['arm'],candidate=s['candidate'],seed=r['seed'],hidden_lr=s['lr'],boundary_lr_ratio=s['boundary_lr_ratio'],primary=r['primary'],lf_correct_pct=r['lf_correct_pct'],query_valid_pct=r['query_valid_pct'],run=r['run']))
    figures=ROOT/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(13.2,4.5),gridspec_kw={'width_ratios':[1,1.35]},layout='constrained')
    colors={'hidden':'#526176','hidden_budget':'#d58421','hidden_both':'#166c80'}
    for arm in ['hidden','hidden_budget','hidden_both']:
        for ratio in ([1.,.25] if arm=='hidden_both' else [1.]):
            rs=sorted([r for r in sel['scores'][arm] if r['boundary_lr_ratio']==ratio],key=lambda r:r['lr'])
            label={'hidden':'H','hidden_budget':'Budget H','hidden_both':'H+E+U'}[arm]+(f' (boundary/H={ratio:g})' if arm=='hidden_both' else '')
            axes[0].plot([r['lr'] for r in rs],[r['mean'] for r in rs],marker='o',linestyle='--' if ratio==.25 else '-',label=label,color=colors[arm],alpha=.75 if ratio==.25 else 1)
    axes[0].set_xscale('log');axes[0].set_xlabel('Hidden learning rate');axes[0].set_ylabel('Development execution accuracy (%)')
    axes[0].set_xticks([5e-5,1e-4,2e-4,4e-4,8e-4],['5e-5','1e-4','2e-4','4e-4','8e-4'])
    axes[0].set_title('Development selection (2 seeds, 1,024 examples)');axes[0].legend(fontsize=8);axes[0].grid(alpha=.18)
    cs=a['comparisons']
    for i,c in enumerate(cs):
        m=c['mean_pp'];lo,hi=c['bonferroni5_ci95'];nlo,nhi=c['ci95']
        axes[1].errorbar(m,i,xerr=[[m-lo],[hi-m]],color='#a7b2ba',capsize=4,lw=2)
        axes[1].errorbar(m,i,xerr=[[m-nlo],[nhi-m]],fmt='o',color='#166c80',capsize=3,lw=3)
    axes[1].axvline(0,color='#777777',lw=1,linestyle='--');axes[1].set_yticks(range(len(cs)),[EN[c['a']]+' − '+EN[c['b']] for c in cs],fontsize=8)
    axes[1].invert_yaxis();axes[1].set_xlabel('Paired execution-accuracy difference (pp)')
    axes[1].set_title('Confirmation (5 seeds, 2,048 examples)\nDark: marginal 95% CI; light: family-5 95% CI',fontsize=10);axes[1].grid(axis='x',alpha=.18)
    fig.savefig(figures/'lr_selection_and_confirmation.pdf');fig.savefig(figures/'lr_selection_and_confirmation.png',dpi=180);plt.close(fig)
    lines=['# Llama-8B WikiSQL 学习率对照实验','','本轮已经完成开发集选参、五种子独立验证和最终审计。此前固定超参的 Llama 负结果保留，不被本轮替换。',
           '','## 设置与选择','','同一模型、2,048 条训练数据、64 次更新，参数量、rank、alpha、裁剪阈值和评测器不变。每种方法 6 个候选 × 2 个开发种子；候选空间维度不同，拟合次数与处理的训练样本数相同，不声称 FLOPs 或耗时完全相同。开发集 1,024 条，验证集 2,048 条，均排除了项目此前已使用的表格。',
           '','| 方法 | 选定内部 LR | E/U 与内部 LR 比例 | 开发集两种子均值 |','|---|---:|---:|---:|']
    for arm,label in [('hidden','H'),('hidden_budget','等参 H'),('hidden_both','H+E+U')]:
        c=sel['selected'][arm];v=next(r for r in sel['scores'][arm] if r['id']==c['id'])
        lines.append(f'| {label} | {c["lr"]:g} | {c["boundary_lr_ratio"]:g} | {v["mean"]:.4f} |')
    identity_note='存在相同配置的标签，复用关系见 SELECTION.json。' if any(c['identity'] for c in a['comparisons']) else '本轮五种设置互不相同，共 25 次新拟合，方法之间按种子配对；不复用旧训练结果。'
    lines+=['','## 新验证集结果','','所有数值为百分比；seed 为 7300–7304。'+identity_note,
            '','| 方法 | 五种子均值 | 标准差 | 逻辑形式准确率 | 合法查询率 |','|---|---:|---:|---:|---:|']
    for role,rs in a['results'].items():
        lines.append(f'| {LABELS[role]} | {statistics.mean(r["primary"] for r in rs):.4f} | {statistics.stdev(r["primary"] for r in rs):.4f} | {statistics.mean(r["lf_correct_pct"] for r in rs):.4f} | {statistics.mean(r["query_valid_pct"] for r in rs):.4f} |')
    lines+=['','| 预先指定的配对比较 | 均值差（百分点） | 五个种子的差值 | 家族 5 校正 95% 区间 |','|---|---:|---|---|']
    for c in a['comparisons']:
        lo,hi=c['bonferroni5_ci95'];ds=', '.join(f'{v:+.3f}' for v in c['paired_seed_deltas_pp'])
        label=LABELS[c['a']]+' − '+LABELS[c['b']]+('（同一配置）' if c['identity'] else '')
        lines.append(f'| {label} | {c["mean_pp"]:+.4f} | {ds} | [{lo:+.4f}, {hi:+.4f}] |')
    lines+=['','置信区间描述固定数据集下的种子波动，不覆盖数据集抽样或预训练污染的不确定性。新旧验证集的表格分布不同，不能用新旧绝对分数之差声称调参收益；调参收益必须来自本表同一验证集上的配对对照。',
            '','## 实现与诊断','','原设置 H+E+U 和等参 H 的复现均通过了初始/最终权重及逐步 loss、梯度范数的精确比对。最初 GPU 诊断分配影响了等参 H 的浮点轨迹；改为 CPU 记录后恢复精确复现。失败预检完整归档在相邻 `llama_lr_20260918_preflight_v1`，正式搜索在修订后才启动。',
            '','边界尺度诊断见 [BOUNDARY_SCALE_DIAGNOSIS_ZH.md](BOUNDARY_SCALE_DIAGNOSIS_ZH.md)。它提出输入偏置相对尺度的机制假设；当前学习率实验不能将 E 偏置、E 低秩项和 U 的作用分别归因。合法查询分区见 `DESCRIPTIVE_PAIRED_CASES_POSTHOC.json`，只作描述。',
            '','完整开发网格、每种子结果和两类区间见 [RESULTS.md](RESULTS.md)，梯度诊断和预先指定统计见 `ANALYSIS.json`。',
            '','![开发集搜索与新验证集配对结果](figures/lr_selection_and_confirmation.png)',
            '','## 结论复核','','此处由完成后的结果复核补充；在补充前不封存本轮。']
    (ROOT/'FINAL_INTERPRETATION_ZH.md').write_text('\n'.join(lines)+'\n')
    print('report tables, paired cases and PDF/PNG figures generated',flush=True)
if __name__=='__main__':main()
