import math,collections
import numpy as np
from scipy.stats import t
from common import *
from prepare import components

SEEDS=list(range(6100,6105))
ARMS=['hidden','input','output','hidden_both','hidden_budget']
def name(task,model,arm,seed):
    return f'pilot_{task}_{model}_hidden' if arm=='hidden' and seed==6100 else f'adapter_{task}_{model}_{arm}_s{seed}'
def get(task,model,arm,seed):
    path=HERE/'heldout'/name(task,model,arm,seed)/'final'
    return read(path/'SUMMARY.json'),rows(path/'responses.jsonl')
def seed_interval(values,family):
    a=np.asarray(values);mean=float(a.mean());half=float(t.ppf(1-.05/(2*family),len(a)-1)*a.std(ddof=1)/math.sqrt(len(a)))
    return {'mean':mean,'lower':mean-half,'upper':mean+half,'seed_deltas':a.tolist(),'family_size':family}
def main():
    assert read(HERE/'HELDOUT_COMPLETE.json')['status']=='passed'
    assert read(HERE/'HELDOUT_AUDIT.json')['status']=='passed'
    tables=[];contrasts=[];reuse=read(HERE/'PILOT_REUSE.json')
    for run in reuse['runs']:
        for rel,h in run['files'].items():assert sha(HERE/rel)==h
    for task in TASKS:
        dev=rows(HERE/f'data/{task}_test.jsonl');ids=[r['id'] for r in dev]
        if task=='toolace':
            mapping={r['id']:i for i,g in enumerate(components(rows(HERE/'data/toolace_test.jsonl'))) for r in g}
            labels=[mapping[r['id']] for r in dev]
        else:labels=[r['text'] for r in dev]
        unique=sorted(set(labels));membership=np.zeros((len(unique),len(dev)))
        for i,label in enumerate(unique):membership[i]=np.array([v==label for v in labels])
        rng=np.random.default_rng(20260916)
        draws=rng.multinomial(len(unique),np.ones(len(unique))/len(unique),size=10000)
        weights=draws@membership;denoms=weights.sum(axis=1)
        for model in MODELS:
            baseline=read(HERE/'heldout'/f'pilot_{task}_{model}_base'/'baseline/SUMMARY.json')
            tables.append({'task':task,'model':model,'arm':'base','seed_count':0,'primary':baseline['primary'],
                'text_micro_f1':baseline.get('text_micro_f1'),'json_valid':baseline['json_valid'],'capped_pct':baseline['capped_pct'],
                'eval_split':'test'})
            values={};records={}
            for arm in ARMS:
                results=[get(task,model,arm,s) for s in SEEDS];summaries=[r[0] for r in results];records[arm]=[r[1] for r in results]
                values[arm]=[s['primary'] for s in summaries]
                for r in records[arm]:assert [x['id'] for x in r]==ids
                for seed in SEEDS:
                    cp=HERE/'checkpoints'/name(task,model,arm,seed);initial=read(cp/'INITIALIZATION.json')
                    if arm.startswith('hidden'):
                        ref=HERE/'checkpoints'/name(task,model,'hidden',seed)
                        assert initial['shared_hidden_initialization_sha256']==read(ref/'INITIALIZATION.json')['shared_hidden_initialization_sha256']
                        assert read(cp/'TRAIN_ORDER.json')==read(ref/'TRAIN_ORDER.json')
                out={'task':task,'model':model,'arm':arm,'seed_count':5,'primary_seeds':values[arm],
                    'trainable_parameters':read(HERE/'checkpoints'/name(task,model,arm,SEEDS[0])/'INITIALIZATION.json')['trainable_parameters']}
                for metric in ['primary','text_micro_f1','json_valid','strict_json','schema_valid','native_eos_pct','capped_pct']:
                    out[metric]=float(np.mean([s[metric] for s in summaries])) if metric in summaries[0] else None
                tables.append(out)
            for control in ['hidden','hidden_budget']:
                deltas=np.array(values['hidden_both'])-np.array(values[control]);interval=seed_interval(deltas,8)
                if control=='hidden_budget':
                    for seed in SEEDS:
                        a=read(HERE/'checkpoints'/name(task,model,'hidden_both',seed)/'INITIALIZATION.json')
                        b=read(HERE/'checkpoints'/name(task,model,control,seed)/'INITIALIZATION.json')
                        assert a['trainable_parameters']==b['trainable_parameters']
                boot=[]
                for seed_index in range(5):
                    a=records['hidden_both'][seed_index];b=records[control][seed_index]
                    if task=='toolace':
                        d=np.array([int(x['content_correct'])-int(y['content_correct']) for x,y in zip(a,b)])
                        boot.append(100*(weights@d)/denoms)
                    else:
                        def scores(rs):
                            tp=np.array([r['span_tp'] for r in rs]);den=np.array([r['span_pred']+r['span_gold'] for r in rs])
                            return 200*(weights@tp)/(weights@den)
                        boot.append(scores(a)-scores(b))
                conditional=np.mean(boot,axis=0)
                interval.update(task=task,model=model,treatment='hidden_both',control=control,
                    paired_cluster_bootstrap_95=np.quantile(conditional,[.025,.975]).tolist(),clusters=len(unique),
                    bootstrap_scope='conditional on five fitted seeds, nominal 95%; not a substitute for seed variation')
                contrasts.append(interval)
    write(HERE/'HELDOUT_ANALYSIS.json',{'at':now(),'scope':'five-seed fixed-small-training-budget held-out confirmation','tables':tables,'primary_contrasts':contrasts,'all_pairing_and_budget_checks':'passed'})
    lines=['# 新后训练任务：五种子保留集评测','',
        '两项任务、两个官方预训练 Base、五个配对种子。每任务仍为 2,048 条训练，统一 LR2e-4、单轮64步；未重训或按测试结果选模型。以下为保留的710条工具调用及1,343条实体抽取 public-dev 结果。没有为各方法独立搜索学习率，不能据此宣称所有调优设置下的优势。','',
        '| 任务 | 模型 | 配置 | 主内容分 | 实体文本F1 | JSON可解析率 | 截断率 |','|---|---|---|---:|---:|---:|---:|']
    for r in tables:
        text='—' if r['text_micro_f1'] is None else f"{r['text_micro_f1']:.2f}"
        lines.append(f"| {r['task']} | {r['model']} | {r['arm']} | {r['primary']:.2f} | {text} | {r['json_valid']:.2f} | {r['capped_pct']:.2f} |")
    lines+=['','ToolACE 主分为完整调用集合正确率，包含无调用情形；不是 BFCL 官方得分。CLUENER 主分为实体类别与字符跨度的 micro F1，实体文本 F1 单列。所有格式失败和截断输出保留。','',
        '| 任务 | 模型 | 叠加组相对 | 增量 pp | 五种子同时95%区间（8比较） | 条件题目/簇95%区间 |','|---|---|---|---:|---|---|']
    for r in contrasts:
        ci=r['paired_cluster_bootstrap_95'];lines.append(f"| {r['task']} | {r['model']} | {r['control']} | {r['mean']:+.3f} | [{r['lower']:+.3f}, {r['upper']:+.3f}] | [{ci[0]:+.3f}, {ci[1]:+.3f}] |")
    lines+=['','所有 hidden 初始化和样本顺序已逐种子核验。叠加组和 hidden_budget 参数严格相等；原始模型参数冻结。JSON率提高本身不证明语义能力提高；实体位置误差、类别/文本误差及工具参数误差必须分别解释。','',
        '统计区间描述本轮保留集评测的变异，不能消除公共数据预训练暴露、小训练集或单一学习率的限制。主张稳定叠加收益需同时考虑普通 LoRA 和等参数对照，且不能用单个正向任务覆盖其他负结果。']
    (HERE/'HELDOUT_RESULTS_ZH.md').write_text('\n'.join(lines)+'\n');print(json.dumps(contrasts,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
