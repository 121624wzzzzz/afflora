"""Aggregate all endpoints; paired seed and document-cluster uncertainty."""
import argparse
import csv
import io
import math
import numpy as np
from scipy.stats import t
from common import HERE, now, read, sha, write

def interval(values,alpha):
    a=np.asarray(values,dtype=float);m=float(a.mean());se=float(a.std(ddof=1)/np.sqrt(len(a)))
    radius=float(t.ppf(1-alpha/2,len(a)-1))*se
    return [m-radius,m+radius]

def boot(cluster_ids,delta,seed):
    unique=sorted(set(cluster_ids));lookup={v:i for i,v in enumerate(unique)}
    indices=np.array([lookup[c] for c in cluster_ids]);n=len(unique)
    sums=np.bincount(indices,weights=delta,minlength=n);counts=np.bincount(indices,minlength=n)
    rng=np.random.default_rng(seed);values=[]
    for start in range(0,10000,128):
        weights=rng.multinomial(n,np.full(n,1/n),size=min(128,10000-start))
        values.extend(((weights@sums)/(weights@counts)).tolist())
    return {'nominal95':np.quantile(values,[.025,.975]).tolist(),
            'bonferroni8':np.quantile(values,[.05/16,1-.05/16]).tolist(),'clusters':n}

def main():
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args()
    manifest=read(HERE/'manifest.json');endpoints=manifest['endpoints'];sources={str(HERE/'manifest.json'):sha(HERE/'manifest.json')}
    loaded={};records=[];missing=[]
    for j in endpoints:
        for task in ['cmrc','c3']:
            path=HERE/'outputs'/j['name']/f'{task}_scores.json'
            if not path.exists():missing.append(f"{j['name']}:{task}");continue
            report=read(path);sources[str(path)]=sha(path)
            for f,h in report['source_sha256'].items():assert sha(f)==h,f;sources[f]=h
            arm='reference' if j.get('reference') else j['placement']
            loaded[j['model'],arm,j['seed'],task]=report
            records.append({'model':j['model'],'arm':arm,'seed':j['seed'],'task':task,**report['metrics']})
    if not a.partial:assert not missing,missing
    contrasts=[]
    primary_complete=all((model,arm,seed,task) in loaded
        for model in {j['model'] for j in endpoints}
        for arm in ['none','both','hidden_budget'] for seed in [42,43,44] for task in ['cmrc','c3'])
    if primary_complete:
        for task,metric in [('cmrc','avg'),('c3','accuracy')]:
            for model in sorted({j['model'] for j in endpoints}):
                for control in ['none','hidden_budget']:
                    deltas=[];per_seed=[];clusters=None
                    for seed in [42,43,44]:
                        left=loaded[model,'both',seed,task]['per_example'];right=loaded[model,control,seed,task]['per_example']
                        assert [(x['id'],x['cluster']) for x in left]==[(x['id'],x['cluster']) for x in right]
                        if clusters is None:clusters=[x['cluster'] for x in left]
                        else:assert clusters==[x['cluster'] for x in left]
                        d=np.array([100*(x[metric]-y[metric]) for x,y in zip(left,right)])
                        deltas.append(d);per_seed.append({'seed':seed,'effect_pp':float(d.mean()),'better_questions':int((d>0).sum()),'worse_questions':int((d<0).sum())})
                    values=[x['effect_pp'] for x in per_seed]
                    ci=boot(clusters,np.mean(deltas,axis=0),20260915)
                    contrasts.append({'task':task,'metric':metric,'model':model,'control':control,
                        'mean_effect_pp':float(np.mean(values)),'per_seed':per_seed,
                        'seed_ci95':interval(values,.05),'seed_ci_bonferroni8':interval(values,.05/8),
                        'cluster_bootstrap_ci95':ci['nominal95'],'cluster_bootstrap_ci_bonferroni8':ci['bonferroni8'],
                        'clusters':ci['clusters']})
    report={'created_at':now(),'complete':not missing,'scored_endpoints_tasks':len(records),'expected_endpoints_tasks':52,
        'records':records,'primary_contrasts':contrasts,'missing':missing,
        'interval_scope':'Seed intervals: three training seeds, fixed data. Cluster bootstrap: conditional on these three fitted models, 10,000 document-cluster resamples. Bonferroni family size 8; intervals are not interchangeable.',
        'sources_sha256':sources}
    write(HERE/'RESULTS.json',report)
    lines=['# 中文任务迁移结果',f"更新：{now()}；完成 {len(records)}/52 项端点评测。",'',
        '每个已训练实验臂的表中均值基于当前已完成的种子；未满三个种子的行不得当作最终结果。none=hidden LoRA，reference=未微调参考。','',
        '| 模型 | 实验臂 | CMRC 种子数 | EM % | F1 % | AVG % | Answer CE | C3 种子数 | Acc % | Acc(sum) % |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for model in sorted({j['model'] for j in endpoints}):
        for arm in ['reference','none','output','both','hidden_budget']:
            cm=[x for x in records if x['model']==model and x['arm']==arm and x['task']=='cmrc']
            mc=[x for x in records if x['model']==model and x['arm']==arm and x['task']=='c3']
            def fmt(values,key,mult=100):return f'{np.mean([x[key] for x in values])*mult:.4f}' if values else '—'
            lines.append(f"| {model} | {arm} | {len(cm)} | {fmt(cm,'em')} | {fmt(cm,'f1')} | {fmt(cm,'avg')} | {fmt(cm,'answer_ce',1)} | {len(mc)} | {fmt(mc,'accuracy')} | {fmt(mc,'accuracy_sum')} |")
    if contrasts:
        lines += ['', '主要效应为 both 减对照，单位百分点；以下为名义 95% 区间，8 重比较校正区间另列于 JSON。', '',
                  '| 任务 | 模型 | 对照 | 平均增量 | 三种子区间 | 篇章簇区间 |','| --- | --- | --- | ---: | --- | --- |']
        for c in contrasts:
            fmtci=lambda x:f'[{x[0]:+.4f}, {x[1]:+.4f}]'
            lines.append(f"| {c['task']} | {c['model']} | {c['control']} | {c['mean_effect_pp']:+.4f} | {fmtci(c['seed_ci95'])} | {fmtci(c['cluster_bootstrap_ci95'])} |")
    lines += ['', 'C3 为条件似然选项评分，不能单独宣称自由生成改善。CMRC 为公开 dev 上的自回归答案评测，不是官方隐藏 test。所有结果来自已有中文 SFT adapters 的迁移，没有新任务训练。','']
    (HERE/'RESULTS.md').write_text('\n'.join(lines))
    if contrasts:
        stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=['task','model','control','seed','effect_pp','better_questions','worse_questions']);writer.writeheader()
        for c in contrasts:
            for x in c['per_seed']:writer.writerow({**{k:c[k] for k in ['task','model','control']},**x})
        (HERE/'paired_effects.csv').write_text(stream.getvalue())
    print(f'{len(records)}/52 endpoint-task reports;',len(contrasts),'primary contrasts',flush=True)

if __name__=='__main__':main()
