import argparse
import ast
import csv
import io
import numpy as np
from scipy.stats import t
from common import HERE, now, read, sha, write
from experiment import MODELS

TRANSFER=HERE.parent/'stacking_chinese_transfer_20260915'

def interval(values,alpha):
    a=np.asarray(values,dtype=float);m=float(a.mean());r=float(t.ppf(1-alpha/2,2)*a.std(ddof=1)/np.sqrt(3))
    return [m-r,m+r]

def bootstrap(clusters,delta):
    unique=sorted(set(clusters));lookup={v:i for i,v in enumerate(unique)};indices=[lookup[v] for v in clusters];n=len(unique)
    sums=np.bincount(indices,weights=delta,minlength=n);counts=np.bincount(indices,minlength=n);rng=np.random.default_rng(20260915);values=[]
    for start in range(0,10000,128):
        weights=rng.multinomial(n,np.full(n,1/n),size=min(128,10000-start));values.extend(((weights@sums)/(weights@counts)).tolist())
    return np.quantile(values,[.025,.975]).tolist(),np.quantile(values,[.05/8,1-.05/8]).tolist()

def reference_reuse():
    audit=read(TRANSFER/'FINAL_AUDIT.json');assert audit['status']=='passed'
    assert sha(HERE/'data/cmrc_eval.jsonl')==sha(TRANSFER/'data/cmrc_eval.jsonl')
    assert sha(HERE/'cmrc_official_py3.py')==sha(TRANSFER/'cmrc_official_py3.py')
    checks={}
    for name,functions in [('evaluate.py',['generation','batches','likelihood','scoring_items','score_batch','task_summary']),('common.py',['prompt_text','prompt_ids'])]:
        trees=[]
        for root in [HERE,TRANSFER]:
            tree=ast.parse((root/name).read_text());trees.append({n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)})
        for f in functions:assert trees[0][f]==trees[1][f],f
        checks[name]={'current_sha256':sha(HERE/name),'reference_sha256':sha(TRANSFER/name),'identical_function_asts':functions}
    result=[];sources={}
    for model in MODELS:
        path=TRANSFER/'outputs'/f'{model}_none_hr0_base/cmrc_scores.json'
        assert sha(path)==audit['input_and_output_sources_sha256'][str(path)]
        scores=read(path);sources[str(path)]=sha(path)
        for p,h in scores['source_sha256'].items():assert sha(p)==h;sources[p]=h
        result.append({'model':model,'arm':'reference','seed':None,**scores['metrics']})
    write(HERE/'REFERENCE_REUSE_AUDIT.json',{'status':'passed','checked_at':now(),'identity_checks':checks,'source_sha256':sources,
        'scope':'Reuse unadapted model outputs with identical data, prompt, token-scoring, generation and scoring function ASTs. No trained adapter reused.'})
    return result,sources

def main():
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args()
    m=read(HERE/'manifest.json');records,sources=reference_reuse();loaded={};missing=[]
    sources[str(HERE/'manifest.json')]=sha(HERE/'manifest.json')
    for j in m['matrix']:
        folder=HERE/'outputs'/j['name'];path=folder/'cmrc_scores.json'
        if not path.exists():missing.append(j['name']);continue
        r=read(path);sources[str(path)]=sha(path)
        for p,h in r['source_sha256'].items():assert sha(p)==h,p;sources[p]=h
        dev=read(folder/'internal_dev_ce.json');sources[str(folder/'internal_dev_ce.json')]=sha(folder/'internal_dev_ce.json')
        loaded[j['model'],j['arm'],j['seed']]=r
        train=read(__import__('pathlib').Path(j['checkpoint'])/'train_results.json')
        records.append({'model':j['model'],'arm':j['arm'],'seed':j['seed'],**r['metrics'],
            'internal_dev_ce':dev['avg_ce'],'training_seconds':train['train_runtime'],'training_steps':train['global_step']})
    if not a.partial:assert not missing,missing
    contrasts=[];auxiliary=[]
    if all((model,arm,seed) in loaded for model in MODELS for arm in ['none','both','hidden_budget'] for seed in [42,43,44]):
        for model in MODELS:
            for control in ['none','hidden_budget']:
                deltas=[];per_seed=[];clusters=None
                for seed in [42,43,44]:
                    left=loaded[model,'both',seed]['per_example'];right=loaded[model,control,seed]['per_example']
                    assert [(r['id'],r['cluster']) for r in left]==[(r['id'],r['cluster']) for r in right]
                    current=[r['cluster'] for r in left]
                    if clusters is None:clusters=current
                    else:assert clusters==current
                    d=np.array([100*(x['avg']-y['avg']) for x,y in zip(left,right)]);deltas.append(d)
                    per_seed.append({'seed':seed,'effect_pp':float(d.mean()),'better_questions':int((d>0).sum()),'worse_questions':int((d<0).sum())})
                    auxiliary.append({'model':model,'control':control,'seed':seed,
                        **{k:float(np.mean([x[k]-y[k] for x,y in zip(left,right)])) for k in ['em','f1','answer_ce','answer_top1','hit_token_cap','generated_tokens']}})
                values=[x['effect_pp'] for x in per_seed];ci,bf=bootstrap(clusters,np.mean(deltas,axis=0))
                contrasts.append({'model':model,'control':control,'mean_effect_pp':float(np.mean(values)),
                    'per_seed':per_seed,'seed_ci95':interval(values,.05),'seed_ci_bonferroni4':interval(values,.05/4),
                    'cluster_bootstrap_ci95':ci,'cluster_bootstrap_ci_bonferroni4':bf,'clusters':len(set(clusters))})
    result={'updated_at':now(),'complete':not missing,'trained_endpoints_completed':24-len(missing),'expected_trained_endpoints':24,
        'records':records,'primary_contrasts':contrasts,'auxiliary_contrasts':auxiliary,'missing':missing,'sources_sha256':sources,
        'scope':'Task-specific fresh CMRC SFT; public dev was previously evaluated for transfer. Seed and document-cluster intervals quantify different uncertainty; bootstrap conditional on fitted models. Four primary comparisons.'}
    write(HERE/'RESULTS.json',result)
    lines=['# CMRC 任务内训练结果',f"更新 {now()}；完成 {24-len(missing)}/24 个训练端点评测。",'',
        '不同完成种子数的均值仅用于进度查看；最终推断须等待配对三种子。reference 从上一轮核验后复用。', '',
        '| 模型 | 实验臂 | 种子数 | EM % | F1 % | AVG % | Answer CE | Answer top-1 % | Internal dev CE |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for model in MODELS:
        for arm in ['reference','none','output','both','hidden_budget']:
            rs=[r for r in records if r['model']==model and r['arm']==arm]
            def fmt(k,mult=100):return f'{np.mean([r[k] for r in rs])*mult:.5f}' if rs and all(k in r for r in rs) else '—'
            lines.append(f"| {model} | {arm} | {len(rs)} | {fmt('em')} | {fmt('f1')} | {fmt('avg')} | {fmt('answer_ce',1)} | {fmt('answer_top1')} | {fmt('internal_dev_ce',1)} |")
    if contrasts:
        lines+=['','主要增量为 both 减对照，百分点。下表为名义 95% 区间，4 重比较区间保存在 JSON。','',
            '| 模型 | 对照 | 增量 | 三种子区间 | 篇章簇区间 |','| --- | --- | ---: | --- | --- |']
        for c in contrasts:
            ci=lambda x:f'[{x[0]:+.5f}, {x[1]:+.5f}]'
            lines.append(f"| {c['model']} | {c['control']} | {c['mean_effect_pp']:+.5f} | {ci(c['seed_ci95'])} | {ci(c['cluster_bootstrap_ci95'])} |")
    (HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    if contrasts:
        stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=['model','control','seed','effect_pp','better_questions','worse_questions']);writer.writeheader()
        for c in contrasts:
            for r in c['per_seed']:writer.writerow({**{k:c[k] for k in ['model','control']},**r})
        (HERE/'paired_effects.csv').write_text(stream.getvalue())
    print(24-len(missing),'/24 trained endpoints;',len(contrasts),'primary contrasts',flush=True)

if __name__=='__main__':main()
