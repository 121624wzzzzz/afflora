import math
import numpy as np
from scipy.stats import t
from common import *

SEEDS=list(range(7100,7105));ARMS=['hidden','hidden_budget','hidden_both']
def seed_interval(a,family):
    a=np.asarray(a);mean=float(a.mean());half=float(t.ppf(1-.05/(2*family),4)*a.std(ddof=1)/math.sqrt(5))
    return {'mean':mean,'lower':mean-half,'upper':mean+half,'seed_deltas':a.tolist(),'family_size':family}

def main():
    assert read(HERE/'RESULT_AUDIT.json')['status']=='passed'
    for split in ['dev','test']:
        tables=[];contrasts=[]
        for task in TASKS:
            data=rows(HERE/f'data/{task}_{split}.jsonl');labels=[r['table_id'] if task=='wikisql' else normalized_premise(r) for r in data]
            unique=sorted(set(labels));idx={v:i for i,v in enumerate(unique)};group=np.array([idx[x] for x in labels]);sizes=np.bincount(group)
            rng=np.random.default_rng(20260917);weights=rng.multinomial(len(unique),np.ones(len(unique))/len(unique),size=10000)
            denom=weights@sizes
            for model in MODELS:
                baseline=read(HERE/'evaluations'/f'{task}_{model}_base'/split/'SUMMARY.json')
                tables.append({'task':task,'model':model,'arm':'base','primary':baseline['primary'],'seed_count':0,'trainable_parameters':0})
                values={};records={}
                for arm in ARMS:
                    paths=[HERE/'evaluations'/f'{task}_{model}_{arm}_s{s}'/split for s in SEEDS]
                    summaries=[read(p/'SUMMARY.json') for p in paths];records[arm]=[rows(p/'responses.jsonl') for p in paths]
                    values[arm]=[r['primary'] for r in summaries]
                    metrics=['primary','lf_correct_pct','query_valid_pct','strict_json_pct','capped_pct','distinct_parameter_execution_correct_pct',
                             'duplicate_condition_column_pct','unrestricted_correct_pct','unrestricted_valid_pct','mean_label_probability_mass']
                    out={'task':task,'model':model,'arm':arm,'seed_count':5,'primary_seeds':values[arm],
                        'trainable_parameters':read(HERE/'checkpoints'/f'{task}_{model}_{arm}_s7100'/'INITIALIZATION.json')['trainable_parameters']}
                    out.update({k:float(np.mean([s[k] for s in summaries])) for k in metrics if k in summaries[0]});tables.append(out)
                for control in ['hidden','hidden_budget']:
                    interval=seed_interval(np.array(values['hidden_both'])-np.array(values[control]),8)
                    item_delta=np.mean([[int(a['content_correct'])-int(b['content_correct']) for a,b in zip(ra,rb)] for ra,rb in zip(records['hidden_both'],records[control])],axis=0)
                    cluster_delta=np.bincount(group,weights=item_delta,minlength=len(unique));boots=100*(weights@cluster_delta)/denom
                    interval.update(task=task,model=model,control=control,conditional_cluster_bootstrap_95=np.quantile(boots,[.025,.975]).tolist(),clusters=len(unique))
                    contrasts.append(interval)
        write(HERE/f'{split.upper()}_ANALYSIS.json',{'at':now(),'tables':tables,'primary_contrasts':contrasts,'scope':'fixed common LR and training budget; conditional bootstrap does not replace seed variability'})
        lines=[f'# {split}：下游迁移五种子结果','',
            'ANLI R1 为固定三标签条件概率准确率；WikiSQL 为官方执行正确率，测试仅固定抽样1024条。两项任务均2048训练样本、共同LR2e-4、单轮64步、固定末步。所有任务保留，无按效果筛选。','',
            '|任务|模型|配置|可训练参数|主指标|','|---|---|---|---:|---:|']
        for r in tables:lines.append(f"|{r['task']}|{r['model']}|{r['arm']}|{r['trainable_parameters']:,}|{r['primary']:.2f}|")
        lines+=['','|任务|模型|叠加组相对|平均增量 pp|校正八项比较的95%种子区间|','|---|---|---|---:|---|']
        for r in contrasts:lines.append(f"|{r['task']}|{r['model']}|{r['control']}|{r['mean']:+.3f}|[{r['lower']:+.3f}, {r['upper']:+.3f}]|")
        lines+=['','区间依赖五种子 t 假设；公共数据预训练暴露未知，两个同系列模型、共同学习率和小训练集限制外推。JSON合法率和逻辑形式等诊断见对应 ANALYSIS.json。不得将不显著解释为证明无效。']
        (HERE/f'{split.upper()}_RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
    print(canonical(contrasts),flush=True)

def normalized_premise(row):return ' '.join(row['premise'].casefold().split())
if __name__=='__main__':main()
