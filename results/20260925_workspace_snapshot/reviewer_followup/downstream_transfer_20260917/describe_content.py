"""Post hoc descriptive diagnostics, not additional hypothesis tests."""
import numpy as np
from common import *

out={'at':now(),'scope':'descriptive only; no task or checkpoint selection','groups':[]}
for task in TASKS:
    gold=rows(HERE/f'data/{task}_test.jsonl')
    for model in MODELS:
        for arm in ['base','hidden','hidden_budget','hidden_both']:
            names=[f'{task}_{model}_base'] if arm=='base' else [f'{task}_{model}_{arm}_s{s}' for s in range(7100,7105)]
            runs=[rows(HERE/'evaluations'/n/'test/responses.jsonl') for n in names]
            row={'task':task,'model':model,'arm':arm,'seed_count':len(runs)}
            if task=='anli_r1':
                matrices=[]
                for records in runs:
                    a=np.zeros((3,3),dtype=int)
                    for r,g in zip(records,gold):
                        assert r['id']==g['id'];a['ABC'.index(g['target']),'ABC'.index(r['predicted_label'])]+=1
                    matrices.append(a)
                avg=np.mean(matrices,axis=0);row.update(confusion_gold_rows_prediction_columns=avg.tolist(),
                    class_order=['entailment','neutral','contradiction'],per_class_recall_pct=(100*np.diag(avg)/avg.sum(axis=1)).tolist(),
                    predicted_label_counts=avg.sum(axis=0).tolist())
            else:
                def avg(k):return 100*float(np.mean([np.mean([r[k] for r in records]) for records in runs]))
                row.update({k+'_pct':avg(k) for k in ['content_correct','lf_correct','query_valid','strict_json','execution_ok','empty_prediction','duplicate_condition_column','distinct_parameter_execution_correct','capped']})
                row['by_gold_aggregation']={str(a):{'n':sum(g['target']['agg']==a for g in gold),
                    'accuracy':100*float(np.mean([np.mean([r['content_correct'] for r,g in zip(records,gold) if g['target']['agg']==a]) for records in runs]))} for a in range(6)}
            out['groups'].append(row)
out['sql_pairwise_decomposition']=[]
for model in MODELS:
    for control in ['hidden','hidden_budget']:
        seeds=[]
        for seed in range(7100,7105):
            a=rows(HERE/'evaluations'/f'wikisql_{model}_hidden_both_s{seed}'/'test/responses.jsonl')
            b=rows(HERE/'evaluations'/f'wikisql_{model}_{control}_s{seed}'/'test/responses.jsonl')
            parts={'both_query_valid':0,'only_treatment_valid':0,'only_control_valid':0,'neither_valid':0};counts=dict(parts)
            for x,y in zip(a,b):
                assert x['id']==y['id']
                k='both_query_valid' if x['query_valid'] and y['query_valid'] else ('only_treatment_valid' if x['query_valid'] else ('only_control_valid' if y['query_valid'] else 'neither_valid'))
                counts[k]+=1;parts[k]+=int(x['content_correct'])-int(y['content_correct'])
            seeds.append({'seed':seed,'counts':counts,'contribution_pp':{k:100*v/len(a) for k,v in parts.items()}})
        out['sql_pairwise_decomposition'].append({'model':model,'control':control,'seeds':seeds,
            'mean_contribution_pp':{k:float(np.mean([s['contribution_pp'][k] for s in seeds])) for k in parts},
            'mean_counts':{k:float(np.mean([s['counts'][k] for s in seeds])) for k in counts}})
write(HERE/'CONTENT_DIAGNOSTICS.json',out)
print('Wrote',len(out['groups']),'descriptive groups')
