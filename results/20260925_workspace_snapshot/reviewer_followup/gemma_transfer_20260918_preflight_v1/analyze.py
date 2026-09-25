"""Four prespecified paired stacking contrasts; seed and cluster uncertainty separately."""
import math,statistics
import numpy as np
from scipy.stats import t
from common import *
from design import CONFIRM_SEEDS,SEARCH_SEEDS
def diagnostics(name):
 h=read(HERE/'checkpoints'/name/'TRAINING.json')['history']
 return {'clipped_steps':sum(x['grad_norm']>1 for x in h),'tail8_training_loss':statistics.mean(x['loss'] for x in h[-8:]),
  'mean_group_grad_norm':{g:statistics.mean(x['group_grad_norms'][g] for x in h) for g in ['hidden','input','output']},
  'mean_group_gradient_energy_share':{g:statistics.mean(x['group_grad_norms'][g]**2/sum(v*v for v in x['group_grad_norms'].values()) for x in h) for g in ['hidden','input','output']},
  'sum_step_update_norms':{g:sum(x['group_update_norms'][g] for x in h) for g in ['hidden','input','output']}}
def main():
 assert read(HERE/'SCHEDULER_COMPLETE.json')['status']=='passed'
 sel=read(HERE/'SELECTION.json');results={};bases={};comparisons=[];bootstrap=[];partitions={}
 for ti,task in enumerate(TASKS):
  results[task]={};predictions={};data=rows(HERE/f'data/{task}_confirm.jsonl')
  for arm in ARMS:
   values=[];predictions[arm]=[]
   for seed in CONFIRM_SEEDS:
    name=sel['roles'][task][arm][str(seed)];audit=read(HERE/'audits'/f'{name}.json');assert audit['status']=='passed'
    p=HERE/'evaluations'/name/'confirm';s=read(p/'SUMMARY.json');assert sha(p/'SUMMARY.json')==audit['summary_sha256']['confirm'] and s['n']==len(data)
    rs=rows(p/'responses.jsonl');assert [r['id'] for r in rs]==[r['id'] for r in data];predictions[arm].append(rs)
    secondary=['lf_correct_pct','query_valid_pct','strict_json_pct','capped_pct','distinct_parameter_execution_correct_pct'] if task=='wikisql' else ['macro_f1_all50','coarse_accuracy']
    values.append({'seed':seed,'run':name,'primary':s['primary'],**{k:s[k] for k in secondary},'diagnostics':diagnostics(name)})
   results[task][arm]=values
  base=next(s for s in read(HERE/'BASE_JOBS.json') if s['task']==task);ap=HERE/'audits'/f'{base["name"]}.json';assert read(ap)['status']=='passed'
  p=HERE/'evaluations'/base['name']/'confirm/SUMMARY.json';assert sha(p)==read(ap)['summary_sha256']['confirm'];bases[task]=read(p)
  table_keys=[r['table_id'] if task=='wikisql' else r['cluster_id'] for r in data];clusters=sorted(set(table_keys));ix={v:i for i,v in enumerate(clusters)}
  assignment=np.array([ix[v] for v in table_keys]);counts=np.bincount(assignment,minlength=len(clusters));sums=[];task_comparisons=[]
  for control in ['hidden','hidden_budget']:
   a='hidden_both';b=control;d=[x['primary']-y['primary'] for x,y in zip(results[task][a],results[task][b])]
   mean=statistics.mean(d);sd=statistics.stdev(d);se=sd/math.sqrt(5)
   c={'task':task,'a':a,'b':b,'paired_seed_deltas_pp':d,'mean_pp':mean,'sd_pp':sd,'positive_seeds':sum(x>0 for x in d),
      'ci95':[mean-t.ppf(.975,4)*se,mean+t.ppf(.975,4)*se],
      'bonferroni4_ci95':[mean-t.ppf(1-.05/8,4)*se,mean+t.ppf(1-.05/8,4)*se]}
   comparisons.append(c);task_comparisons.append(c)
   delta=np.array([[int(x['content_correct'])-int(y['content_correct']) for x,y in zip(xs,ys)] for xs,ys in zip(predictions[a],predictions[b])]).mean(axis=0)
   assert abs(100*delta.mean()-mean)<1e-12;sums.append(np.bincount(assignment,weights=delta,minlength=len(clusters)))
   case=[]
   for seed,xs,ys in zip(CONFIRM_SEEDS,predictions[a],predictions[b]):
    wins=sum(x['content_correct'] and not y['content_correct'] for x,y in zip(xs,ys));losses=sum(y['content_correct'] and not x['content_correct'] for x,y in zip(xs,ys))
    entry={'seed':seed,'recovered':wins,'regressed':losses,'net_questions':wins-losses}
    if task=='wikisql':
     bv=sum(int(x['content_correct'])-int(y['content_correct']) for x,y in zip(xs,ys) if x['query_valid'] and y['query_valid'])
     entry.update(both_valid_net_pp=100*bv/len(data),other_validity_net_pp=100*(wins-losses-bv)/len(data))
    case.append(entry)
   partitions[task+' '+a+' minus '+b]=case
  rng=np.random.default_rng(20300918+ti);sums=np.array(sums).T;draws=[]
  for start in range(0,20000,500):
   weights=rng.multinomial(len(clusters),np.full(len(clusters),1/len(clusters)),size=500)
   draws.append(100*(weights@sums)/(weights@counts)[:,None])
  draws=np.concatenate(draws)
  for i,c in enumerate(task_comparisons):
   bootstrap.append({'task':task,'a':c['a'],'b':c['b'],'mean_pp':c['mean_pp'],'clusters':len(clusters),'replicates':20000,'seed':20300918+ti,
       'percentile_ci95':np.percentile(draws[:,i],[2.5,97.5]).tolist(),'bonferroni4_percentile_ci95':np.percentile(draws[:,i],[.625,99.375]).tolist()})
 report={'at':now(),'family':4,'n_seeds':5,'selection_sha256':sha(HERE/'SELECTION.json'),'results':results,'base':bases,'comparisons':comparisons,
  'limitations':['Finite common LR grid; two development seeds, five confirmation seeds; no globally optimal hyperparameter claim.',
   'Seed t intervals condition on fixed data and selected configurations and use a small-sample distributional assumption.',
   'Supplementary cluster bootstrap fixes the five fitted models; not joint seed, data and model-selection uncertainty.',
   'Shared public benchmark splits previously evaluated on other families; not globally unseen project data or proof against pretraining exposure.',
   'E has bias and U does not; tied frozen weights with independent effective E/U updates; no pure placement or preserved-tying claim.',
   'Source checkpoint is FP32; all arms use the same BF16-rounded frozen base, cast back to FP32 for evaluation.']}
 write(HERE/'ANALYSIS.json',report);write(HERE/'CLUSTER_BOOTSTRAP.json',{'status':'passed','family':4,'comparisons':bootstrap,'interpretation':'Paired table clusters for WikiSQL and question clusters for TREC50, conditional on the same five fitted models.'})
 write(HERE/'PAIRED_CASES.json',{'description':'Descriptive recovered/regressed cases; validity partitions use the fixed full-set denominator and are not causal decompositions.','comparisons':partitions})
 lines=['# Gemma-2-9B Base cross-family stacking study','','## Development search','','| Task | Arm | LR | Seed 7400 | Seed 7401 | Mean | Selected |','|---|---|---:|---:|---:|---:|---|']
 for task in TASKS:
  for arm,rs in sel['scores'][task].items():
   for r in rs:lines.append(f'| {task} | {arm} | {r["lr"]:g} | {r["seed_scores"][0]:.4f} | {r["seed_scores"][1]:.4f} | {r["mean"]:.4f} | {r["id"]==sel["selected"][task][arm]["id"]} |')
 lines+=['','## Confirmation','','| Task | Arm | 7500 | 7501 | 7502 | 7503 | 7504 | Mean |','|---|---|---:|---:|---:|---:|---:|---:|']
 for task in TASKS:
  lines.append(f'| {task} | base | — | — | — | — | — | {bases[task]["primary"]:.4f} |')
  for arm,rs in results[task].items():lines.append('| '+task+' | '+arm+' | '+' | '.join(f'{x["primary"]:.4f}' for x in rs)+f' | {statistics.mean(x["primary"] for x in rs):.4f} |')
 lines+=['','| Task | Contrast | Mean pp | Marginal seed 95% CI | Family-4 seed 95% CI | Positive seeds |','|---|---|---:|---|---|---:|']
 for c in comparisons:lines.append(f'| {c["task"]} | {c["a"]} − {c["b"]} | {c["mean_pp"]:+.4f} | [{c["ci95"][0]:+.4f},{c["ci95"][1]:+.4f}] | [{c["bonferroni4_ci95"][0]:+.4f},{c["bonferroni4_ci95"][1]:+.4f}] | {c["positive_seeds"]}/5 |')
 lines+=['','## Limitations','']+['- '+x for x in report['limitations']]
 (HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(canonical({'comparisons':comparisons,'bootstrap':bootstrap}),flush=True)
if __name__=='__main__':main()
