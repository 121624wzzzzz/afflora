"""Descriptive paired summaries after the full rank8 audit passes."""
from pathlib import Path
import sys,statistics,csv
R=Path(__file__).resolve().parent;S=R/'study';sys.path.insert(0,str(S))
from common import *
assert read(S/'FINAL_AUDIT.json')['status']=='passed'
shared=read(S/'RESULTS.json')['groups'];untied=read(S/'UNTIED_RESULTS.json')['groups'];bud=read(S/'BUDGET_PLAN.json');models=read(S/'models.json')
def summary(ds):return dict(mean=statistics.mean(ds),seed_deltas=ds,positive=sum(d>0 for d in ds),ties=sum(d==0 for d in ds),negative=sum(d<0 for d in ds),sample_sd=statistics.stdev(ds))
rows_out=[];contrasts=[];lines=['# Rank8 共享与 untied 单侧/双侧消融','', '48次正式训练及16次短训练全部完成并通过审计。两个任务均沿用固定数据和三个配对种子。以下为三种子描述性结果，不将均值接近解释为统计等效。','', '## Tied：共享 rank8 /16 /32','', '|模型|任务|H|独立双侧r16|共享r8|共享r16|共享r32|','|---|---|---:|---:|---:|---:|---:|']
for g in shared:
 p=g['paired_seeds'];assert len(p)==3
 vals=[statistics.mean(x['controls']['hidden'] for x in p),statistics.mean(x['controls']['hidden_both'] for x in p),statistics.mean(x['shared8'] for x in p),statistics.mean(x['prior_shared']['16'] for x in p),statistics.mean(x['prior_shared']['32'] for x in p)]
 lines.append('|'+g['model']+'|'+g['task']+'|'+'|'.join(f'{v:.3f}' for v in vals)+'|')
 for x in p:rows_out.append(dict(model=g['model'],task=g['task'],arm='hidden_shared',rank=8,seed=x['seed'],primary=x['shared8']))
 contrasts.append(dict(model=g['model'],task=g['task'],arm='hidden_shared8',comparisons={k:summary([x['deltas'][k] for x in p]) for k in ['hidden','hidden_budget','hidden_both','shared16','shared32']}))
lines+=['','共享边界额外参数分别为17d、33d、65d，独立双侧r16为65d；hidden LoRA参数不变。旧hidden_budget按65d设置，不是共享r8的等预算对照。共享仅用于tied模型。','', '## Untied：独立单侧/双侧 rank8 /16','', '|模型|任务|H|H+E8|H+U8|H+EU8|H+E16|H+U16|H+EU16|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
for g in untied:
 p=g['paired_seeds'];assert len(p)==3;arms=['hidden_input','hidden_output','hidden_both']
 vals=[statistics.mean(x['prior_rank16_and_H']['hidden'] for x in p)]+[statistics.mean(x['rank8'][a] for x in p) for a in arms]+[statistics.mean(x['prior_rank16_and_H'][a] for x in p) for a in arms]
 lines.append('|'+g['model']+'|'+g['task']+'|'+'|'.join(f'{v:.3f}' for v in vals)+'|')
 for a in arms:
  for x in p:rows_out.append(dict(model=g['model'],task=g['task'],arm=a,rank=8,seed=x['seed'],primary=x['rank8'][a]))
  comparisons={'H':summary([x['deltas_vs_H'][a] for x in p]),'same_arm_rank16':summary([x['deltas_vs_same_arm_rank16'][a] for x in p])}
  if a=='hidden_both':
   comparisons['equal_budget_E16']=summary([x['rank8'][a]-x['prior_rank16_and_H']['hidden_input'] for x in p]);comparisons['U16_smaller_budget_by_d']=summary([x['rank8'][a]-x['prior_rank16_and_H']['hidden_output'] for x in p]);comparisons['E8']=summary([x['rank8'][a]-x['rank8']['hidden_input'] for x in p]);comparisons['U8']=summary([x['rank8'][a]-x['rank8']['hidden_output'] for x in p])
  contrasts.append(dict(model=g['model'],task=g['task'],arm=a,comparisons=comparisons))
lines+=['','## 同预算：双侧各r8 对 E-only r16','', '两种设置的额外参数均为33d，hidden LoRA完全相同。该比较在首批正式结果前已记录于 PLANNED_CROSS_RANK_CONTRAST.md。它比较固定预算的分配方式，不单独隔离rank和优化几何的影响。','', '|模型|任务|EU8 − E16 均值（严格等参）|三个种子差值|正/平/负|EU8 − U16 均值（U16少d参数）|','|---|---|---:|---|---|---:|']
for c in contrasts:
 if 'equal_budget_E16' not in c['comparisons']:continue
 v=c['comparisons']['equal_budget_E16'];lines.append('|'+c['model']+'|'+c['task']+f"|{v['mean']:+.3f}|"+', '.join(f'{d:+.3f}' for d in v['seed_deltas'])+f"|{v['positive']}/{v['ties']}/{v['negative']}|{c['comparisons']['U16_smaller_budget_by_d']['mean']:+.3f}|")
lines+=['','单侧r8 E为17d、U为16d、双侧为33d；U16为32d，与双侧r8不严格等参。未通过无效参数补齐预算。所有负面种子和条件均保留。','', '原始结果与每个种子的对比见 PAIRED_CONTRASTS.json；训练、参数范围与输出审计见 study/FINAL_AUDIT.json。']
(R/'FINAL_INTERPRETATION_ZH.md').write_text('\n'.join(lines)+'\n');write(R/'PAIRED_CONTRASTS.json',contrasts)
assert len(rows_out)==48
with (R/'FORMAL_RESULTS.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows_out[0]));w.writeheader();w.writerows(rows_out)
print('\n'.join(lines))
