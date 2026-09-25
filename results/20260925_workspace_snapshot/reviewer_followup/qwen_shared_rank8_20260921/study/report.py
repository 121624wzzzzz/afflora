from common import *
import statistics

def main():
 old=read(HERE/'REUSE_AUDIT.json')['runs'];prior=read(HERE/'PRIOR_SHARED_AUDIT.json')['runs'];out=[];lines=['# Shared rank8 extension','', '|Model|Task|Seeds|H|Independent16/16|Shared8|Shared16|Shared32|','|---|---|---:|---:|---:|---:|---:|---:|']
 for m in sorted({j['model'] for j in read(HERE/'FORMAL_JOBS.json')}):
  for task in TASKS:
   pairs=[]
   for seed in range(TASK_SETTINGS[task]['seed_start'],TASK_SETTINGS[task]['seed_start']+3):
    n=f'{task}_{m}_hidden_shared8_s{seed}';a=HERE/'audits'/f'{n}.json'
    if not a.exists():continue
    a=read(a);p=HERE/'evaluations'/n/'test/SUMMARY.json';assert a['status']=='passed' and sha(p)==a['summary_sha256']['test'];value=read(p)['primary']
    ctrl={r['arm']:r['primary'] for r in old if (r['model'],r['task'],r['seed'])==(m,task,seed)}
    prev={str(r['rank']):r['primary'] for r in prior if (r['model'],r['task'],r['seed'])==(m,task,seed)};assert set(prev)=={'16','32'}
    pairs.append(dict(seed=seed,shared8=value,prior_shared=prev,controls=ctrl,deltas={**{k:value-v for k,v in ctrl.items()},**{'shared'+k:value-v for k,v in prev.items()}}))
   if pairs:
    vals=[statistics.mean(p['controls']['hidden'] for p in pairs),statistics.mean(p['controls']['hidden_both'] for p in pairs),statistics.mean(p['shared8'] for p in pairs),statistics.mean(p['prior_shared']['16'] for p in pairs),statistics.mean(p['prior_shared']['32'] for p in pairs)]
    lines.append('|'+m+'|'+task+'|'+str(len(pairs))+'|'+'|'.join(f'{v:.3f}' for v in vals)+'|');out.append(dict(model=m,task=task,paired_seeds=pairs))
 write(HERE/'RESULTS.json',dict(at=now(),groups=out));(HERE/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()

def untied_report():
 old=read(HERE/'REUSE_AUDIT.json')['runs'];groups=[];lines=['# Untied independent rank8 ablations','', '|Model|Task|Seeds|H|H+E r8|H+U r8|H+EU r8|H+E r16|H+U r16|H+EU r16|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 arms=['hidden_input','hidden_output','hidden_both']
 for m in ['qwen25_7b_base','qwen3_8b_base']:
  for task in TASKS:
   pairs=[]
   for seed in range(TASK_SETTINGS[task]['seed_start'],TASK_SETTINGS[task]['seed_start']+3):
    vals={}
    for arm in arms:
     n=f'{task}_{m}_{arm}_r8_s{seed}';p=HERE/'audits'/f'{n}.json'
     if not p.exists():continue
     a=read(p);p=HERE/'evaluations'/n/'test/SUMMARY.json';assert a['status']=='passed' and sha(p)==a['summary_sha256']['test'];vals[arm]=read(p)['primary']
    if len(vals)!=3:continue
    ctrl={r['arm']:r['primary'] for r in old if (r['model'],r['task'],r['seed'])==(m,task,seed)}
    pairs.append(dict(seed=seed,rank8=vals,prior_rank16_and_H=ctrl,deltas_vs_H={a:v-ctrl['hidden'] for a,v in vals.items()},deltas_vs_same_arm_rank16={a:v-ctrl[a] for a,v in vals.items()}))
   if pairs:
    vals=[statistics.mean(p['prior_rank16_and_H']['hidden'] for p in pairs)]+[statistics.mean(p['rank8'][a] for p in pairs) for a in arms]+[statistics.mean(p['prior_rank16_and_H'][a] for p in pairs) for a in arms]
    lines.append('|'+m+'|'+task+'|'+str(len(pairs))+'|'+'|'.join(f'{v:.3f}' for v in vals)+'|');groups.append(dict(model=m,task=task,paired_seeds=pairs))
 write(HERE/'UNTIED_RESULTS.json',dict(at=now(),groups=groups));(HERE/'UNTIED_RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':untied_report()
