from common import *
import statistics

def main():
 old=read(HERE/'REUSE_AUDIT.json')['runs'];out=[];lines=['# 共享 E/U 阶段结果','', '仅计入独立审计通过的正式训练；三个种子全齐前不报告为完整组。','', '|模型|任务|种子数|共享 r16|共享 r32|独立 E/U r16|H|','|---|---|---:|---:|---:|---:|---:|']
 for m in sorted({j['model'] for j in read(HERE/'FORMAL_JOBS.json')}):
  for task in TASKS:
   pairs=[]
   for seed in range(TASK_SETTINGS[task]['seed_start'],TASK_SETTINGS[task]['seed_start']+3):
    vals={}
    for r in [16,32]:
     n=f'{task}_{m}_hidden_shared{r}_s{seed}'
     if (HERE/'audits'/f'{n}.json').exists():
      a=read(HERE/'audits'/f'{n}.json');p=HERE/'evaluations'/n/'test/SUMMARY.json';assert a['status']=='passed' and sha(p)==a['summary_sha256']['test'];vals[str(r)]=read(p)['primary']
    if len(vals)!=2:continue
    controls={z['arm']:z['primary'] for z in old if (z['model'],z['task'],z['seed'])==(m,task,seed)}
    pairs.append(dict(seed=seed,shared=vals,controls=controls,deltas={r:{k:v-c for k,c in controls.items()} for r,v in vals.items()}))
   if pairs:
    means={r:statistics.mean(p['shared'][r] for p in pairs) for r in ['16','32']};ctrl={k:statistics.mean(p['controls'][k] for p in pairs) for k in ['hidden_both','hidden']}
    lines.append(f"|{m}|{task}|{len(pairs)}|{means['16']:.3f}|{means['32']:.3f}|{ctrl['hidden_both']:.3f}|{ctrl['hidden']:.3f}|")
    out.append(dict(model=m,task=task,paired_seeds=pairs,means=means,controls=ctrl))
 write(HERE/'RESULTS.json',dict(at=now(),groups=out));(HERE/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
