"""Freeze development-only choices after every search fit/audit passes."""
from common import *
from design import *
def main():
 assert not (HERE/'SELECTION.json').exists() and not (HERE/'CONFIRMATION_JOBS.json').exists()
 assert read(HERE/'PREFLIGHT_GATE.json')['status']=='passed'
 scores={};selected={};inputs={};jobs=[];roles={}
 for task in TASKS:
  scores[task]={};selected[task]={};roles[task]={}
  for arm in ARMS:
   values=[]
   for i,c in enumerate(CANDIDATES):
    seed_scores=[]
    for seed in SEARCH_SEEDS:
     s=spec('search',task,arm,i,seed);ap=HERE/'audits'/f'{s["name"]}.json';audit=read(ap);assert audit['status']=='passed'
     p=HERE/'evaluations'/s['name']/'dev/SUMMARY.json';v=read(p);assert v['spec']==dict(s,eval_split='dev') and v['n']==len(rows(HERE/f'data/{task}_dev.jsonl'))
     assert audit['summary_sha256']['dev']==sha(p);inputs[str(ap.relative_to(HERE))]=sha(ap);inputs[str(p.relative_to(HERE))]=sha(p);seed_scores.append(v['primary'])
    values.append(dict(c,seed_scores=seed_scores,mean=sum(seed_scores)/2))
   winner=max(range(6),key=lambda i:(values[i]['mean'],-i));scores[task][arm]=values;selected[task][arm]=CANDIDATES[winner]
   roles[task][arm]={}
   for seed in CONFIRM_SEEDS:
    s=spec('confirmation',task,arm,winner,seed);roles[task][arm][str(seed)]=s['name'];jobs.append(s)
 assert len(jobs)==30;write(HERE/'CONFIRMATION_JOBS.json',jobs)
 write(HERE/'SELECTION.json',{'at':now(),'status':'frozen','scores':scores,'selected':selected,'roles':roles,'inputs':inputs,
       'protocol_sha256':sha(HERE/'PROTOCOL.md'),'code_frozen_sha256':sha(HERE/'CODE_FROZEN.json'),'data_frozen_sha256':sha(HERE/'DATA_FROZEN.json'),
       'confirmation_jobs_sha256':sha(HERE/'CONFIRMATION_JOBS.json')})
 print(canonical({'selected':selected,'confirmation_fits':len(jobs)}),flush=True)
if __name__=='__main__':main()
