"""Read only development results; freeze one configuration for each arm."""
from common import *
from design import *

def main():
    assert not (HERE/'SELECTION.json').exists() and not (HERE/'CONFIRMATION_JOBS.json').exists()
    assert read(HERE/'COMPATIBILITY.json')['status']=='passed'
    scores={};selected={};inputs={}
    for arm,cs in CANDIDATES.items():
        values=[]
        for i,c in enumerate(cs):
            seed_scores=[]
            for seed in SEARCH_SEEDS:
                s=spec('search',arm,i,seed)
                a=HERE/'audits'/f'{s["name"]}.json';assert read(a)['status']=='passed';inputs[str(a.relative_to(HERE))]=sha(a)
                p=HERE/'evaluations'/s['name']/'dev/SUMMARY.json';v=read(p)
                assert v['spec']==dict(s,eval_split='dev') and v['n']==1024
                assert read(a)['summary_sha256']['dev']==sha(p);inputs[str(p.relative_to(HERE))]=sha(p)
                seed_scores.append(v['primary'])
            values.append(dict(c,seed_scores=seed_scores,mean=sum(seed_scores)/len(seed_scores)))
        winner=max(range(len(values)),key=lambda i:(values[i]['mean'],-i));scores[arm]=values;selected[arm]=winner
    jobs=[];roles={}
    configs={'tuned_hidden':('hidden',selected['hidden']),'tuned_budget':('hidden_budget',selected['hidden_budget']),
             'tuned_heu':('hidden_both',selected['hidden_both']),'anchor_budget':('hidden_budget',0),'anchor_heu':('hidden_both',0)}
    for role,(arm,c) in configs.items():
        roles[role]={}
        for seed in CONFIRM_SEEDS:
            s=spec('confirmation',arm,c,seed);roles[role][str(seed)]=s['name']
            if s not in jobs:jobs.append(s)
    write(HERE/'CONFIRMATION_JOBS.json',jobs)
    write(HERE/'SELECTION.json',{'at':now(),'status':'frozen','protocol_sha256':sha(HERE/'PROTOCOL.md'),
          'code_frozen_sha256':sha(HERE/'CODE_FROZEN.json'),'data_frozen_sha256':sha(HERE/'DATA_FROZEN.json'),
          'scores':scores,'selected':{a:CANDIDATES[a][i] for a,i in selected.items()},'roles':roles,
          'inputs':inputs,'confirmation_jobs_sha256':sha(HERE/'CONFIRMATION_JOBS.json')})
    print(canonical({'selected':{a:CANDIDATES[a][i] for a,i in selected.items()},'confirmation_fits':len(jobs)}),flush=True)
if __name__=='__main__':main()
