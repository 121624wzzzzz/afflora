"""Descriptive post-fit diagnostics. Never changes primary scoring or inference."""
from collections import Counter
import numpy as np
from common import *
from scoring_ner import parse as parse_ner

ARMS=['base','hidden','hidden_budget','hidden_both']

def records(task, model, arm, seed):
    name=f'{task}_{model}_base' if arm=='base' else f'{task}_{model}_{arm}_s{seed}'
    return rows(HERE/'evaluations'/name/'test'/'responses.jsonl')

def ner_counts(rs):
    tp=sum(r['span_tp'] for r in rs);p=sum(r['span_pred'] for r in rs);g=sum(r['span_gold'] for r in rs)
    return {'n':len(rs),'tp':tp,'pred':p,'gold':g,'precision':100*tp/p if p else 0.,
            'recall':100*tp/g if g else 0.,'f1':200*tp/(p+g) if p+g else 100.}

def by_type(rs,data):
    counts={t:Counter() for t in TYPES};invalid=0
    for r,g in zip(rs,data):
        pred=Counter();value,_,_=parse_ner(r['text'])
        for e in value or []:
            typed=isinstance(e,dict) and isinstance(e.get('type'),str) and e['type'] in TYPES and isinstance(e.get('text'),str) and bool(e['text'])
            matches=[i for i in range(len(g['text'])) if g['text'].startswith(e['text'],i)] if typed else []
            good=typed and set(e)=={'type','text','occurrence'} and type(e['occurrence'])==int and 0<=e['occurrence']<len(matches)
            if good:
                start=matches[e['occurrence']];pred[(e['type'],start,start+len(e['text'])-1)]+=1
            else:invalid+=1
        gold=Counter((e['type'],e['start'],e['end']) for e in g['gold_spans'])
        for t in TYPES:
            counts[t].update({'tp':sum(v for k,v in (pred&gold).items() if k[0]==t),
                              'pred':sum(v for k,v in pred.items() if k[0]==t),
                              'gold':sum(v for k,v in gold.items() if k[0]==t)})
    overall=ner_counts(rs)
    assert sum(c['tp'] for c in counts.values())==overall['tp']
    assert sum(c['pred'] for c in counts.values())+invalid==overall['pred']
    assert sum(c['gold'] for c in counts.values())==overall['gold']
    return {t:dict(c,f1=200*c['tp']/(c['pred']+c['gold']) if c['pred']+c['gold'] else 100.) for t,c in counts.items()},invalid

def main():
    assert read(HERE/'RESULT_AUDIT.json')['status']=='passed'
    results=[];paired=[];panel=['# Fixed qualitative panel','',
        'Post-fit descriptive diagnostics only. For each new model/task, compare the first pre-specified seed of stack and budget LoRA. Select the first two exact-content wins and losses in fixed test order, if available. These are illustrative outcomes, not a random estimate of quality.','']
    for task in TASKS:
        data=rows(HERE/f'data/{task}_test.jsonl');seeds=list(range(TASK_SETTINGS[task]['seed_start'],TASK_SETTINGS[task]['seed_start']+5))
        for model in MODELS:
            for arm in ARMS:
                raw=[]
                for seed in (seeds[:1] if arm=='base' else seeds):
                    rs=records(task,model,arm,seed);assert [r['id'] for r in rs]==[g['id'] for g in data]
                    lengths=[r['length'] for r in rs]
                    row={'seed':seed,'mean_tokens':float(np.mean(lengths)),'p95_tokens':float(np.percentile(lengths,95)),
                         'native_eos_pct':100*np.mean([r['native_eos'] for r in rs]),'capped_pct':100*np.mean([r['capped'] for r in rs])}
                    if task=='cluener':
                        row['spans']=ner_counts(rs);row['per_type'],row['invalid_items']=by_type(rs,data)
                    raw.append(row)
                results.append({'task':task,'model':model,'arm':arm,'seeds':raw})
            for seed in seeds:
                stack=records(task,model,'hidden_both',seed)
                for arm in ARMS[1:3]:
                    control=records(task,model,arm,seed)
                    if task=='wikisql':
                        strata={}
                        for sv in [False,True]:
                            for cv in [False,True]:
                                idx=[i for i,(s,c) in enumerate(zip(stack,control)) if s['query_valid']==sv and c['query_valid']==cv]
                                strata[f'stack_valid={sv},control_valid={cv}']={'n':len(idx),
                                  'stack_correct':sum(stack[i]['content_correct'] for i in idx),
                                  'control_correct':sum(control[i]['content_correct'] for i in idx),
                                  'net_contribution_pp':100*sum(int(stack[i]['content_correct'])-int(control[i]['content_correct']) for i in idx)/len(data)}
                        assert sum(x['n'] for x in strata.values())==len(data)
                        delta=100*np.mean([int(s['content_correct'])-int(c['content_correct']) for s,c in zip(stack,control)])
                        assert abs(sum(x['net_contribution_pp'] for x in strata.values())-delta)<1e-9
                    else:
                        idx=[i for i,(s,c) in enumerate(zip(stack,control)) if s['schema_valid'] and c['schema_valid']]
                        strata={'both_schema_valid':{'n':len(idx),'stack':ner_counts([stack[i] for i in idx]),'control':ner_counts([control[i] for i in idx])}}
                    paired.append({'task':task,'model':model,'seed':seed,'control':arm,'strata':strata})
                if seed==seeds[0]:
                    control=records(task,model,'hidden_budget',seed)
                    for label,sv,cv in [('win',True,False),('loss',False,True)]:
                        idx=[i for i,(s,c) in enumerate(zip(stack,control)) if s['content_correct']==sv and c['content_correct']==cv][:2]
                        for i in idx:
                            g=data[i];panel += [f'## {task} / {model} / seed {seed} / {label} / {g["id"]}','',
                                'Input: '+(g['text'] if task=='cluener' else g['question']+'; columns: '+canonical(g['header'])),'',
                                'Gold: `'+canonical(g['target'])+'`','',
                                'Budget LoRA:','```text',control[i]['text'],'```','',
                                'LoRA + A-LoRA:','```text',stack[i]['text'],'```','']
    write(HERE/'CONTENT_DIAGNOSTICS.json',{'at':now(),'status':'passed','scope':'post-fit descriptive only; per-type scores exclude unassignable invalid items; overall primary retains them; NER valid-subset F1 is not additive or causal',
                                         'conditions':results,'paired_validity':paired})
    (HERE/'QUALITATIVE_PANEL.md').write_text('\n'.join(panel))

if __name__=='__main__':main()
