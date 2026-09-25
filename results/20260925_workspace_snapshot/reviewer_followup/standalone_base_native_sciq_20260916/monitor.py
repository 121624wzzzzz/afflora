"""Read-only progress snapshot; no selection or experiment mutation."""
from settings import *
from collections import defaultdict

def main():
    cps=list((HERE/'checkpoints').glob('confirmation*'))
    summary={'at':now(),'pipeline':read(HERE/'PIPELINE_PROGRESS.json'),
        'confirmation_trained':sum((p/'TRAINING.json').exists() for p in cps),
        'confirmation_completed':sum((p/'COMPLETE.json').exists() for p in cps),'total':30,
        'failures':[str(p) for p in (HERE/'checkpoints').glob('*/FAILED.json')]}
    groups=defaultdict(list)
    for cp in cps:
        path=cp/'test_metrics.json'
        if path.exists():
            r=read(path);s=r['spec'];groups[(s['model'],s['arm'])].append({'seed':s['seed'],
                'accuracy':r['primary']['accuracy'],'strict_generation':r['generation']['strict_accuracy'],
                'valid_format':r['generation']['valid_answer_rate'],'length_cap':r['generation']['length_cap_rate']})
    summary['partial_results']={f'{m}/{a}':{'n':len(v),'mean_accuracy':sum(x['accuracy'] for x in v)/len(v),
        'runs':sorted(v,key=lambda x:x['seed'])} for (m,a),v in sorted(groups.items())}
    summary['active_progress']={cp.name:read(cp/'PROGRESS.json')['step'] for cp in cps if not (cp/'COMPLETE.json').exists() and (cp/'PROGRESS.json').exists()}
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
