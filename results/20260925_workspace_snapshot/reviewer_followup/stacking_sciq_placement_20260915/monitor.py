"""Read-only progress summary; outside the frozen experimental computation."""
from collections import Counter
from common import *

def main():
    phase='confirmation' if (HERE/'CONFIRMATION_JOBS.json').exists() else 'tuning'
    jobs=read(HERE/f'{phase.upper()}_JOBS.json');states=Counter();running=[];scores=[]
    for spec in jobs:
        cp=Path(spec['checkpoint'])
        if (cp/'FAILED.json').exists():state='failed'
        elif (cp/'COMPLETE.json').exists():state='complete'
        elif (cp/'TRAINING.json').exists():state='evaluating'
        elif (cp/'PROGRESS.json').exists():state='training'
        elif cp.exists():state='loading'
        else:state='queued'
        states[state]+=1
        if state in ['training','loading','evaluating','failed']:
            progress=read(cp/'PROGRESS.json') if (cp/'PROGRESS.json').exists() else {}
            running.append({'name':spec['name'],'state':state,**progress})
        mf=cp/('test_metrics.json' if phase=='confirmation' else 'validation_metrics.json')
        if mf.exists():scores.append({'model':spec['model'],'arm':spec['arm'],'seed':spec['seed'],'lr':spec['lr'],**read(mf)['primary']})
    print(__import__('json').dumps({'at':now(),'phase':phase,'states':dict(states),'active':running,'scores':scores},indent=2))

if __name__=='__main__':main()
