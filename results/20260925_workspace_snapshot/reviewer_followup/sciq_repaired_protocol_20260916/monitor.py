"""Read-only progress; retained preflight failures are distinguished from new failures."""
from settings import *


def main():
    tests=list((HERE/'evaluations').glob('*/test_metrics.json'))
    validations=list((HERE/'evaluations').glob('*/validation_metrics.json'))
    active=[]
    for file in (HERE/'checkpoints').glob('*/PROGRESS.json'):
        r=read(file)
        if not (file.parent/'TRAINING.json').exists():active.append({'run':file.parent.name,'step':r['step'],'total':r['steps']})
    failures=[file.name for file in (HERE/'failures').glob('*.json') if not file.name.startswith('smoke_')]
    print(json.dumps({'at':now(),'test_evaluations_done':len(tests),'test_evaluations_expected':64,
        'validation_done':len(validations),'validation_expected':6,'selection_completed':(HERE/'SELECTION.json').exists(),
        'active_training':active,'formal_failures':failures,'sealed':(HERE/'ARTIFACT_MANIFEST.json').exists()},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
