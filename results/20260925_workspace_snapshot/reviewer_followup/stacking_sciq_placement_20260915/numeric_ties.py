"""Post hoc numeric diagnostic; never changes frozen primary predictions."""
import argparse,json
import numpy as np
import torch
from common import *
from modeling import load_checkpoint,batch,forward_selected
from run import specs

def one(cp):
    rr=rows(cp/'test_predictions.jsonl');flagged=[];smallest=float('inf')
    for x in rr:
        scores=sorted(x['label_logits']);gap=scores[-1]-scores[-2];smallest=min(smallest,gap)
        if gap<=.001:flagged.append(x)
    result={'checkpoint':str(cp),'threshold':.001,'minimum_top_two_gap':smallest,'flagged_rows':len(flagged),'rechecks':[]}
    if flagged:
        model,spec,_=load_checkpoint(cp)
        data=read(HERE/'tokens'/f"{spec['model']}_test_rotations.json")
        lookup={(x['id'],x['rotation']):x for x in data};meta=read(HERE/'DATA_AUDIT.json')['tokenization'][spec['model']]
        with torch.inference_mode():
            for x in flagged:
                item=lookup[x['id'],x['rotation']];ids,mask,lengths=batch([item],meta)
                scores=forward_selected(model,ids,mask,lengths)[0,meta['label_ids']].cpu().numpy()
                error=float(np.max(np.abs(scores-np.array(x['label_logits']))))
                result['rechecks'].append({'id':x['id'],'rotation':x['rotation'],'max_abs_logit_error':error,
                    'within_preexisting_5e_4_shape_tolerance':error<5e-4,
                    'original_prediction':x['prediction'],'single_prediction':int(scores.argmax()),
                    'prediction_changed':int(scores.argmax())!=x['prediction'],'ambiguous_gold':x['ambiguous_gold']})
    write(cp/'NUMERIC_TIES.json',result)
    print(json.dumps({'checkpoint':cp.name,'flagged':len(flagged),'flips':sum(x['prediction_changed'] for x in result['rechecks'])}))

def summarize():
    rr=[read(Path(s['checkpoint'])/'NUMERIC_TIES.json') for s in specs('confirmation')]
    checks=[x for r in rr for x in r['rechecks']]
    write(HERE/'NUMERIC_TIES_SUMMARY.json',{'at':now(),'analysis':'post hoc numeric diagnostic, no primary prediction replaced',
        'screened_rows':160000,'top_two_gap_threshold':.001,'flagged_rows':sum(x['flagged_rows'] for x in rr),
        'rechecked_rows':len(checks),'prediction_flips':sum(x['prediction_changed'] for x in checks),
        'all_rechecks_within_preexisting_shape_tolerance':all(x['within_preexisting_5e_4_shape_tolerance'] for x in checks),
        'maximum_observed_shape_logit_difference':max([x['max_abs_logit_error'] for x in checks],default=0),
        'scope':'Rechecks only observed near ties. This is not a global mathematical bound on untested batch shapes.',
        'script_sha256':sha(HERE/'numeric_ties.py'),'details':rr})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint');a=p.parse_args()
    if a.checkpoint:one(Path(a.checkpoint))
    else:summarize()
