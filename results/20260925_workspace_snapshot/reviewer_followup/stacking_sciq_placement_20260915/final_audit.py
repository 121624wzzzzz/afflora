"""Independent artifact, prediction arithmetic and pairing audit."""
import importlib.metadata,math
import numpy as np
from safetensors.torch import load_file
import torch
from common import *
from run import specs

def main():
    frozen=read(HERE/'FROZEN_PROTOCOL.json')
    for p,h in frozen['sha256'].items():assert sha(p)==h,p
    data_audit=read(HERE/'DATA_AUDIT.json')
    for p,h in data_audit['model_files_verified'].items():assert sha(p)==h,p
    for x in data_audit['source'].values():assert sha(HERE/'raw'/Path(x['upstream']).name)==x['sha256']
    checked=[];paired={};rows_checked=0
    for phase in ['smoke','tuning','confirmation']:
        for s in specs(phase):
            cp=Path(s['checkpoint']);assert (cp/'COMPLETE.json').exists() and not (cp/'FAILED.json').exists()
            assert read(cp/'spec.json')==s
            state=load_file(str(cp/'adapter.safetensors'))
            assert all(x.dtype==torch.float32 and torch.isfinite(x).all() for x in state.values())
            init=read(cp/'INITIALIZATION.json');training=read(cp/'TRAINING.json')
            assert sum(x.numel() for x in state.values())==init['total_parameters']
            assert training['adapter_sha256']==sha(cp/'adapter.safetensors')
            order=read(cp/'TRAIN_ORDER.json')['indices']
            assert len(set(order))==len(order)==training['examples']
            assert training['steps']==math.ceil(len(order)/32)
            assert training['steps']==(2 if phase=='smoke' else math.ceil(data_audit['splits']['train']['rows']/32))
            paired.setdefault((phase,s['model'],s['seed']),[]).append((s,init,sha(cp/'TRAIN_ORDER.json')))
            if phase=='smoke':assert read(cp/'RELOAD_AUDIT.json')['status']=='passed'
            else:
                split='validation' if phase=='tuning' else 'test';rr=rows(cp/f'{split}_predictions.jsonl');r=read(cp/f'{split}_metrics.json')
                assert len(rr)==(1000 if phase=='tuning' else 4000)
                assert r['prediction_sha256']==sha(cp/f'{split}_predictions.jsonl')
                source={x['id']:x for x in rows(HERE/'data'/f'{split}.jsonl')}
                for x in rr:
                    assert x['gold']==(source[x['id']]['gold']-x['rotation'])%4
                    assert x['ambiguous_gold']==source[x['id']]['ambiguous_gold']
                    logits=np.array(x['label_logits'],dtype=np.float64);gold=x['gold'];logz=np.logaddexp.reduce(logits)
                    assert int(logits.argmax())==x['prediction']
                    assert x['correct']==(x['prediction']==gold)
                    assert abs(x['nll']-(logz-logits[gold]))<1e-10
                    prob=np.exp(logits-logz);prob[gold]-=1
                    assert abs(x['brier']-float((prob**2).sum()))<1e-10
                    wrong=logits.copy();wrong[gold]=-np.inf
                    assert abs(x['margin']-float(logits[gold]-wrong.max()))<1e-5
                canonical=[x for x in rr if x['rotation']==0 and not x['ambiguous_gold']]
                assert len(canonical)==r['primary']['n']==(1000 if phase=='tuning' else 998)
                assert abs(100*np.mean([x['correct'] for x in canonical])-r['primary']['accuracy'])<1e-10
                assert abs(np.mean([x['nll'] for x in canonical])-r['primary']['candidate_nll'])<1e-10
                rows_checked+=len(rr)
            checked.append(s['name'])
    for key,group in paired.items():
        assert len({x[1]['shared_hidden_init_sha256'] for x in group})==1,key
        assert len({x[2] for x in group})==1,key
        assert len({x[1]['total_parameters'] for x in group if x[0]['arm']!='none'})==1,key
        assert len({x[1]['affine_init_sha256'] for x in group if x[0]['arm'] in ['both','interior']})==1,key
    result=read(HERE/'RESULTS.json')
    assert len(result['comparisons'])==6
    for c in result['comparisons']:
        control=c['contrast'].removeprefix('both-minus-');dd=[]
        for seed in range(2002,2007):
            rr={s['arm']:read(Path(s['checkpoint'])/'test_metrics.json')['primary']['accuracy'] for s in specs('confirmation') if s['model']==c['model'] and s['seed']==seed}
            dd.append(rr['both']-rr[control])
        assert np.allclose(dd,c['primary_adjusted']['seed_deltas'],atol=1e-12,rtol=0)
    write(HERE/'FINAL_AUDIT.json',{'status':'passed','at':now(),'completed_jobs':checked,'job_count':len(checked),
        'prediction_rows_checked':rows_checked,'paired_groups_checked':len(paired),
        'frozen_protocol_files_checked':len(frozen['sha256']),'base_files_rehashed':len(data_audit['model_files_verified']),
        'versions':{p:importlib.metadata.version(p) for p in ['torch','transformers','peft','numpy','scipy','safetensors']}})
    print(json.dumps({'status':'passed','jobs':len(checked),'prediction_rows':rows_checked,'paired_groups':len(paired)}))

if __name__=='__main__':main()
