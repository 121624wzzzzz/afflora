"""One initial training-batch backward pass; no updates, eval or search decisions."""
import argparse,hashlib,json,math,sys
from datetime import datetime
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['llama31_8b_base','qwen3_8b_base'],required=True);a=ap.parse_args()
    assert read(ROOT/'llama_lr_20260918/SCHEDULER_COMPLETE.json')['status']=='passed'
    if a.model.startswith('llama'):
        study=sealroot=ROOT/'llama_transfer_20260918';expected='a995f154415e13d1a3200ebcb85b377b9ad704c5628d8e2d545e845d58fd7599'
    else:
        sealroot=ROOT/'model_architecture_14h_20260917';study=sealroot/'core';expected='e0c21b5788e013a90b9700fe8dcdf0c55959afb5e35fd1d69d8107f581262e00'
    assert sha(sealroot/'SEAL_MANIFEST.json')==expected
    manifest=read(sealroot/'SEAL_MANIFEST.json')['files'];verified={}
    name=f'wikisql_{a.model}_hidden_both_s7100'
    for rel in ['common.py','modeling.py','run.py','scoring.py','scoring_sql.py','scoring_ner.py','source/affine_adapter.py','source/budget.py','models.json',
                f'tokens/wikisql_{a.model}_train.json',f'checkpoints/{name}/INITIALIZATION.json',f'checkpoints/{name}/TRAINING.json',f'checkpoints/{name}/TRAIN_ORDER.json']:
        p=study/rel;h=sha(p);assert h==manifest[str(p.relative_to(sealroot))]['sha256'];verified[str(p)]=h
    cfg=read(study/'models.json')[a.model]
    for path,h in cfg['files'].items():assert sha(Path(path))==h;verified[path]=h
    sys.path.insert(0,str(study))
    from modeling import build,batch,loss,adapter_state,digest,frozen_digest,is_adapter
    from run import preflight
    spec={'name':'initial_gradient_probe','arm':'hidden_both','model':a.model,'task':'wikisql','seed':7100,'lr':2e-4,'microbatch':4,'smoke':False}
    model,init=build(spec);tokens=read(study/f'tokens/wikisql_{a.model}_train.json')
    old_init=read(study/f'checkpoints/{name}/INITIALIZATION.json');assert init['initialization_sha256']==old_init['initialization_sha256']
    assert init['boundary_initialization_sha256']==old_init['boundary_initialization_sha256']
    pre=preflight(model,spec,tokens);frozen_before=frozen_digest(model);adapter_before=digest(adapter_state(model))
    order=torch.randperm(len(tokens),generator=torch.Generator().manual_seed(7100)).tolist()[:32]
    old_order=read(study/f'checkpoints/{name}/TRAIN_ORDER.json');assert order==old_order['indices'][:32]
    assert [tokens[i]['id'] for i in order]==old_order['ids'][:32]
    denom=sum(len(tokens[i]['target_ids']) for i in order);total=0.;model.train();model.zero_grad(set_to_none=True)
    for start in range(0,32,4):
        ids,mask,labels=batch([tokens[i] for i in order[start:start+4]])
        with torch.autocast('cuda',dtype=torch.bfloat16):_,ce,_=loss(model,ids,mask,labels);value=ce.sum()/denom
        assert torch.isfinite(value);value.backward();total+=float(value.detach())
    assert all(p.grad is None for n,p in model.named_parameters() if not is_adapter(n))
    groups={}
    for n,p in model.named_parameters():
        if not p.requires_grad:continue
        group=n if n.startswith('boundary_') else 'hidden'
        record=groups.setdefault(group,{'parameters':0,'gradient_squared_norm':0.})
        record['parameters']+=p.numel();assert p.grad is not None
        g=p.grad.detach().cpu().double();assert torch.isfinite(g).all();record['gradient_squared_norm']+=float(g.square().sum())
    squared=sum(g['gradient_squared_norm'] for g in groups.values());norm=math.sqrt(squared)
    for g in groups.values():
        g['gradient_norm']=math.sqrt(g['gradient_squared_norm']);g['gradient_rms']=math.sqrt(g['gradient_squared_norm']/g['parameters']);g['total_gradient_energy_share']=g['gradient_squared_norm']/squared
    first=read(study/f'checkpoints/{name}/TRAINING.json')['history'][0]
    assert total==first['loss'],(total,first['loss'])
    assert math.isclose(norm,first['grad_norm'],rel_tol=2e-6),(norm,first['grad_norm'])
    assert digest(adapter_state(model))==adapter_before and frozen_digest(model)==frozen_before
    out=ROOT/'llama_lr_20260918/mechanism_probe';out.mkdir(exist_ok=True);p=out/f'{a.model}.json';assert not p.exists()
    result={'at':datetime.now().astimezone().isoformat(),'model':a.model,'status':'passed','seed':7100,'purpose':'Posthoc initial-gradient decomposition; zero optimizer updates; no heldout evaluation.',
            'source_manifest_sha256':expected,'verified_files':verified,'initialization_sha256':init['initialization_sha256'],
            'boundary_initialization_sha256':init['boundary_initialization_sha256'],'example_ids':[tokens[i]['id'] for i in order],
            'target_tokens':denom,'preflight':pre,'loss':total,'matches_original_first_step_loss_exactly':True,
            'total_grad_norm_fp64_aggregation':norm,'original_total_grad_norm':first['grad_norm'],'groups':groups,
            'frozen_and_adapter_parameters_unchanged':True}
    p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['model','loss','total_grad_norm_fp64_aggregation','groups']}),flush=True)
if __name__=='__main__':main()
