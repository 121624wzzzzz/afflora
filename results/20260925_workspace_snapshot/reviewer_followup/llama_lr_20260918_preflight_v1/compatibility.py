"""Require the instrumented, grouped optimizer to reproduce the original fit."""
from common import *
from safetensors.torch import load_file
import torch

def main():
    source=HERE.parent/'llama_transfer_20260918'
    assert sha(source/'SEAL_MANIFEST.json')=='a995f154415e13d1a3200ebcb85b377b9ad704c5628d8e2d545e845d58fd7599'
    manifest=read(source/'SEAL_MANIFEST.json')['files'];checks=[]
    for spec in read(HERE/'COMPATIBILITY_JOBS.json'):
        assert read(HERE/'audits'/f'{spec["name"]}.json')['status']=='passed'
        old=source/'checkpoints'/f'wikisql_llama31_8b_base_{spec["arm"]}_s7100'
        new=HERE/'checkpoints'/spec['name'];hashes={}
        for filename in ['spec.json','INITIALIZATION.json','TRAIN_ORDER.json','TRAINING.json','initial_adapter.safetensors','adapter.safetensors']:
            p=old/filename;h=sha(p);assert manifest[str(p.relative_to(source))]['sha256']==h;hashes[filename]=h
        for key in ['arm','task','model','seed','lr','microbatch','smoke']:assert read(old/'spec.json')[key]==spec[key]
        assert read(old/'TRAIN_ORDER.json')==read(new/'TRAIN_ORDER.json')
        for key in ['initialization_sha256','shared_hidden_initialization_sha256','frozen_before','trainable_parameters','parameter_groups']:
            assert read(old/'INITIALIZATION.json')[key]==read(new/'INITIALIZATION.json')[key],key
        for filename in ['initial_adapter.safetensors','adapter.safetensors']:
            a,b=load_file(str(old/filename)),load_file(str(new/filename));assert a.keys()==b.keys()
            assert all(torch.equal(a[k],b[k]) for k in a),(spec['arm'],filename,'changed tensor')
        a,b=read(old/'TRAINING.json'),read(new/'TRAINING.json')
        assert [(x['loss'],x['grad_norm']) for x in a['history']]==[(x['loss'],x['grad_norm']) for x in b['history']]
        checks.append({'name':spec['name'],'source':str(old),'source_hashes_verified':hashes,'initial_and_final_tensors_exact':True,'loss_and_norm_history_exact':True})
    write(HERE/'COMPATIBILITY.json',{'at':now(),'status':'passed','checks':checks})
    print('compatibility passed: actual initial/final tensors, losses and norms are identical',flush=True)
if __name__=='__main__':main()
