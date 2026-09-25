"""Technical admission: all smokes, native semantics and actual common initialization."""
import torch
from safetensors.torch import load_file
from common import *
def main():
 assert not (HERE/'PREFLIGHT_GATE.json').exists()
 assert read(HERE/'MODEL_FORMULA_AUDIT.json')['status']=='passed'
 ref=boundary=None;frozen=set();checks=0;repetition=[]
 revision=read(HERE/'PREFLIGHT_REVISION.json');archive=Path(revision['archive'])
 assert sha(archive/'PREFLIGHT_ARCHIVE_MANIFEST.json')==revision['archive_manifest_sha256']
 for s in read(HERE/'SMOKE_JOBS.json'):
  root=HERE/'checkpoints'/s['name'];a=read(HERE/'audits'/f'{s["name"]}.json');assert a['status']=='passed'
  init=read(root/'INITIALIZATION.json');frozen.add(init['frozen_before'])
  old=archive/'checkpoints'/s['name'];oi=read(old/'INITIALIZATION.json')
  assert init['initialization_sha256']==oi['initialization_sha256']
  assert read(root/'TRAIN_ORDER.json')==read(old/'TRAIN_ORDER.json')
  nt=read(root/'TRAINING.json');ot=read(old/'TRAINING.json')
  repetition.append({'run':s['name'],'initialization_exact':True,'order_exact':True,
       'final_adapter_exact':nt['adapter_tensor_sha256']==ot['adapter_tensor_sha256'],
       'step_losses_exact':[r['loss'] for r in nt['history']]==[r['loss'] for r in ot['history']]})
  assert init['preflight']['zero_residual_max_abs_error']==0
  assert init['worst_length_memory_probe']['status']=='passed'
  d=load_file(str(root/'initial_adapter.safetensors'))
  shared={k:(v[:8] if '.lora_A.' in k else v[:,:8]) for k,v in d.items() if '.lora_' in k}
  if ref is None:ref=shared
  else:
   assert shared.keys()==ref.keys() and all(torch.equal(v,ref[k]) for k,v in shared.items());checks+=len(shared)
  bd={k:v for k,v in d.items() if k.startswith('boundary_')}
  if bd:
   if boundary is None:boundary=bd
   else:assert bd.keys()==boundary.keys() and all(torch.equal(v,boundary[k]) for k,v in bd.items())
 assert len(frozen)==1
 write(HERE/'PREFLIGHT_GATE.json',{'at':now(),'status':'passed','smokes':len(read(HERE/'SMOKE_JOBS.json')),'shared_tensors_compared':checks,
       'native_formula_audit_sha256':sha(HERE/'MODEL_FORMULA_AUDIT.json'),'technical_repetition':repetition,
       'preflight_revision_sha256':sha(HERE/'PREFLIGHT_REVISION.json')})
 print('preflight gate passed',flush=True)
if __name__=='__main__':main()
