"""Independent initialization, frozen-base and paired-order checks on available runs."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict
import torch
from safetensors import safe_open
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def boundary_digest(state):
 h=hashlib.sha256()
 for n,p in sorted(state.items()):
  q=p.detach().cpu().contiguous();h.update(n.encode());h.update(str(q.dtype).encode());h.update(str(tuple(q.shape)).encode());h.update(q.view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
def hidden_digest(state):
 h=hashlib.sha256()
 for n,p in sorted(state.items()):
  if '.lora_' not in n:continue
  v=p[:8,:] if '.lora_A.' in n else p[:,:8];h.update(n.encode());h.update(v.float().contiguous().numpy().tobytes())
 return h.hexdigest()
def main():
 groups=defaultdict(lambda:defaultdict(list));base=defaultdict(list);runs=[];anchors_checked=0
 for phase in ['core','new_tasks']:
  path=ROOT/phase
  if not (path/'INVENTORY.json').exists():continue
  anchors={a['directory']:a for a in read(path/'ANCHOR_INDEX.json')} if (path/'ANCHOR_INDEX.json').exists() else {}
  budgets=read(path/'BUDGET_PLAN.json');models=read(path/'models.json')
  for entry in read(path/'INVENTORY.json'):
   spec=entry['spec'];key=(phase,spec['task'],spec['model'],spec['seed']);arm=spec['arm']
   if entry['reused']:
    a=anchors[entry['anchor_directory']];ck=path/a['directory'];original=Path(a['source_checkpoint']);adapter=original/'initial_adapter.safetensors'
    for name,h in a['adapter_hashes'].items():assert sha(original/name)==h,(original,name)
    anchors_checked+=1
   else:
    ck=path/'checkpoints'/spec['name'];adapter=ck/'initial_adapter.safetensors'
    if not (ck/'INITIALIZATION.json').exists() or (arm!='base' and not (ck/'TRAIN_ORDER.json').exists()):continue
   init=read(ck/'INITIALIZATION.json');assert init['trainable_parameters']==budgets[spec['model']][arm]
   assert init['tied_weights']==models[spec['model']]['tie_word_embeddings']
   base[spec['model']].append(init['frozen_before'])
   if arm!='base':
    with safe_open(str(adapter),framework='pt',device='cpu') as f:state={n:f.get_tensor(n) for n in f.keys()}
    assert set(state)==set(init['trainable_names'])
    assert boundary_digest(state)==init['initialization_sha256']
    if arm.startswith('hidden'):
     h=hidden_digest(state);assert h==init['shared_hidden_initialization_sha256'];groups[key]['H'].append(h)
    for side in ['input','output']:
     sub={n:p for n,p in state.items() if n.startswith('boundary_'+side)}
     if sub:
      h=boundary_digest(sub)
      if 'boundary_initialization_sha256' in init:assert h==init['boundary_initialization_sha256'][side]
      groups[key][side].append(h)
    groups[key]['order'].append(sha(ck/'TRAIN_ORDER.json'))
   runs.append({'phase':phase,'name':spec['name'],'reused':entry['reused'],'arm':arm})
 for model,values in base.items():assert len(set(values))==1,(model,'different original model tensors across arms/tasks')
 for key,g in groups.items():
  for part,values in g.items():assert len(set(values))==1,(key,part,'different paired initialization or order')
 result={'at':datetime.now(timezone.utc).isoformat(),'status':'passed','scope':'available initialization/order files only; consult scheduler for completion; no inference from missing runs','runs':runs,'model_frozen_digest_groups':len(base),'paired_groups':len(groups),'reused_runs_rehashed':anchors_checked}
 p=ROOT/'PAIR_AUDIT_PROGRESS.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(p)
 print('Pair audit passed:',len(runs),'available runs,',len(groups),'groups,',anchors_checked,'reused checkpoints rehashed')
if __name__=='__main__':main()
