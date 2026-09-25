from common import *
from transformers import AutoTokenizer
import collections

def mc_prompt(task,row,order):
 if task=='sciq':context='Question: '+row['question'];choices=row['choices']
 else:context='Premise: '+row['premise']+'\nHypothesis: '+row['hypothesis'];choices=['entailment (the hypothesis must be true)','neutral (the premise does not determine the hypothesis)','contradiction (the hypothesis must be false)']
 return ('Choose the correct answer. Return only its letter.\n\n'+context+'\n\n'+ '\n'.join(f'{chr(65+j)}. {choices[k]}' for j,k in enumerate(order))+'\n\nAnswer:')

def main():
 sources=read(HERE/'SOURCES.json');models=read(HERE/'models.json');jobs=[];source_checks={};exposures={};source_states={}
 for task in ['clinc150','wikisql']:
  src=Path(sources[task]);assert read(src/'STATE.json')['stage']=='complete';source_states[task]=read(src/'FINAL_AUDIT.json');assert source_states[task]['status']=='passed'
  assert sha(src/'OUTPUT_MANIFEST.json')==source_states[task]['manifest_sha256']
  for rel,h in read(src/'CODE_FROZEN.json')['files'].items():assert sha(src/rel)==h
  exposures[task]={' '.join(r.get('text',r.get('question','')).casefold().split()) for r in rows(src/f'data/{task}_train.jsonl')}
  for rec in read(src/'RESULTS.json')['records']:
   s=rec['spec']
   if s['stage']!='confirmation' or s['model'] not in models or s['task']!=task:continue
   if s['method'] not in ['none','affine','vocab'] or s['rank']!={'none':0,'affine':16,'vocab':1}[s['method']]:continue
   if rec['reused']:
    rr=next(x for x in read(src/'REUSE_AUDIT.json')['runs'] if x['spec']['name']==s['name']);cp=Path(rr['checkpoint']);ep=Path(rr['summary']).parent
   else:cp=src/'checkpoints'/s['name'];ep=src/'evaluations'/s['name']/'test'
   saved=read(cp/'spec.json');tr=read(cp/'TRAINING.json');assert tr['reload_loss_error']==0 and tr['frozen_before']==tr['frozen_after']
   assert all(saved[k]==s[k] for k in ['model','task','seed','method','placement','rank','bias','lr','boundary_lr_ratio'])
   assert read(cp.parent.parent/'audits'/f'{cp.name}.json')['status']=='passed'
   expected=read(cp.parent.parent/'OUTPUT_MANIFEST.json')['files']
   for rel in ['initial_adapter.safetensors','adapter.safetensors','spec.json','TRAINING.json','INITIALIZATION.json','TRAIN_ORDER.json']:
    f=cp/rel;h=sha(f);assert h==expected[str(f.relative_to(cp.parent.parent))];source_checks[str(f)]=h
   assert sha(cp/'adapter.safetensors')==tr['adapter_sha256']
   summ=read(ep/'SUMMARY.json');assert summ['primary']==rec['primary'];assert sha(ep/'responses.jsonl')==summ['responses_sha256'];source_checks[str(ep/'SUMMARY.json')]=sha(ep/'SUMMARY.json')
   jobs.append(dict(name=task+'__'+s['name'],model=s['model'],source_task=task,method=s['method'],seed=s['seed'],spec=saved,checkpoint=str(cp),checkpoint_sha256=tr['adapter_sha256'],tensor_sha256=tr['adapter_tensor_sha256'],source_primary=summ['primary'],source_summary=str(ep/'SUMMARY.json')))
 assert len(jobs)==60,len(jobs)
 for m in models:
  for task in ['clinc150','wikisql']:
   sets=[{j['seed'] for j in jobs if j['model']==m and j['source_task']==task and j['method']==method} for method in ['none','vocab','affine']]
   assert all(len(s)==5 for s in sets) and sets[0]==sets[1]==sets[2]
  s=dict(model=m,arm='base',seed=0,method='none',placement='none',rank=0,bias=False)
  jobs.append(dict(name='base__'+m,model=m,source_task='base',method='base',seed=0,spec=s,checkpoint=None,source_primary=None))
 files={};overlap={};token_stats={}
 for task in ['sciq','anli_r1']:
  ds=rows(HERE/f'data/{task}.jsonl');assert len(ds)==1000
  texts={' '.join((r['question'] if task=='sciq' else r['premise']+' '+r['hypothesis']).casefold().split()) for r in ds}
  for source_task,train in exposures.items():
   overlap[source_task+'/'+task]=len(train&texts);assert not train&texts
  files[str(HERE/f'data/{task}.jsonl')]=sha(HERE/f'data/{task}.jsonl')
 for m,c in models.items():
  for f,h in c['files'].items():assert sha(f)==h;files[f]=h
  tok=AutoTokenizer.from_pretrained(c['path'],local_files_only=True);eos=128001 if m.startswith('llama') else 151643
  assert read(Path(c['path'])/'config.json')['eos_token_id']==eos
  for task in ['sciq','anli_r1']:
   n=4 if task=='sciq' else 3;labels=[' '+chr(65+j) for j in range(n)];encoded=[tok.encode(s,add_special_tokens=False) for s in labels];assert all(len(t)==1 for t in encoded);label_ids=[t[0] for t in encoded];ts=[]
   for row in rows(HERE/f'data/{task}.jsonl'):
    gold=row['gold'] if task=='sciq' else 'ABC'.index(row['target'])
    variants=[]
    for rotation in range(n):
     order=[(j+rotation)%n for j in range(n)];prompt=mc_prompt(task,row,order);ids=tok.encode(prompt,add_special_tokens=False)
     for label,label_id in zip(labels,label_ids):assert tok.encode(prompt+label,add_special_tokens=False)==ids+[label_id]
     if m.startswith('llama'):ids=[128000]+ids
     assert len(ids)<=4096
     variants.append(dict(order=order,input_ids=ids))
    ts.append(dict(id=row['id'],gold=gold,variants=variants))
   path=HERE/f'tokens/{m}_{task}.json';write(path,dict(label_ids=label_ids,rows=ts));files[str(path)]=sha(path);token_stats[m+'/'+task]=dict(examples=len(ts),rotations=n,max_tokens=max(len(v['input_ids']) for r in ts for v in r['variants']))
  print('verified',m,flush=True)
 write(HERE/'JOBS.json',jobs);write(HERE/'INPUT_AUDIT.json',dict(at=now(),status='passed',files=files,source_checkpoint_files=source_checks,source_states=source_states,exact_source_train_overlap=overlap,token_stats=token_stats))
 print('jobs',len(jobs),flush=True)
if __name__=='__main__':main()
