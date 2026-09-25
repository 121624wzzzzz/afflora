"""Read-only compact snapshot; never import or modify a study's source."""
import collections,json,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
S=ROOT/'lora/reviewer_followup/gemma_transfer_20260918'
M=ROOT/'models/Gemma-2-9B-Base'
def read(p):return json.loads(p.read_text())
def main():
 out={'at':time.strftime('%Y-%m-%d %H:%M:%S %z')}
 meta=S/'provenance/google_gemma_2_9b_official_metadata.json'
 if meta.exists():
  shards=[x for x in read(meta)['siblings'] if x['rfilename'].endswith('.safetensors')]
  done=total=verified=0
  for x in shards:
   p=M/x['rfilename'];q=p.with_suffix(p.suffix+'.partial');total+=x['size']
   if p.exists():done+=p.stat().st_size;verified+=1
   else:
    if q.exists():done+=q.stat().st_size
    done+=sum(r.stat().st_size for r in M.glob(x['rfilename']+'.range_*') if r.is_file())
  out['download']={'GB':round(done/1e9,3),'total_GB':round(total/1e9,3),'percent':round(100*done/total,2),'verified_shards':verified,'total_shards':len(shards)}
 p=S/'STATE.json'
 if p.exists():
  state=read(p);out.update(phase=state['phase'],counts=state['counts'],state_at=state['at'])
  out['stage_passed']={stage:sum(j['state']=='passed' and j['spec']['stage']==stage for j in state['jobs']) for stage in ['smoke','search','confirmation','base']}
  active=[]
  for j in state['jobs']:
   if j['state'] not in ['running','auditing','failed']:continue
   name=j['spec']['name'];v={'name':name,'state':j['state'],'gpu':j.get('gpu'),'pid':j.get('pid')}
   p=S/'checkpoints'/name/'PROGRESS.json'
   if p.exists():
    r=read(p);v['train']={k:r[k] for k in ['step','steps','loss','elapsed_s']}
   for p in (S/'evaluations'/name).glob('*/PROGRESS.json'):
    v['evaluation']=dict(tag=p.parent.name,**read(p))
   if j['state']=='failed':
    logs=[S/'logs'/f'{name}.log',S/'logs'/f'{name}.audit.log']
    v['tails']={p.name:p.read_text(errors='replace').splitlines()[-8:] for p in logs if p.exists()}
   active.append(v)
  out['active']=active
 for f in ['MODEL_IDENTITY_AUDIT','PREFLIGHT_GATE','SELECTION','SCHEDULER_COMPLETE','FINAL_AUDIT']:
  p=S/(f+'.json')
  if p.exists():out[f]=read(p).get('status','present')
 if '--compact' in sys.argv:
  if 'active' in out:
   out['active']=[dict(name=x['name'],state=x['state'],gpu=x['gpu'],
     **({'train_step':x['train']['step']} if 'train' in x else {}),
     **({'eval':str(x['evaluation']['done'])+'/'+str(x['evaluation']['total'])} if 'evaluation' in x else {}),
     **({'tails':x['tails']} if 'tails' in x else {})) for x in out['active']]
  out.pop('download',None) if out.get('MODEL_IDENTITY_AUDIT')=='passed' else None
  print(json.dumps(out,ensure_ascii=False))
 else:print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
