"""Read-only compact status snapshot."""
import json,collections,sys
from pathlib import Path
root=Path(__file__).resolve().parent/'qwen35_transfer_20260919'
def read(p):return json.loads(p.read_text())
def main():
 p=root/'STATE.json'
 if not p.exists():print('no queue state yet');return
 d=read(p);out={'at':d['at'],'phase':d['phase'],'counts':d['counts'],'active':[],'recent_finished':[]}
 for j in d['jobs']:
  s=j['spec'];n=s['name'];c=root/'checkpoints'/n
  if j['state'] in ['running','auditing']:
   r={'name':n,'gpu':j['gpu'],'state':j['state']}
   p=c/'PROGRESS.json'
   if p.exists():
    v=read(p);r['training']={k:v[k] for k in ['step','steps','elapsed_s']}
   ps=list((root/'evaluations'/n).glob('*/PROGRESS.json'))
   if ps:r['evaluation']={p.parent.name:read(p) for p in ps}
   out['active'].append(r)
  elif j['state']=='failed':
   r={'name':n,'state':'failed'}
   for typ in ['log','audit.log']:
    p=root/'logs'/f'{n}.{typ}'
    if p.exists():r[typ]=p.read_text().splitlines()[-12:]
   out['recent_finished'].append(r)
  elif j['state']=='passed':out['recent_finished'].append({'name':n,'at':j['finished_at']})
 out['recent_finished']=out['recent_finished'][-6:]
 if '--compact' in sys.argv:
  print(out['at'],out['phase'],out['counts'])
  for r in out['active']:
   v=r.get('training',{});e=r.get('evaluation',{})
   print(r['gpu'],r['name'].replace('search_','').replace('confirmation_',''),r['state'],str(v.get('step','-'))+'/'+str(v.get('steps','-')),{k:str(x['done'])+'/'+str(x['total']) for k,x in e.items()})
  for r in out['recent_finished']:
   if r.get('state')=='failed':print(r)
 else:print(json.dumps(out,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
