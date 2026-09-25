import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parent
MODELS=['qwen25_05b_base','qwen25_15b_base','qwen25_3b_base','qwen25_7b_base','qwen3_06b_base','qwen3_17b_base','qwen3_4b_base','qwen3_8b_base']
LABELS=dict(zip(MODELS,['Qwen2.5-0.5B','Qwen2.5-1.5B','Qwen2.5-3B','Qwen2.5-7B','Qwen3-0.6B','Qwen3-1.7B','Qwen3-4B','Qwen3-8B']))
TASKS={'core':['cluener','wikisql'],'new_tasks':['trec50','squad2']}
SEEDS={'cluener':6100,'wikisql':7100,'trec50':9100,'squad2':10100}
ARMS=['base','hidden','hidden_budget','hidden_input','hidden_output','hidden_both','input','output','both']
ARM_LABELS={'base':'Base','hidden':'H','hidden_budget':'预算H','hidden_input':'H+E','hidden_output':'H+U','hidden_both':'H+E+U','input':'E','output':'U','both':'E+U'}
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def now():return datetime.now(timezone.utc).isoformat()
def collect(phase,split='test'):
 path=ROOT/phase;result={}
 anchors={a['directory']:a for a in read(path/'ANCHOR_INDEX.json')} if (path/'ANCHOR_INDEX.json').exists() else {}
 for entry in read(path/'INVENTORY.json'):
  s=entry['spec'];key=(s['task'],s['model'],s['arm'],None if s['arm']=='base' else s['seed']);summary=None
  if entry['reused']:
   if split!='test':continue
   a=anchors[entry['anchor_directory']];p=path/a['directory']/'SUMMARY.json';summary=read(p)
  else:
   ap=path/'audits'/f"{s['name']}.json"
   if not ap.exists():continue
   audit=read(ap);assert audit['status']=='passed'
   p=path/'evaluations'/s['name']/split/'SUMMARY.json';assert sha(p)==audit['summary_sha256'][split];summary=read(p)
  result[key]={'primary':summary['primary'],'summary':summary,'path':p,'reused':entry['reused'],'name':s['name']}
 return result
def series(data,task,model,arm,n=3):
 seeds=[None] if arm=='base' else list(range(SEEDS[task],SEEDS[task]+n))
 records=[data[task,model,arm,s] for s in seeds if (task,model,arm,s) in data]
 return [r['primary'] for r in records] if len(records)==len(seeds) else None
def cell(data,task,model,arm,n=3):
 v=series(data,task,model,arm,n)
 if v is not None:return f'{np.mean(v):.3f}'
 done=sum((task,model,arm,s) in data for s in ([None] if arm=='base' else range(SEEDS[task],SEEDS[task]+n)))
 return f'未齐（{done}/{1 if arm=="base" else n}）'
