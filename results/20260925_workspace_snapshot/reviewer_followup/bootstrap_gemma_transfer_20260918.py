"""Copy only individually verified sealed inputs into a separate Gemma study."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];NEW=ROOT/'lora/reviewer_followup/gemma_transfer_20260918'
LR=ROOT/'lora/reviewer_followup/llama_lr_20260918';WIDE=ROOT/'lora/reviewer_followup/model_architecture_14h_20260917'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert not (NEW/'common.py').exists()
sources={};entries={}
for old,expected in [(LR,'3193a683cf4fed9f6c85ba07cfb24227c6c8d321c934d3c125afa83ba2b6aa90'),(WIDE,'e0c21b5788e013a90b9700fe8dcdf0c55959afb5e35fd1d69d8107f581262e00')]:
 assert sha(old/'SEAL_MANIFEST.json')==expected
 sources[str(old)]={'manifest_sha256':expected,'files':json.loads((old/'SEAL_MANIFEST.json').read_text())['files']}
def copy(old,rel,out):
 p=old/rel;h=sha(p);assert h==sources[str(old)]['files'][rel]['sha256'],rel
 q=NEW/out;q.parent.mkdir(parents=True,exist_ok=True);assert not q.exists();shutil.copyfile(p,q)
 entries[out]={'source':str(p),'sha256':h}
for f in ['common.py','modeling.py','run.py','audit_one.py','scoring_sql.py']:
 copy(LR,f,f)
for directory in ['source','raw']:
 for p in (LR/directory).rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':copy(LR,str(p.relative_to(LR)),str(p.relative_to(LR)))
for split in ['train','dev','confirm']:copy(LR,f'data/wikisql_{split}.jsonl',f'data/wikisql_{split}.jsonl')
for split in ['train','dev','test']:copy(WIDE,f'new_tasks/data/trec50_{split}.jsonl',f'data/trec50_{"confirm" if split=="test" else split}.jsonl')
copy(WIDE,'new_tasks/data/categories.json','data/categories.json')
for f in ['TREC_10.label','train_5500.label','definition.html']:copy(WIDE,'new_tasks/raw/trec/'+f,'raw/trec/'+f)
copy(WIDE,'new_tasks/scoring.py','scoring_trec.py')
copy(WIDE,'new_tasks/classification.py','classification.py')
copy(WIDE,'new_tasks/common.py','provenance/reference_trec_common.py.txt')
copy(LR,'common.py','provenance/reference_sql_common.py.txt')
(NEW/'SOURCE_REUSE.json').write_text(json.dumps({'status':'passed','source_manifests':{p:v['manifest_sha256'] for p,v in sources.items()},'copied_files':entries,
 'scope':'verified source code, official scoring dependencies, fixed benchmark splits and raw references; no trained adapters or predictions reused',
 'holdout_status':'Gemma outputs are new; these shared benchmark splits have been evaluated on other model families and are not globally unseen project data'},indent=2)+'\n')
print('Verified copied files',len(entries),flush=True)
