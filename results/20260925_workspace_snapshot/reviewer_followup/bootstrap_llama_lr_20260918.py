"""Create a separate, bounded LR follow-up; never mutate a sealed source."""
import hashlib,json,shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SRC=ROOT/'llama_transfer_20260918'
DST=ROOT/'llama_lr_20260918'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
assert sha(SRC/'SEAL_MANIFEST.json')=='a995f154415e13d1a3200ebcb85b377b9ad704c5628d8e2d545e845d58fd7599'
manifest=json.loads((SRC/'SEAL_MANIFEST.json').read_text())['files']
DST.mkdir(exist_ok=False)
copied={}
paths=['common.py','modeling.py','run.py','audit_one.py','scoring.py','scoring_sql.py','scoring_ner.py','models.json','BUDGET_PLAN.json']
paths += [str(p.relative_to(SRC)) for directory in ['source','raw'] for p in (SRC/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
paths += ['data/wikisql_train.jsonl','tokens/wikisql_llama31_8b_base_train.json']
for rel in paths:
    p=SRC/rel;h=sha(p);assert manifest[rel]['sha256']==h,rel
    q=DST/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert sha(q)==h
    copied[rel]=h
(DST/'SOURCE_REUSE.json').write_text(json.dumps({'source':str(SRC),'source_manifest_sha256':sha(SRC/'SEAL_MANIFEST.json'),'verified_copied_files':copied,'checkpoint_reuse':False},indent=2)+'\n')
print(DST)
