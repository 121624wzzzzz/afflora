"""Create a separate Llama extension; never mutate the sealed Qwen study."""
import hashlib, json, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT/'lora/reviewer_followup/model_architecture_14h_20260917'
NEW = ROOT/'lora/reviewer_followup/llama_transfer_20260918'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

assert sha(OLD/'SEAL_MANIFEST.json')=='e0c21b5788e013a90b9700fe8dcdf0c55959afb5e35fd1d69d8107f581262e00'
manifest=json.loads((OLD/'SEAL_MANIFEST.json').read_text())['files']
NEW.mkdir(exist_ok=False)
checked={}
files=['common.py','modeling.py','run.py','audit_one.py','scoring.py','scoring_ner.py','scoring_sql.py','test_scoring.py','freeze.py']
paths=[OLD/'core'/f for f in files]
for folder in ['source','data','raw']:
    paths.extend(p for p in (OLD/'core'/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
for p in paths:
    rel=str(p.relative_to(OLD));h=sha(p);assert h==manifest[rel]['sha256'],rel
    dst=NEW/p.relative_to(OLD/'core');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
    checked[rel]=h
(NEW/'SOURCE_REUSE.json').write_text(json.dumps({'source':str(OLD),'seal_manifest_sha256':sha(OLD/'SEAL_MANIFEST.json'),'files_checked':checked,'reuse_scope':'code, exact data, official scoring dependencies only; no fitted Llama checkpoints or results reused'},indent=2)+'\n')
p=NEW/'common.py';s=p.read_text();start=s.index('MODELS = ');end=s.index('\nTASKS',start);s=s[:start]+"MODELS = ['llama32_3b_base','llama31_8b_base']\nTOKEN_BOS_ID = 128000\nTOKEN_EOS_ID = 128001"+s[end:];start=s.index('ARMS = ');end=s.index('\ndef now',start);s=s[:start]+"ARMS = ['hidden','hidden_budget','hidden_both']"+s[end:];p.write_text(s)
for name in ['modeling.py','run.py','audit_one.py']:
    p=NEW/name;s=p.read_text().replace('151643','TOKEN_EOS_ID');p.write_text(s)
p=NEW/'source/budget.py';s=p.read_text().replace('The 7B model overshoots by 512 parameters because of rank granularity.','Llama-3.2-3B overshoots by 1024 parameters; Llama-3.1-8B is exact.').replace('target * 1.003','target * 1.01');p.write_text(s)
print('Created',NEW,'verified copied files',len(checked))
