"""Seal a finished study, or independently verify it without writing inside."""
import argparse, fcntl, hashlib, json
from datetime import datetime
from pathlib import Path

R=Path(__file__).resolve().parent/'llama_transfer_20260918'
EXCLUDED={'SEAL.json','SEAL_MANIFEST.json'}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def now():return datetime.now().astimezone().isoformat()
def files():return sorted(p for p in R.rglob('*') if p.is_file() and p.name not in EXCLUDED and p.suffix!='.pyc' and '__pycache__' not in p.parts)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--verify',action='store_true');args=ap.parse_args()
    if args.verify:
        seal=read(R/'SEAL.json');assert sha(R/'SEAL_MANIFEST.json')==seal['manifest_sha256']
        manifest=read(R/'SEAL_MANIFEST.json');current=files();assert {str(p.relative_to(R)) for p in current}==set(manifest['files'])
        for p in current:
            item=manifest['files'][str(p.relative_to(R))];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],p
        assert len(current)==manifest['file_count']
        print(json.dumps({'status':'passed','verified_at':now(),'file_count':len(current),'manifest_sha256':seal['manifest_sha256']}),flush=True);return
    assert not (R/'SEAL.json').exists() and not (R/'SEAL_MANIFEST.json').exists()
    lock=(R/'scheduler.lock').open('r+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert read(R/'SCHEDULER_COMPLETE.json')['status']=='passed'
    state=read(R/'STATE.json');assert state['counts']=={'pending':0,'running':0,'auditing':0,'passed':48,'failed':0}
    assert all(j['state']=='passed' for j in state['jobs'])
    audit=read(R/'FINAL_AUDIT.json');assert audit['status']=='passed' and audit['runs']==48
    analysis=read(R/'TEST_ANALYSIS.json');assert len(analysis['conditions'])==4 and all(c['complete'] for c in analysis['conditions'])
    assert read(R/'REPORT_REVIEW.json')['status']=='passed'
    assert (R/'FINAL_INTERPRETATION_ZH.md').is_file()
    write(R/'COMPLETION.json',{'at':now(),'status':'complete','all_planned_jobs_completed':True,'formal_fits':36,'base_evaluations':4,'technical_smokes':8,'formal_reuse':0,'responses_audited':audit['responses_checked'],'complete_three_seed_conditions':4,'scope':'finite Llama extension, separate from the sealed prior 14-hour Qwen study'})
    collected={str(p.relative_to(R)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files()}
    write(R/'SEAL_MANIFEST.json',{'at':now(),'status':'sealed','excluded':sorted(EXCLUDED)+['**/__pycache__/**','**/*.pyc'],'file_count':len(collected),'total_bytes':sum(v['bytes'] for v in collected.values()),'files':collected})
    h=sha(R/'SEAL_MANIFEST.json');write(R/'SEAL.json',{'at':now(),'status':'sealed','manifest_sha256':h,'file_count':len(collected)})
    print(json.dumps({'status':'sealed','manifest_sha256':h,'file_count':len(collected)}),flush=True)
if __name__=='__main__':main()
