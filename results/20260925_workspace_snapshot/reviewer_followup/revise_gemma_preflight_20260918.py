"""Archive a failed technical gate, preserve every artifact, copy verified inputs to a fresh root."""
import hashlib,json,shutil
from datetime import datetime
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P/'gemma_transfer_20260918';ARCHIVE=P/'gemma_transfer_20260918_preflight_v1'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 assert not ARCHIVE.exists() and read(ROOT/'SCHEDULER_COMPLETE.json')['status']=='failed'
 state=read(ROOT/'STATE.json');assert state['phase']=='preflight'
 assert sum(j['state']=='failed' for j in state['jobs'])==1 and not any(j['state'] in ['running','auditing'] for j in state['jobs'])
 assert all(j['state']=='pending' for j in state['jobs'] if j['spec']['stage']=='search')
 assert not (ROOT/'SELECTION.json').exists()
 d=read(ROOT/'NATIVE_SCORE_DIAGNOSIS.json');assert d['parameter_tensors_unchanged'] and d['native_loss_joint_max_abs_error']==0
 for f in ['CODE_FROZEN.json','DATA_FROZEN.json']:
  for rel,h in read(ROOT/f)['files'].items():assert sha(ROOT/rel)==h,rel
 for n in ['probe_gemma_native_scores_20260918.py','gemma_transfer_20260918_native_score_probe.log']:(ROOT/'provenance'/(n+'.txt')).write_bytes((P/n).read_bytes())
 ROOT.rename(ARCHIVE)
 fs={str(f.relative_to(ARCHIVE)):{'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(ARCHIVE.rglob('*')) if f.is_file() and '__pycache__' not in f.parts}
 write(ARCHIVE/'PREFLIGHT_ARCHIVE_MANIFEST.json',{'at':datetime.now().astimezone().isoformat(),'status':'sealed_failed_preflight','files':fs,'file_count':len(fs)})
 ROOT.mkdir()
 skip={'CODE_FROZEN.json','READY.json','RUNTIME.json','STATE.json','EVENTS.jsonl','SCHEDULER_COMPLETE.json','scheduler.log','scheduler.lock','NATIVE_SCORE_DIAGNOSIS.json','PREFLIGHT_ARCHIVE_MANIFEST.json'}
 copied={}
 for f in ARCHIVE.iterdir():
  if f.name in skip or f.name in {'checkpoints','evaluations','logs','specs','audits','__pycache__'}:continue
  if f.is_dir():shutil.copytree(f,ROOT/f.name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
  else:shutil.copy2(f,ROOT/f.name)
 for f in sorted(ROOT.rglob('*')):
  if f.is_file():
   rel=str(f.relative_to(ROOT));assert sha(f)==fs[rel]['sha256'];copied[rel]=fs[rel]['sha256']
 write(ROOT/'PREFLIGHT_REVISION.json',{'at':datetime.now().astimezone().isoformat(),'revision':2,'archive':str(ARCHIVE),
  'archive_manifest_sha256':sha(ARCHIVE/'PREFLIGHT_ARCHIVE_MANIFEST.json'),'original_code_frozen_sha256':sha(ARCHIVE/'CODE_FROZEN.json'),
  'failed_run':'smoke_trec50_hidden_both_c5_s7399','cause':'Cached versus native full-prefix joint logprob error 0.000336647 exceeds the fixed 0.0002 tolerance; argmax agrees on both smoke examples.',
  'diagnosis_sha256':sha(ARCHIVE/'NATIVE_SCORE_DIAGNOSIS.json'),'correction':'Use five native complete-prefix forwards for all TREC scores, with independent native ForCausalLMLoss checks; retain tolerance 0.0002.',
  'unchanged':['checkpoint','data and splits','tokenizer and prompts','parameter budget and architecture','training code and optimizer','all LR candidates','search and confirmation seeds','primary metrics and statistical family'],
  'no_formal_run_before_revision':True,'repeat_all_ten_smokes':True,'verified_copied_inputs':copied})
 print(json.dumps({'status':'archived_and_copied','archive_manifest_sha256':sha(ARCHIVE/'PREFLIGHT_ARCHIVE_MANIFEST.json'),'archived_files':len(fs),'copied_files':len(copied)}),flush=True)
if __name__=='__main__':main()
