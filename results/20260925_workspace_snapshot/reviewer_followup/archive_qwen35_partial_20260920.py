"""Preserve all old-policy partial fits; no old outcome is reused in corrected fits."""
import os,json,hashlib,subprocess
from pathlib import Path
from datetime import datetime
PARENT=Path(__file__).resolve().parent
MASTER=PARENT/'qwen35_multiscale_20260919'
PYTHON='/home/wz/anaconda3/envs/qwen35_t26/bin/python'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 retirement=read(MASTER/'RETIREMENT.json');result=[]
 for key in ['08b','2b','9b']:
  root=PARENT/f'qwen35_{key}_20260919';assert not (root/'PARTIAL_ARCHIVE_MANIFEST.json').exists()
  for manifest in ['CODE_FROZEN.json','DATA_FROZEN.json']:
   for rel,h in read(root/manifest)['files'].items():assert sha(root/rel)==h,(key,rel)
  for j in retirement['workers_finished_during_pause']:
   if j['model']!=key:continue
   with (root/'logs'/f'{j["name"]}.audit.log').open('x') as f:
    subprocess.run([PYTHON,'-u',str(root/'audit_one.py'),'--name',j['name']],cwd=root,
     env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=f,stderr=subprocess.STDOUT,check=True)
  completed=sorted(p.parent.name for p in (root/'checkpoints').glob('*/COMPLETE.json'))
  assert all(read(root/'audits'/f'{n}.json')['status']=='passed' for n in completed)
  note=dict(at=datetime.now().astimezone().isoformat(),status='retired_incomplete',completed_jobs=completed,
   reason=retirement['reason'],scope='All original frozen source/data and completed old-policy fits retained in place. These are not corrected-policy effects; no selective checkpoint/prediction reuse.',
   original_state_is_stale=True,retirement_sha256=sha(MASTER/'RETIREMENT.json'))
  write(root/'PARTIAL_ARCHIVE.json',note)
  files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
  write(root/'PARTIAL_ARCHIVE_MANIFEST.json',dict(status='archived_incomplete',files=files))
  result.append(dict(model=key,root=str(root),completed_jobs=len(completed),manifest_sha256=sha(root/'PARTIAL_ARCHIVE_MANIFEST.json')))
  print(json.dumps(result[-1]),flush=True)
 write(MASTER/'PARTIAL_ARCHIVES.json',dict(status='passed',studies=result))
if __name__=='__main__':main()
