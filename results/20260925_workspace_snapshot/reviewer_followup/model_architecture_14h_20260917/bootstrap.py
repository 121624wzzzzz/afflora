"""Copy only manifest-verified sources; pin and verify every new model identity."""
import concurrent.futures,hashlib,json,pathlib,requests,shutil,subprocess
from datetime import datetime,timezone,timedelta
ROOT=pathlib.Path(__file__).resolve().parent
HERE=ROOT/'core';OLD=ROOT.parent/'model_scaling_20260917';REPO=ROOT.parents[2]
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def write(p,x):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 start=datetime(2026,9,17,13,7,10,tzinfo=timezone.utc)
 write(ROOT/'WINDOW.json',{'start_utc':start.isoformat(),'deadline_utc':(start+timedelta(hours=14)).isoformat(),'stop_new_launches_utc':(start+timedelta(hours=12,minutes=30)).isoformat(),'finish_workers_by_utc':(start+timedelta(hours=13)).isoformat(),'policy':'Fixed question-driven priorities; no score-based scheduling. Reserve final hour for audit/report/seal. Incomplete blocks are disclosed, never silently discarded.'})
 manifest=read(OLD/'ARTIFACT_MANIFEST.json');assert sha(OLD/'ARTIFACT_MANIFEST.json')=='adb6c2f3b26e26208b9bb3b71ecd445cfd525e01bcc7a02ad0f2168594e06ae9'
 copied=[]
 names=['common.py','modeling.py','run.py','scoring.py','scoring_ner.py','scoring_sql.py','test_scoring.py']
 names += [p for p in manifest['files'] if p.startswith(('source/','data/','raw/'))]
 for name in names:
  source=OLD/name;assert sha(source)==manifest['files'][name]['sha256'],name
  target=HERE/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
  assert sha(target)==sha(source);copied.append({'source':str(source),'destination':name,'sha256':sha(target)})
 write(HERE/'SOURCE_REUSE.json',{'status':'passed','source_manifest_sha256':sha(OLD/'ARTIFACT_MANIFEST.json'),'files':copied})
 configs={}
 for old in [OLD,ROOT.parent/'posttraining_tasks_grounded_20260916']:
  m=read(old/'ARTIFACT_MANIFEST.json');assert sha(old/'models.json')==m['files']['models.json']['sha256']
  for key,cfg in read(old/'models.json').items():
   for p,h in cfg['files'].items():assert sha(p)==h,p
   configs[key]=cfg
 def verify_one(item):
  key,repo,localname=item;session=requests.Session();session.trust_env=False
  response=session.get('https://huggingface.co/api/models/'+repo+'?blobs=true',timeout=60);response.raise_for_status();meta=response.json();revision=meta['sha']
  write(HERE/'provenance'/f'{key}_api.json',meta)
  dest=HERE/'verified_models'/repo.split('/')[-1];dest.mkdir(parents=True,exist_ok=True);local=REPO/'models'/localname;files={};details=[]
  wanted={'config.json','generation_config.json','tokenizer_config.json','tokenizer.json','merges.txt','vocab.json','model.safetensors.index.json','README.md','LICENSE'}
  for record in meta['siblings']:
   name=record['rfilename']
   if name not in wanted and not name.endswith('.safetensors'):continue
   target=dest/name
   if name.endswith('.safetensors'):
    expected=record['lfs']['sha256'];assert sha(local/name)==expected,(local/name,'weight identity mismatch')
    subprocess.run(['cp','--reflink=auto','--preserve=mode,timestamps',str(local/name),str(target)],check=True)
    assert sha(target)==expected
   else:
    response=session.get(f'https://huggingface.co/{repo}/resolve/{revision}/{name}',timeout=60);response.raise_for_status();data=response.content
    if record.get('lfs'):assert hashlib.sha256(data).hexdigest()==record['lfs']['sha256']
    else:assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==record['blobId'],name
    target.write_bytes(data)
   files[str(target)]=sha(target);details.append({'file':name,'sha256':files[str(target)],'official':record})
  cfg=read(dest/'config.json');assert cfg['eos_token_id']==151643
  result={'path':str(dest),'repo':repo,'revision':revision,'hidden_size':cfg['hidden_size'],'tie_word_embeddings':cfg['tie_word_embeddings'],'files':files,'training_stage':'pretraining_only'}
  write(HERE/'provenance'/f'{key}_identity.json',{'status':'passed','model':result,'files':details});print(key,'verified',flush=True)
  return key,result
 new=[('qwen25_05b_base','Qwen/Qwen2.5-0.5B','Qwen2.5-0.5B-Base'),('qwen3_17b_base','Qwen/Qwen3-1.7B-Base','Qwen3-1.7B-Base'),('qwen3_4b_base','Qwen/Qwen3-4B-Base','Qwen3-4B-Base'),('qwen3_8b_base','Qwen/Qwen3-8B-Base','Qwen3-8B-Base')]
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for key,cfg in pool.map(verify_one,new):configs[key]=cfg
 write(HERE/'models.json',configs);write(HERE/'MODEL_IDENTITY_AUDIT.json',{'status':'passed','models':configs,'source_models_rehashed':True})
 print('All eight official Base identities verified.',flush=True)
if __name__=='__main__':main()
