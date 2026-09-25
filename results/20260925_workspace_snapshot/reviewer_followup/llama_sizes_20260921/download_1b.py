"""Download public ModelScope distribution; verify against official Meta hashes."""
import json,hashlib,time
from pathlib import Path
import requests
R=Path(__file__).resolve().parent;DEST=R.parents[2]/'models/Llama-3.2-1B-Base'
S=requests.Session();S.trust_env=False
meta_path=R/'provenance/llama32_1b_official.json'
def get(url,**kw):
 for i in range(4):
  try:
   q=S.get(url,timeout=(30,180),**kw);q.raise_for_status();return q
  except requests.RequestException:
   if i==3:raise
   time.sleep(3)
if meta_path.exists():meta=json.loads(meta_path.read_text())
else:
 meta=get('https://huggingface.co/api/models/meta-llama/Llama-3.2-1B',params={'blobs':'true'}).json();meta_path.write_text(json.dumps(meta,indent=2)+'\n')
DEST.mkdir(exist_ok=True);files={}
for rec in meta['siblings']:
 n=rec['rfilename']
 if '/' in n or not(n.endswith('.safetensors') or n in ['config.json','generation_config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','model.safetensors.index.json','LICENSE','README.md']):continue
 p=DEST/n
 def verify():
  if not p.exists() or p.stat().st_size!=rec['size']:return False
  h=hashlib.sha256()
  with p.open('rb') as f:
   for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
  if 'lfs' in rec:ok=h.hexdigest()==rec['lfs']['sha256']
  else:
   raw=p.read_bytes();ok=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==rec['blobId']
  if ok:files[str(p)]=h.hexdigest()
  return ok
 if not verify():
  print('download',n,rec['size'],flush=True);tmp=p.with_suffix(p.suffix+'.part')
  response=get('https://modelscope.cn/models/LLM-Research/Llama-3.2-1B/resolve/master/'+n,stream=True)
  with tmp.open('wb') as f:
   for chunk in response.iter_content(4*1024*1024):f.write(chunk)
  tmp.replace(p);assert verify(),('official identity mismatch',n)
 print('verified',n,flush=True)
cfg=json.loads((DEST/'config.json').read_text());assert cfg['model_type']=='llama' and cfg['hidden_size']==2048
model=dict(path=str(DEST),repo='meta-llama/Llama-3.2-1B',revision=meta['sha'],files=files,hidden_size=cfg['hidden_size'],tie_word_embeddings=cfg['tie_word_embeddings'],training_stage='pretrained_base',bos_token_id=cfg['bos_token_id'],eos_token_id=cfg['eos_token_id'])
(R/'MODEL_1B.json').write_text(json.dumps(model,indent=2)+'\n');print('MODEL VERIFIED',meta['sha'],flush=True)
