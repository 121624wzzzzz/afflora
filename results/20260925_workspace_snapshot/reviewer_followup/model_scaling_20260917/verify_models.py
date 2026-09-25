"""Pin official revisions; verify weights against HF LFS SHA256, small files against git blobs."""
import hashlib,json,pathlib,requests,subprocess
HERE=pathlib.Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,a):p.write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n')
def main():
 results={};details=[];session=requests.Session();session.trust_env=False
 for size in [3,7]:
  meta=json.loads((HERE/f'provenance/qwen25_{size}b_api.json').read_text());revision=meta['sha'];repo=f'Qwen/Qwen2.5-{size}B'
  dest=HERE/'verified_models'/f'Qwen2.5-{size}B';dest.mkdir(parents=True,exist_ok=True)
  local=HERE.parents[2]/'models'/f'Qwen2.5-{size}B-Base';files={}
  wanted={'config.json','generation_config.json','tokenizer_config.json','tokenizer.json','merges.txt','vocab.json','model.safetensors.index.json','README.md','LICENSE'}
  for record in meta['siblings']:
   name=record['rfilename']
   if name not in wanted and not name.endswith('.safetensors'):continue
   target=dest/name;old=local/name
   if name.endswith('.safetensors'):
    expected=record['lfs']['sha256'];actual=sha(old);assert actual==expected,(old,'local weight differs from official pinned Base',actual,expected)
    if not target.exists():subprocess.run(['cp','--reflink=auto','--preserve=mode,timestamps',str(old),str(target)],check=True)
    assert sha(target)==expected
    details.append({'repo':repo,'revision':revision,'file':name,'official_lfs_sha256':expected,'local_path':str(old),'verified_copy':str(target),'local_matches_official':True})
   else:
    url=f'https://huggingface.co/{repo}/resolve/{revision}/{name}';response=session.get(url,timeout=60);response.raise_for_status();data=response.content
    if record.get('lfs'):assert hashlib.sha256(data).hexdigest()==record['lfs']['sha256']
    else:assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==record['blobId'],name
    target.write_bytes(data);details.append({'repo':repo,'revision':revision,'file':name,'url':url,'git_blob':record['blobId'],'local_matches_official':old.is_file() and old.read_bytes()==data})
   files[str(target)]=sha(target);print(repo,name,'verified',flush=True)
  cfg=json.loads((dest/'config.json').read_text());assert cfg['eos_token_id']==151643
  results[f'qwen25_{size}b_base']={'path':str(dest),'repo':repo,'revision':revision,'hidden_size':cfg['hidden_size'],'files':files,'tie_word_embeddings':cfg['tie_word_embeddings']}
 write(HERE/'models.json',results);write(HERE/'MODEL_IDENTITY_AUDIT.json',{'status':'passed','models':results,'files':details})
 print('All official Base files verified',sum(len(x['files']) for x in results.values()),flush=True)
if __name__=='__main__':main()
