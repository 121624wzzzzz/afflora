from pathlib import Path
import os,json,hashlib
root=Path(__file__).resolve().parent
plan=json.loads((root/'MODEL_DUPLICATE_PLAN.json').read_text())['duplicates']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
changed=[];released=0
for r in plan:
 p=Path(r['path']);keeper=Path(r['keeper']);st=p.stat()
 assert not p.is_symlink() and not keeper.is_symlink() and st.st_ino==r['old_inode']
 assert st.st_dev==keeper.stat().st_dev
 assert sha(p)==sha(keeper)==r['sha256']
 tmp=p.with_name(p.name+'.verified_dedup_tmp');assert not tmp.exists();os.link(keeper,tmp);os.replace(tmp,p)
 assert p.stat().st_ino==keeper.stat().st_ino and sha(p)==r['sha256']
 if st.st_nlink==1:released+=st.st_blocks*512
 changed.append(r)
 out=root/'MODEL_DEDUP_RESULT.json';temp=out.with_suffix('.tmp');temp.write_text(json.dumps(dict(status='running',changed=changed,released_bytes=released),indent=2)+'\n');temp.replace(out)
out.write_text(json.dumps(dict(status='passed',changed=changed,released_bytes=released,note='Immutable pretrained weights with identical SHA256 share physical storage. Original paths, contents, configs and tokenizer files remain.'),indent=2)+'\n')
print('deduplicated model shards',len(changed),'freed GiB',released/2**30,flush=True)
