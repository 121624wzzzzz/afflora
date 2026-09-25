"""Lossless deduplication of byte-identical, finalized initial checkpoints."""
from pathlib import Path
import json,hashlib,os,collections,datetime
root=Path(__file__).resolve().parent;study=root.parent/'boundary_lora_closure_20260922/study'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(name,obj):
 p=root/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2)+'\n');t.replace(p)
assert json.loads((study/'STATE.json').read_text())['stage']=='complete'
audit=json.loads((study/'FINAL_AUDIT.json').read_text());assert audit['status']=='passed' and sha(study/'OUTPUT_MANIFEST.json')==audit['manifest_sha256']
manifest=json.loads((study/'OUTPUT_MANIFEST.json').read_text())['files'];groups=collections.defaultdict(list)
for rel,h in manifest.items():
 if rel.endswith('/initial_adapter.safetensors'):groups[h].append(study/rel)
plan=[]
for h,paths in groups.items():
 if len(paths)<2:continue
 keeper=paths[0];assert not keeper.is_symlink() and sha(keeper)==h
 for path in paths[1:]:
  assert not path.is_symlink() and sha(path)==h
  a,b=keeper.stat(),path.stat();assert a.st_dev==b.st_dev
  if a.st_ino==b.st_ino:continue
  plan.append(dict(keeper=str(keeper),path=str(path),sha256=h,bytes=b.st_size,old_inode=b.st_ino,old_links=b.st_nlink))
save('DEDUP_PLAN.json',dict(at=datetime.datetime.now().isoformat(),files=plan,logical_duplicate_bytes=sum(x['bytes'] for x in plan)))
changed=[];released=0
for row in plan:
 keeper=Path(row['keeper']);path=Path(row['path']);before=path.stat();assert before.st_ino==row['old_inode']
 temp=path.with_name(path.name+'.dedup_link_tmp');assert not temp.exists();os.link(keeper,temp);os.replace(temp,path)
 assert path.stat().st_ino==keeper.stat().st_ino and sha(path)==row['sha256']
 if before.st_nlink==1:released+=before.st_blocks*512
 changed.append(row);save('DEDUP_RESULT.json',dict(status='running',changed=changed,released_bytes=released,note='All original paths and byte hashes preserved. Hardlinked frozen files must stay immutable.'))
save('DEDUP_RESULT.json',dict(status='passed',changed=changed,released_bytes=released,manifest_sha256=sha(study/'OUTPUT_MANIFEST.json'),note='No experiment content removed; finalized byte-identical initial files now share storage. Each original path rehashed after replacement.'))
print('deduplicated',len(changed),'files; released GiB',released/2**30,flush=True)
