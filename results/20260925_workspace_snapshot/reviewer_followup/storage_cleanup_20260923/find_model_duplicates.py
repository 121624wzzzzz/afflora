from pathlib import Path
import json,hashlib,collections,os
root=Path(__file__).resolve().parent;base=root.parent.parent.parent
models=json.loads((root.parent/'tied_model_expansion_20260923/study/models.json').read_text())
refs={}
for c in models.values():
 for path,h in c['files'].items():
  p=Path(path)
  if p.suffix=='.safetensors':refs[str(p)]=h
by_size=collections.defaultdict(list)
for path,h in refs.items():
 p=Path(path);by_size[p.stat().st_size].append((p,h))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
plan=[];scanned=0
for p in (base/'models').rglob('*.safetensors'):
 if p.is_symlink() or str(p) in refs:continue
 st=p.stat();matches=[(q,h) for q,h in by_size.get(st.st_size,[]) if q.stat().st_dev==st.st_dev and q.stat().st_ino!=st.st_ino]
 if not matches:continue
 actual=sha(p);scanned+=1
 for q,h in matches:
  if actual==h:
   assert sha(q)==h
   plan.append(dict(path=str(p),keeper=str(q),sha256=h,bytes=st.st_size,old_inode=st.st_ino,links=st.st_nlink));break
(root/'MODEL_DUPLICATE_PLAN.json').write_text(json.dumps(dict(scanned=scanned,duplicates=plan,duplicate_GiB=sum(x['bytes'] for x in plan)/2**30),indent=2)+'\n')
print('candidate matches scanned',scanned,'verified duplicate files',len(plan),'GiB',sum(x['bytes'] for x in plan)/2**30,flush=True)
