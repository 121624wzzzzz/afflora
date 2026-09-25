"""Seal the finished study; verification only reads files and independently recomputes contrasts."""
import argparse,hashlib,json,math,statistics
from datetime import datetime
from pathlib import Path
from scipy.stats import t
ROOT=Path(__file__).resolve().parent/'gemma_transfer_20260918'
EXCLUDE={'SEAL.json','SEAL_MANIFEST.json'}
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def files():return sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name not in EXCLUDE and '__pycache__' not in p.parts and p.suffix!='.pyc')
def now():return datetime.now().astimezone().isoformat()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def integrity():
 assert read(ROOT/'SCHEDULER_COMPLETE.json')['status']=='passed'
 a=read(ROOT/'FINAL_AUDIT.json');assert a['status']=='passed' and a['jobs']==114
 for key,file in [('analysis_sha256','ANALYSIS.json'),('selection_sha256','SELECTION.json'),('bootstrap_sha256','CLUSTER_BOOTSTRAP.json')]:assert a[key]==sha(ROOT/file)
 for file in ['CODE_FROZEN.json','DATA_FROZEN.json']:
  for rel,h in read(ROOT/file)['files'].items():assert sha(ROOT/rel)==h,(file,rel)
 for p,h in read(ROOT/'MODEL_IDENTITY_AUDIT.json')['files'].items():assert sha(Path(p))==h,p
 for p,h in read(ROOT/'RUNTIME.json')['native_sources'].items():assert sha(Path(p))==h,p
 reuse=read(ROOT/'SOURCE_REUSE.json')
 for root,h in reuse['source_manifests'].items():assert sha(Path(root)/'SEAL_MANIFEST.json')==h,root
 for v in reuse['copied_files'].values():assert sha(Path(v['source']))==v['sha256'],v['source']
 revision=ROOT/'PREFLIGHT_REVISION.json'
 if revision.exists():
  v=read(revision);archive=Path(v['archive']);assert sha(archive/'PREFLIGHT_ARCHIVE_MANIFEST.json')==v['archive_manifest_sha256']
  for rel,m in read(archive/'PREFLIGHT_ARCHIVE_MANIFEST.json')['files'].items():assert sha(archive/rel)==m['sha256'] and (archive/rel).stat().st_size==m['bytes']
 state=read(ROOT/'STATE.json');assert len(state['jobs'])==114 and all(j['state']=='passed' for j in state['jobs'])
 measured={};response_count=0
 for j in state['jobs']:
  s=j['spec'];name=s['name'];audit=read(ROOT/'audits'/f'{name}.json');assert audit['status']=='passed'
  if s['arm']=='hidden_budget':assert audit['extra_rank_modules_updated']==33
  for tag,h in audit['summary_sha256'].items():
   p=ROOT/'evaluations'/name/tag;assert sha(p/'SUMMARY.json')==h
   v=read(p/'SUMMARY.json');assert v['responses_sha256']==sha(p/'responses.jsonl');rs=rows(p/'responses.jsonl')
   primary=100*sum(bool(r['content_correct']) for r in rs)/len(rs);assert math.isclose(primary,v['primary'],rel_tol=0,abs_tol=1e-12)
   assert len(rs)==v['n'];response_count+=len(rs);measured[name]=primary
 assert response_count==a['responses']
 diagnostic=read(ROOT/'DEVELOPMENT_OPTIMIZATION_DIAGNOSTICS.json')
 assert diagnostic['status']=='passed' and len(diagnostic['results'])==36
 assert diagnostic['selection_sha256']==sha(ROOT/'SELECTION.json')
 expected={j['spec']['name'] for j in state['jobs'] if j['spec']['stage']=='search' and j['spec']['task']=='trec50'}
 assert {r['run'] for r in diagnostic['results']}==expected
 for r in diagnostic['results']:
  for rel,h in r['inputs'].items():assert sha(ROOT/rel)==h,rel
  responses=rows(ROOT/'evaluations'/r['run']/'dev/responses.jsonl');hist={}
  for v in responses:hist[v['predicted_code']]=hist.get(v['predicted_code'],0)+1
  assert hist==r['predicted_code_histogram'] and len(hist)==r['distinct_predicted_codes']
  assert measured[r['run']]==r['accuracy_pct']
  history=read(ROOT/'checkpoints'/r['run']/'TRAINING.json')['history']
  assert statistics.mean(v['loss'] for v in history[-8:])==r['tail8_training_loss']
  assert max(v['grad_norm'] for v in history)==r['maximum_gradient_norm']
 assert (ROOT/'DEVELOPMENT_INSTABILITY_ZH.md').exists()
 analysis=read(ROOT/'ANALYSIS.json');selection=read(ROOT/'SELECTION.json')
 for rel,h in selection['inputs'].items():assert sha(ROOT/rel)==h
 assert selection['confirmation_jobs_sha256']==sha(ROOT/'CONFIRMATION_JOBS.json')
 for task,arms in selection['scores'].items():
  for arm,candidates in arms.items():
   for c in candidates:
    actual=[measured[f'search_{task}_{arm}_{c["id"]}_s{seed}'] for seed in [7400,7401]]
    assert actual==c['seed_scores'] and sum(actual)/2==c['mean']
   chosen=max(range(6),key=lambda i:(candidates[i]['mean'],-i));assert selection['selected'][task][arm]['id']==candidates[chosen]['id']
 for c in analysis['comparisons']:
  arms=analysis['results'][c['task']];xs=arms[c['a']];ys=arms[c['b']]
  assert [r['seed'] for r in xs]==[r['seed'] for r in ys]==list(range(7500,7505))
  deltas=[measured[x['run']]-measured[y['run']] for x,y in zip(xs,ys)];assert deltas==c['paired_seed_deltas_pp']
  avg=statistics.mean(deltas);se=statistics.stdev(deltas)/math.sqrt(5);assert avg==c['mean_pp']
  for key,level in [('ci95',.975),('bonferroni4_ci95',1-.05/8)]:
   radius=t.ppf(level,4)*se;assert c[key]==[avg-radius,avg+radius]
 assert (ROOT/'FINAL_INTERPRETATION_ZH.md').exists() and (ROOT/'ALL_RESULTS.csv').exists()
 for name in ['development_search','confirmation_intervals']:
  for ext in ['pdf','png']:assert (ROOT/'figures'/f'{name}.{ext}').stat().st_size>0
 return {'jobs':114,'responses':response_count,'independently_recomputed_contrasts':len(analysis['comparisons'])}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--verify',action='store_true');args=ap.parse_args()
 if args.verify:
  seal=read(ROOT/'SEAL.json');m=read(ROOT/'SEAL_MANIFEST.json');assert seal['manifest_sha256']==sha(ROOT/'SEAL_MANIFEST.json')
  assert set(m['files'])=={str(p.relative_to(ROOT)) for p in files()}
  for rel,v in m['files'].items():assert (ROOT/rel).stat().st_size==v['bytes'] and sha(ROOT/rel)==v['sha256'],rel
  checks=integrity();print(json.dumps({'at':now(),'status':'passed','file_count':len(m['files']),'manifest_sha256':sha(ROOT/'SEAL_MANIFEST.json'),**checks}),flush=True);return
 assert not (ROOT/'SEAL.json').exists() and not (ROOT/'SEAL_MANIFEST.json').exists()
 checks=integrity()
 for name in ['seal_gemma_transfer_20260918.py','report_gemma_transfer_20260918.py','bootstrap_gemma_transfer_20260918.py','revise_gemma_preflight_20260918.py','probe_gemma_native_scores_20260918.py','download_gemma9_verified_20260918.py','download_gemma9_verified_20260918_pre_segment.py.txt','download_gemma9_verified_20260918_segment_v1.py.txt','monitor_gemma_transfer_20260918.py','gemma_transfer_20260918_download.log']:
  p=ROOT.parent/name
  if p.exists():(ROOT/'provenance'/(name+'.txt')).write_bytes(p.read_bytes())
 entries={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files()}
 write(ROOT/'SEAL_MANIFEST.json',{'at':now(),'status':'sealed','excluded':sorted(EXCLUDE)+['**/__pycache__/**','**/*.pyc'],'file_count':len(entries),'total_bytes':sum(v['bytes'] for v in entries.values()),'files':entries})
 write(ROOT/'SEAL.json',{'at':now(),'status':'sealed','manifest_sha256':sha(ROOT/'SEAL_MANIFEST.json'),'file_count':len(entries),**checks})
 print(json.dumps(read(ROOT/'SEAL.json')),flush=True)
if __name__=='__main__':main()
