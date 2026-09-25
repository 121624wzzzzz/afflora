"""Verified fresh copies for four sizes under one explicit numerical policy."""
import json,hashlib,shutil,ast
from pathlib import Path
from datetime import datetime
PARENT=Path(__file__).resolve().parent
MASTER=PARENT/'qwen35_fixed_multiscale_20260920'
KEYS=[('08b','0.8B'),('2b','2B'),('4b','4B'),('9b','9B')]
def read(p):return json.loads(p.read_text())
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def edit(p,a,b):
 s=p.read_text();assert a in s,(p,a);p.write_text(s.replace(a,b))
REFERENCE='''"""Training-only, zero-update first-batch references for every admitted seed."""
import gc,torch
from common import *
from modeling import build,fixed_runtime_audit
from run import preflight
from initial_repeat import initial_repeat
SEEDS=[7599,7600,7601,7700,7701,7702,7703,7704]
def main():
 assert not (HERE/'INITIAL_REFERENCES.json').exists();records={}
 for seed in SEEDS:
  spec=dict(seed=seed,model=MODEL,arm='hidden');model,init=build(spec)
  for task in TASKS:
   data=read(HERE/f'tokens/{task}_{MODEL}_train.json');pf=preflight(model,spec,data)
   repeat=initial_repeat(model,data,seed,2);assert repeat['status']=='passed' and repeat['gradients_bitwise_equal']
   assert repeat['microbatch_losses'][0]==repeat['microbatch_losses'][1]
   records[f'{task}:{seed}']=dict(task=task,seed=seed,reference_arm='hidden',repeat=repeat,preflight=pf,initialization=init)
   write(HERE/'INITIAL_REFERENCES_PROGRESS.json',dict(records=records));print(task,seed,repeat['losses'][0],flush=True)
  del model;gc.collect();torch.cuda.empty_cache()
 write(HERE/'INITIAL_REFERENCES.json',dict(at=now(),status='passed',records=records,numerical_policy=fixed_runtime_audit(),
  scope='Training examples only; no optimizer updates or dev/confirm scoring. All arms must exactly match the 16 reference microbatch losses before any update.'))
if __name__=='__main__':main()
'''
CHECK='''"""Enforce independent-process references before any optimizer update."""
from common import HERE,read,sha
def check_reference(repeat,task,seed):
 path=HERE/'INITIAL_REFERENCES.json';refs=read(path);assert refs['status']=='passed'
 ref=refs['records'][f'{task}:{seed}']['repeat']
 assert repeat['status']=='passed' and repeat['gradients_bitwise_equal']
 assert repeat['loss_absolute_difference']==0 and repeat['gradient_relative_rms']==0
 assert repeat['order_sha256']==ref['order_sha256']
 for values in repeat['microbatch_losses']:assert values==ref['microbatch_losses'][0],(task,seed,values,ref['microbatch_losses'][0])
 return dict(status='passed',reference_sha256=sha(path),reference_key=f'{task}:{seed}',exact_microbatch_match=True,
  scope='Independent-process first-batch loss parity only; not proof of a whole-training-trajectory invariant.')
'''
def main():
 assert read(PARENT/'qwen35_multiscale_20260919/PARTIAL_ARCHIVES.json')['status']=='passed'
 old=PARENT/'qwen35_transfer_20260919';assert sha(old/'SEAL_MANIFEST.json')=='35af1951cb22c7c414927a847b3aa7848cc2ce0be052633d6b6b052543f114d9'
 oldfiles=read(old/'SEAL_MANIFEST.json')['files'];template=PARENT/'qwen35_08b_20260919'
 templatefiles=read(template/'PARTIAL_ARCHIVE_MANIFEST.json')['files']
 numerics=PARENT/'qwen35_numerics_20260920'
 pilots={}
 for arm in ['budget','both']:
  pair=[read(numerics/f'fixed_{arm}_gpu{g}/RESULTS.json') for g in [5,6]]
  keys=['step','loss','grad_norm','gradients_sha256','adapter_sha256']
  assert all(r['status']=='passed' and len(r['records'])==8 for r in pair)
  assert [{k:r[k] for k in keys} for r in pair[0]['records']]==[{k:r[k] for k in keys} for r in pair[1]['records']]
  pilots[arm]=[sha(numerics/f'fixed_{arm}_gpu{g}/RESULTS.json') for g in [5,6]]
 MASTER.mkdir(exist_ok=False)
 for key,size in KEYS:
  root=PARENT/f'qwen35_fixed_{key}_20260920';root.mkdir(exist_ok=False);copied={}
  def copy(src,rel,expected):
   assert sha(src)==expected,(src,'source mismatch');dst=root/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);assert sha(dst)==expected
   copied[rel]=dict(source=str(src),sha256=expected)
  for p in sorted(template.glob('*.py')):copy(p,p.name,templatefiles[p.name]['sha256'])
  for rel,v in oldfiles.items():
   if rel.startswith(('data/','raw/','source/')) or rel in ['provenance/reference_sql_common.py.txt','provenance/reference_trec_common.py.txt']:
    copy(old/rel,rel,v['sha256'])
  src=old if key=='4b' else PARENT/f'qwen35_{key}_20260919'
  sf=oldfiles if key=='4b' else read(src/'PARTIAL_ARCHIVE_MANIFEST.json')['files']
  for rel in ['models.json','MODEL_IDENTITY_AUDIT.json',f'provenance/qwen35_{key}_official_metadata.json']:
   copy(src/rel,rel,sf[rel]['sha256'])
  for path,h in read(root/'MODEL_IDENTITY_AUDIT.json')['files'].items():assert sha(Path(path))==h,path
  write(root/'SOURCE_REUSE.json',dict(status='passed',copied_files=copied,
   source_manifests={str(old):sha(old/'SEAL_MANIFEST.json'),str(template):sha(template/'PARTIAL_ARCHIVE_MANIFEST.json'),str(src):sha(src/('SEAL_MANIFEST.json' if key=='4b' else 'PARTIAL_ARCHIVE_MANIFEST.json'))},
   scope='Only source/data/scorers and verified pretrained model identities copied. Retokenize all data. No trained adapters, predictions, selection, runtime gates or numerical references reused.'))
  edit(root/'common.py',"MODEL='qwen35_08b_base'",f"MODEL='qwen35_{key}_base'")
  edit(root/'prepare.py','Qwen3.5-0.8B-Base',f'Qwen3.5-{size}-Base')
  edit(root/'prepare.py','qwen35_08b_official',f'qwen35_{key}_official')
  shutil.copyfile(numerics/'fixed_runtime.py',root/'fixed_runtime.py')
  (root/'initial_references.py').write_text(REFERENCE);(root/'reference_check.py').write_text(CHECK)
  edit(root/'modeling.py','import sys,hashlib',"from fixed_runtime import install as install_fixed_runtime,audit as fixed_runtime_audit\ninstall_fixed_runtime()\nimport sys,hashlib")
  edit(root/'run.py',"audit['initial_repeat_sha256']=sha(root/'INITIAL_REPEAT.json')", "audit['initial_repeat_sha256']=sha(root/'INITIAL_REPEAT.json')\n        from reference_check import check_reference\n        audit['cross_process_initial_reference']=check_reference(repeat,spec['task'],spec['seed'])")
  edit(root/'run.py',"audit['frozen_before']=frozen_digest(model);", "audit['numerical_policy']=fixed_runtime_audit()\n    audit['frozen_before']=frozen_digest(model);")
  edit(root/'run.py',"write(root/'COMPLETE.json',{'at':now(),'status':'passed'})", "write(root/'NUMERICAL_POLICY_FINAL.json',fixed_runtime_audit());write(root/'COMPLETE.json',{'at':now(),'status':'passed'})")
  edit(root/'run.py',"'seconds':time.monotonic()-started})\n    if spec.get('smoke'):", "'seconds':time.monotonic()-started,'numerical_policy':fixed_runtime_audit()})\n    if spec.get('smoke'):")
  edit(root/'audit_one.py',"assert abs(tr['history'][0]['loss']-repeat['losses'][0])<2e-4", "assert tr['history'][0]['loss']==repeat['losses'][0]\n        from reference_check import check_reference\n        assert check_reference(repeat,task,spec['seed'])==init['cross_process_initial_reference']\n        assert tr['numerical_policy']['policy']==init['numerical_policy']['policy']")
  edit(root/'audit_one.py',"extra_rank_checks=0", "policy=read(root/'NUMERICAL_POLICY_FINAL.json');expected=read(HERE/'INITIAL_REFERENCES.json')['numerical_policy']\n    for record in [policy,init['numerical_policy']]:\n        assert record['policy']==expected['policy'] and record['torch_deterministic_algorithms'] and not record['tf32'] and not record['reduced_precision_gemm_reduction']\n    extra_rank_checks=0")
  edit(root/'test_initial_repeat.py',"assert r['status']=='passed',r", "assert r['status']=='passed',r\n   from reference_check import check_reference\n   records[-1]['cross_process_initial_reference']=check_reference(r,task,7600)")
  edit(root/'test_initial_repeat.py',"assert max(losses)-min(losses)<2e-4", "assert max(losses)==min(losses)")
  edit(root/'test_initial_repeat.py',"'cross_arm_first_loss_tolerance':2e-4", "'cross_arm_first_loss_tolerance':0,'independent_process_reference_required':True")
  edit(root/'freeze.py',"assert read(HERE/'INITIAL_REPEAT_GATE.json')['status']=='passed'", "assert read(HERE/'INITIAL_REPEAT_GATE.json')['status']=='passed'\n assert read(HERE/'INITIAL_REFERENCES.json')['status']=='passed'")
  edit(root/'freeze.py',"'INITIAL_REPEAT_GATE.json','DISPATCHER_IDENTITY.json'", "'INITIAL_REPEAT_GATE.json','INITIAL_REFERENCES.json','NUMERICAL_POLICY_SPEC.json','DISPATCHER_IDENTITY.json'")
  edit(root/'analyze.py','bonferroni12','bonferroni16');edit(root/'analyze.py','1-.05/24','1-.05/32')
  edit(root/'analyze.py','[100*.05/24,100*(1-.05/24)]','[100*.05/32,100*(1-.05/32)]') if '[100*.05/24,100*(1-.05/24)]' in (root/'analyze.py').read_text() else edit(root/'analyze.py','100*.05/24','100*.05/32')
  edit(root/'analyze.py',"'family':12","'family':16");edit(root/'analyze.py','Family-12','Family-16');edit(root/'analyze.py','Qwen3.5-0.8B',f'Qwen3.5-{size}')
  write(root/'NUMERICAL_POLICY_SPEC.json',dict(status='specified_before_corrected_fits',prototype_sha256=sha(numerics/'fixed_runtime.py'),
   controlled_reproduction_sha256=sha(numerics/'CONTROLLED_REPRODUCTION.json'),cross_gpu_8_step_pilots=pilots,
   policy='Process-local fixed official FLA configurations; PyTorch deterministic algorithms; fixed CUBLAS workspace; TF32 and reduced-precision GEMM reductions disabled.',
   admission='All 16 first-batch micro-losses must exactly match independently generated training-only references before any update. Two repeated backward passes require exact gradient parity.',
   scope='Not a universal deterministic training guarantee. Uniform policy across all four sizes and all arms. Historical outcomes retained separately.'))
  write(root/'TECHNICAL_CORRECTIONS.json',dict(status='specified_before_corrected_fits',changes=['Common explicit numerical policy for 0.8B/2B/4B/9B.',
    'Fresh all-arm rerun; no old fit reused. First-batch cross-process reference required before updates.','Family of 16 comparisons across all four corrected sizes.'],
    retained='Same benchmark data/prompts, exact-budget architecture rule, six LRs, 2 selection seeds, 5 confirmation seeds, 2048 examples and 64 updates.'))
  protocol=(template/'PROTOCOL.md').read_text()
  start=protocol.index('Official pretrained-only');end=protocol.index('Verified copies')
  body=protocol[start:].replace('Qwen3.5-0.8B',f'Qwen3.5-{size}')
  body=body.replace('114 jobs per size, 342 new jobs total.','114 jobs per size, 456 corrected-policy jobs total.')
  body=body.replace('Twelve primary contrasts across the THREE new sizes','Sixteen primary contrasts across the FOUR corrected sizes').replace('family-12','family-16').replace('Four-B results remain historical family4.','The earlier 4B study remains a separate historical result, not an independent replication.')
  body=body.replace('Before freezing: formula, actual checkpoint, hybrid kernel, memory and repeated\ninitial forward/backward technical probes. Known 4B BF16 first-loss variation\nmust be reported; passing a local repeat does not prove bitwise reproducibility\nacross kernels/devices. Numerical failures halt admission for diagnosis.',
   'Before freezing: fresh formula, actual checkpoint, hybrid-kernel and memory probes. Generate first-batch references for all eight smoke/search/confirmation seeds on training data only, no updates. A separate process verifies all three arms for both tasks at seed7600 against these references. Every admitted non-Base job must exactly match its 16 reference micro-losses before any optimizer update and pass exact local gradient repetition. Numerical failures halt admission and drain active workers for diagnosis. This does not prove full-trajectory cross-device determinism.')
  heading=f'''# Qwen3.5-{size}-Base: corrected four-size numerical-policy study

Four sizes 0.8B/2B/4B/9B are rerun with one common numerical policy after a
controlled zero-update intervention exactly reproduced an old first-loss
discrepancy by changing an admissible FLA L2-normalization kernel configuration.
The old job did not log its chosen configuration; the intervention establishes
a sufficient mechanism, not the historical kernel choice or the cause of task
accuracy differences. No partial old outcome is spliced into this study.
The prior 4B result and all incomplete old-policy fits are retained.

fixed_runtime.py pins 12 official FLA autotuners to architecture/data/score-
independent admissible configurations in each process. No installed packages
are modified. CUBLAS_WORKSPACE_CONFIG=:4096:8 is set before startup; deterministic
PyTorch algorithms and cuDNN are enabled, TF32 and reduced-precision GEMM
reductions disabled. This policy changes multiple numerical settings, so any
effect difference from old results cannot be attributed solely to L2 norm.
NUMERICAL_POLICY_SPEC.json records the controlled experiment and eight-step
cross-GPU pilots. All formal choices remain fixed before corrected task fits.

'''
  (root/'PROTOCOL.md').write_text(heading+body)
  for p in root.glob('*.py'):ast.parse(p.read_text())
  print('configured verified fresh study',key,'copied',len(copied),flush=True)
 write(MASTER/'PLAN.json',dict(at=datetime.now().astimezone().isoformat(),status='preparing',models=[s for k,s in KEYS],
  jobs=456,family=16,selection_seeds=[7600,7601],confirmation_seeds=list(range(7700,7705)),tasks=['wikisql','trec50'],
  comparison='H, exact-budget H, H+E+U; all old-policy results retained separately; no outcome reuse',
  numerical_policy='fixed FLA configs, explicit deterministic Torch math, exact cross-process first-batch admission',
  resource_rule='One study GPU worker per eligible GPU with at least 68 GiB free. Other workloads untouched.'))
if __name__=='__main__':main()
