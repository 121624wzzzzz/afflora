"""Final scientific integrity audit after scheduler and per-run auditors have exited."""
import subprocess,sys
from reporting_common import *
from scheduler_lineage import verify_scheduler_lineage
PYTHON='/home/wz/anaconda3/envs/torch24/bin/python'
def main():
 assert (ROOT/'SCHEDULER_COMPLETE.json').exists()
 state=read(ROOT/'STATE.json');assert not any(j['status'] in ['running','auditing','awaiting_audit','pending'] for j in state['jobs'].values())
 lineage=verify_scheduler_lineage(state)
 totals={};models={};frozen={}
 for phase in ['core','new_tasks']:
  path=ROOT/phase;ready=read(path/'READY.json');assert ready['status']=='passed';assert ready==state['phases'][phase]['ready']
  for kind in ['CODE','DATA']:
   mf=path/f'{kind}_FROZEN.json';assert sha(mf)==ready[kind.lower()+'_sha256']
   for rel,h in read(mf)['files'].items():assert sha(path/rel)==h,(phase,rel)
   frozen[phase+'/'+mf.name]=sha(mf)
  for cfg in read(path/'models.json').values():
   for file,h in cfg['files'].items():
    if file not in models:assert sha(file)==h,file;models[file]=h
    else:assert models[file]==h,file
  done=[j for j in state['jobs'].values() if j['phase']==phase and j['status']=='passed'];responses=sql=qa=0;formal=smoke=0
  for j in done:
   name=j['spec']['name'];audit=read(path/'audits'/f'{name}.json');assert audit['status']=='passed';responses+=audit['responses'];sql+=audit.get('official_sql_executions',0);qa+=audit.get('official_squad_predictions',0)
   assert read(path/'checkpoints'/name/'COMPLETE.json')['status']=='passed'
   assert read(path/'checkpoints'/name/'spec.json')==j['spec']
   if j['spec']['arm']!='base':
    tr=read(path/'checkpoints'/name/'TRAINING.json');assert sha(path/'checkpoints'/name/'adapter.safetensors')==tr['adapter_sha256']
   for tag,h in audit['summary_sha256'].items():
    p=path/'evaluations'/name/tag/'SUMMARY.json';assert sha(p)==h;assert sha(p.parent/'responses.jsonl')==read(p)['responses_sha256']
   if j['spec']['smoke']:smoke+=1
   else:formal+=1
  totals[phase]={'new_formal_audited':formal,'smoke_audited':smoke,'responses_independently_audited':responses,'official_sql_prediction_checks':sql,'official_squad_prediction_checks':qa,'planned_new_formal':len(read(path/'FORMAL_JOBS.json'))}
  subprocess.run([PYTHON,str(ROOT/'audit_tokenization.py'),phase],check=True)
  subprocess.run([PYTHON,str(path/'analyze.py')],cwd=path,check=True)
 subprocess.run([PYTHON,str(ROOT/'audit_pairs.py')],check=True)
 subprocess.run([PYTHON,str(ROOT/'audit_boundary_algebra.py')],check=True)
 pair=read(ROOT/'PAIR_AUDIT_PROGRESS.json');assert pair['status']=='passed'
 missing=[{'name':key,'status':j['status'],'spec':j['spec']} for key,j in state['jobs'].items() if j['status']!='passed']
 write(ROOT/'FINAL_AUDIT.json',{'at':now(),'status':'passed','all_planned_new_jobs_completed':not missing,'counts':state['counts'],'phases':totals,'frozen_manifests':frozen,'model_files_rehashed':models,'pair_audit_sha256':sha(ROOT/'PAIR_AUDIT_PROGRESS.json'),'scheduler_lineage':lineage,'uncompleted_jobs':missing,'deadline':read(ROOT/'WINDOW.json')})
 print(json.dumps({'status':'passed','totals':totals,'uncompleted_jobs':len(missing)},ensure_ascii=False))
if __name__=='__main__':main()
