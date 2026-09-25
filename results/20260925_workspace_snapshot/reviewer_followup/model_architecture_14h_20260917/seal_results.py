"""Seal the completed, audited time-bounded study. Run only after all reports."""
import fcntl
from reporting_common import *
from scheduler_lineage import verify_scheduler_lineage

EXCLUDED={'SEAL_MANIFEST.json','SEAL.json'}
def files():
    result=[]
    for p in ROOT.rglob('*'):
        if '__pycache__' in p.parts or p.suffix=='.pyc':continue
        if p.is_symlink():raise AssertionError('Unexpected symlink: '+str(p))
        if not p.is_file() or p.relative_to(ROOT).as_posix() in EXCLUDED:continue
        assert not p.name.endswith('.tmp'),p
        result.append(p)
    return sorted(result)

def main():
    assert not (ROOT/'SEAL_MANIFEST.json').exists() and not (ROOT/'SEAL.json').exists(),'Already sealed; do not rewrite.'
    lock=(ROOT/'scheduler.lock').open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    scheduler=read(ROOT/'SCHEDULER_COMPLETE.json');state=read(ROOT/'STATE.json')
    assert scheduler['counts']==state['counts']
    assert not any(j['status'] in ['pending','running','auditing','awaiting_audit'] for j in state['jobs'].values())
    required=['FINAL_AUDIT.json','REPORT_STATUS.json','FIGURE_REVIEW.json','PAIR_AUDIT_PROGRESS.json','DIAGNOSTICS_STATUS.json',
              'CORE_FINAL_TOKEN_AUDIT.json','NEW_TASKS_FINAL_TOKEN_AUDIT.json','BOUNDARY_ALGEBRA_AUDIT.json','TREC_ERROR_ANALYSIS.json']
    for name in required:assert read(ROOT/name)['status']=='passed',name
    for name in ['FINAL_INTERPRETATION_ZH.md','CORE_MAIN_ARMS_DESCRIPTIVE.json','CORE_CONTRASTS.md','NEW_TASK_CONTRASTS.md','ALL_AVAILABLE_TEST_RUNS.csv','README.md','RUN_HANDOFF.md','TREC_ERROR_ANALYSIS.md','TREC_ERROR_ANALYSIS_CLASSES.csv']:
        assert (ROOT/name).stat().st_size>0,name
    assert verify_scheduler_lineage(state)==read(ROOT/'FINAL_AUDIT.json')['scheduler_lineage']
    for phase in ['core','new_tasks']:
        path=ROOT/phase;ready=read(path/'READY.json');assert ready==state['phases'][phase]['ready']
        for kind in ['CODE','DATA']:
            mf=path/f'{kind}_FROZEN.json';assert sha(mf)==ready[kind.lower()+'_sha256']
            for rel,h in read(mf)['files'].items():assert sha(path/rel)==h,(phase,rel)
    review=read(ROOT/'FIGURE_REVIEW.json')
    assert len(review['reviewed_pngs'])==4
    for rel,h in review['reviewed_pngs'].items():assert sha(ROOT/rel)==h,rel
    for rel,h in read(ROOT/'figures/PLOT_DATA_STATUS.json')['files'].items():assert sha(ROOT/'figures'/rel)==h,rel
    final=read(ROOT/'FINAL_AUDIT.json');report=read(ROOT/'REPORT_STATUS.json')
    mains=read(ROOT/'CORE_MAIN_ARMS_DESCRIPTIVE.json')['conditions'];assert len(mains)==16
    assert sum(r['complete_main_arms'] for r in mains)==report['core_main_three_seed_blocks']
    core_conditions=read(ROOT/'core/TEST_ANALYSIS.json')['conditions']
    assert sum(c['complete'] and len(c['seeds'])==3 for c in core_conditions)==report['core_complete_three_seed_blocks']
    assert sum(c['complete'] and len(c['seeds'])==5 for c in core_conditions)==report['core_complete_five_seed_blocks']
    assert report['core_complete_three_seed_blocks']<=report['core_main_three_seed_blocks']
    assert sum(c['complete'] for c in read(ROOT/'new_tasks/TEST_ANALYSIS.json')['conditions'])==report['new_task_complete_three_seed_blocks']
    assert len(final['uncompleted_jobs'])==report['uncompleted_jobs']
    assert sum(j['status']!='passed' for j in state['jobs'].values())==report['uncompleted_jobs']
    write(ROOT/'COMPLETION.json',{'at':now(),'status':'audited_time_bounded_study_complete',
          'all_planned_jobs_completed':scheduler['all_planned_jobs_completed'],
          'reports_ready_within_requested_deadline':datetime.now(timezone.utc)<=datetime.fromisoformat(read(ROOT/'WINDOW.json')['deadline_utc']),
          'counts':state['counts'],'coverage':report,
          'audit_artifacts':{name:sha(ROOT/name) for name in required},
          'scope':'Completion refers to final audit/report/seal of the time-bounded study. Uncompleted planned jobs, if any, remain explicitly listed.'})
    records={}
    selected=files()
    for i,p in enumerate(selected):
        before=p.stat();h=sha(p);after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),p
        records[p.relative_to(ROOT).as_posix()]={'bytes':after.st_size,'sha256':h}
        if (i+1)%500==0:print(f'Hashed {i+1}/{len(selected)} files',flush=True)
    assert selected==files(),'Archive file set changed during sealing'
    write(ROOT/'SEAL_MANIFEST.json',{'at':now(),'status':'sealed','excluded':['SEAL_MANIFEST.json','SEAL.json','**/__pycache__/**','**/*.pyc'],
          'file_count':len(records),'total_bytes':sum(v['bytes'] for v in records.values()),'files':records})
    manifest_hash=sha(ROOT/'SEAL_MANIFEST.json')
    write(ROOT/'SEAL.json',{'at':now(),'status':'sealed','manifest_sha256':manifest_hash,
          'file_count':len(records),'all_planned_jobs_completed':scheduler['all_planned_jobs_completed'],
          'within_requested_deadline':datetime.now(timezone.utc)<=datetime.fromisoformat(read(ROOT/'WINDOW.json')['deadline_utc'])})
    print(json.dumps({'status':'sealed','files':len(records),'manifest_sha256':manifest_hash}))

if __name__=='__main__':main()
