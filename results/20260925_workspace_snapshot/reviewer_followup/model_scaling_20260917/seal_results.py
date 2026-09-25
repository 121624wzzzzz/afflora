"""Call after scheduler exit, diagnostics, reports, and figure review."""
from common import *

def main():
    assert not (HERE/'ARTIFACT_MANIFEST.json').exists()
    for name in ['SMOKE_COMPLETE.json','FORMAL_COMPLETE.json','EXPERIMENT_COMPLETE.json','RESULT_AUDIT.json',
                 'PARAMETER_AUDIT.json','SCORER_TESTS.json','CONTENT_DIAGNOSTICS.json','FIGURE_REVIEW.json','OPERATIONAL_RESUME_CHECK.json']:
        assert read(HERE/name)['status']=='passed',name
    state=read(HERE/'FORMAL_STATE.json')
    assert not state['pending'] and not state['active'] and not state['failed']
    assert len(state['finished'])==64
    for name in ['CODE_FROZEN.json','DATA_FROZEN.json']:
        for rel,h in read(HERE/name)['files'].items():assert sha(HERE/rel)==h,rel
    verified=[]
    for cfg in read(HERE/'models.json').values():
        for path,h in cfg['files'].items():
            assert sha(path)==h,path
            verified.append({'path':path,'sha256':h})
    audit=read(HERE/'RESULT_AUDIT.json')
    assert audit['responses']==90432 and audit['evaluations']==132 and audit['tokenizations']==13838
    assert len(read(HERE/'PARAMETER_AUDIT.json')['runs'])==68
    amendment=read(HERE/'RESOURCE_AMENDMENT.json')
    assert amendment['new_scheduler_sha256']==sha(HERE/'pipeline_shared.py')
    for name in ['FINAL_INTERPRETATION_ZH.md','METHOD_AND_LIMITS_ZH.md','README.md','QUALITATIVE_REVIEW_NOTES.md']:
        assert (HERE/name).is_file(),name
    for rel,h in read(HERE/'FIGURE_REVIEW.json')['files'].items():assert sha(HERE/rel)==h,rel
    write(HERE/'FINAL_AUDIT.json',{'at':now(),'status':'passed','model_files_rehashed':verified,
        'formal_jobs':64,'smoke_jobs':4,'responses_verified':90432,'historical_anchor_responses_reverified':37872,
        'resource_amendment_sha256':sha(HERE/'RESOURCE_AMENDMENT.json'),
        'data_manifest_sha256':sha(HERE/'DATA_FROZEN.json'),'code_manifest_sha256':sha(HERE/'CODE_FROZEN.json')})
    write(HERE/'COMPLETION.json',{'at':now(),'status':'passed','summary':'FINAL_INTERPRETATION_ZH.md','artifact_manifest':'ARTIFACT_MANIFEST.json'})
    files={str(p.relative_to(HERE)):{'bytes':p.stat().st_size,'sha256':sha(p)}
           for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='ARTIFACT_MANIFEST.json'}
    write(HERE/'ARTIFACT_MANIFEST.json',{'at':now(),'status':'sealed',
        'exclusions':['ARTIFACT_MANIFEST.json itself','__pycache__'],'files':files})
    print(canonical({'files':len(files),'manifest_sha256':sha(HERE/'ARTIFACT_MANIFEST.json')}))

if __name__=='__main__':main()
