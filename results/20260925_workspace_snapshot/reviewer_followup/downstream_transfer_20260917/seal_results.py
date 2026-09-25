"""Call only after the scheduler has exited and final interpretation is reviewed."""
from common import *

def main():
    assert not (HERE/'ARTIFACT_MANIFEST.json').exists()
    for name in ['SMOKE_COMPLETE.json','FORMAL_COMPLETE.json','EXPERIMENT_COMPLETE.json','RESULT_AUDIT.json','PARAMETER_AUDIT.json','SCORER_TESTS.json','TRAINING_INPUT_AUDIT.json']:
        assert read(HERE/name)['status']=='passed',name
    state=read(HERE/'FORMAL_STATE.json');assert not state['pending'] and not state['active'] and not state['failed']
    assert len(state['finished'])==64
    for name in ['CODE_FROZEN.json','DATA_FROZEN.json']:
        for rel,h in read(HERE/name)['files'].items():assert sha(HERE/rel)==h,rel
    verified=[]
    for cfg in read(HERE/'models.json').values():
        for path,h in cfg['files'].items():assert sha(path)==h;verified.append({'path':path,'sha256':h})
    assert read(HERE/'RESULT_AUDIT.json')['responses_checked']==104960
    assert (HERE/'FINAL_INTERPRETATION_ZH.md').is_file()
    write(HERE/'FINAL_AUDIT.json',{'at':now(),'status':'passed','model_files_rehashed':verified,
        'formal_jobs':64,'smoke_jobs':4,'responses_verified':104960,
        'data_manifest_sha256':sha(HERE/'DATA_FROZEN.json'),'code_manifest_sha256':sha(HERE/'CODE_FROZEN.json')})
    write(HERE/'COMPLETION.json',{'at':now(),'status':'passed','summary':'FINAL_INTERPRETATION_ZH.md','artifact_manifest':'ARTIFACT_MANIFEST.json'})
    files={str(p.relative_to(HERE)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='ARTIFACT_MANIFEST.json'}
    write(HERE/'ARTIFACT_MANIFEST.json',{'at':now(),'status':'sealed','exclusions':['ARTIFACT_MANIFEST.json itself','__pycache__'],'files':files})
    print(canonical({'files':len(files),'manifest_sha256':sha(HERE/'ARTIFACT_MANIFEST.json')}))
if __name__=='__main__':main()
