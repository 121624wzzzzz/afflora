from common import *

def main():
    assert not (HERE/'ARTIFACT_MANIFEST.json').exists()
    audit=read(HERE/'FINAL_AUDIT.json');assert audit['status']=='passed' and audit['job_count']==64
    assert read(HERE/'CONFIRMATION_COMPLETE.json')['jobs']==40
    assert read(HERE/'TUNING_COMPLETE.json')['jobs']==16
    assert read(HERE/'SMOKE_COMPLETE.json')['jobs']==8
    assert read(HERE/'NUMERIC_TIES_SUMMARY.json')['rechecked_rows']==21
    for p,h in read(HERE/'FROZEN_PROTOCOL.json')['sha256'].items():assert sha(p)==h,p
    files=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.tmp')]
    assert not list(HERE.rglob('FAILED.json'))
    manifest={'status':'complete','sealed_at':now(),'file_count':len(files),
        'sha256':{str(p):sha(p) for p in sorted(files)},
        'scope':'All final artifacts except this manifest, transient tmp and Python bytecode caches.',
        'upstream_inputs':{'source':read(HERE/'REUSE_AUDIT.json'),'base_models':read(HERE/'DATA_AUDIT.json')['model_files_verified']}}
    write(HERE/'ARTIFACT_MANIFEST.json',manifest)
    print(__import__('json').dumps({'status':'sealed','files':len(files),'at':manifest['sealed_at']}))

if __name__=='__main__':main()
