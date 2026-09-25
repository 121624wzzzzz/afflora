import ast,platform,sys
from common import *
from design import make

def main():
    assert read(HERE/'DATA_AUDIT.json')['status']=='passed'
    assert not (HERE/'CODE_FROZEN.json').exists() and not (HERE/'checkpoints').exists()
    make()
    files={}
    for p in sorted(HERE.glob('*.py')):
        ast.parse(p.read_text());files[p.name]=sha(p)
    for p in sorted((HERE/'source').glob('*.py')):files[str(p.relative_to(HERE))]=sha(p)
    for n in ['models.json','BUDGET_PLAN.json','PROTOCOL.md','CANDIDATES.json','SEARCH_JOBS.json','SMOKE_JOBS.json','COMPATIBILITY_JOBS.json','SOURCE_REUSE.json','PREFLIGHT_REVISION.json','DATA_AUDIT.json','DATA_FROZEN.json']:
        files[n]=sha(HERE/n)
    write(HERE/'ENVIRONMENT.json',{'at':now(),'python':sys.version,'platform':platform.platform()})
    write(HERE/'CODE_FROZEN.json',{'at':now(),'files':files})
    write(HERE/'READY.json',{'at':now(),'status':'passed','code_frozen_sha256':sha(HERE/'CODE_FROZEN.json'),'data_frozen_sha256':sha(HERE/'DATA_FROZEN.json')})
    print('frozen',len(files),'files',sha(HERE/'CODE_FROZEN.json'),flush=True)
if __name__=='__main__':main()
