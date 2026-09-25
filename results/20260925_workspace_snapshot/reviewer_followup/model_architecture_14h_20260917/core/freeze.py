from common import *
assert not (HERE/'checkpoints').exists()
for file in ['MODEL_IDENTITY_AUDIT.json','DATA_AND_REUSE_AUDIT.json','SCORER_TESTS.json']:assert read(HERE/file)['status']=='passed'
files={str(p.relative_to(HERE)):sha(p) for folder in ['data','tokens','anchors','raw','provenance'] for p in (HERE/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
write(HERE/'DATA_FROZEN.json',{'at':now(),'files':files})
code=list(HERE.glob('*.py'))+list((HERE/'source').glob('*.py'))+[HERE/x for x in ['PROTOCOL.md','models.json','BUDGET_PLAN.json','FORMAL_JOBS.json','SMOKE_JOBS.json','INVENTORY.json','ANCHOR_INDEX.json','SOURCE_REUSE.json','MODEL_IDENTITY_AUDIT.json']]
write(HERE/'CODE_FROZEN.json',{'at':now(),'files':{str(p.relative_to(HERE)):sha(p) for p in code}})
import torch,transformers,peft,platform
write(HERE/'ENVIRONMENT.json',{'at':now(),'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__})
write(HERE/'READY.json',{'at':now(),'status':'passed','code_sha256':sha(HERE/'CODE_FROZEN.json'),'data_sha256':sha(HERE/'DATA_FROZEN.json')})
print('Frozen and ready',sha(HERE/'CODE_FROZEN.json'),sha(HERE/'DATA_FROZEN.json'))
