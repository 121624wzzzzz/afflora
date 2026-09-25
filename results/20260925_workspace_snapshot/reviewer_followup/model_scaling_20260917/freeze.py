from common import *
assert not (HERE/'checkpoints').exists()
assert read(HERE/'MODEL_IDENTITY_AUDIT.json')['status']=='passed'
assert read(HERE/'DATA_AND_REUSE_AUDIT.json')['status']=='passed'
assert read(HERE/'SCORER_TESTS.json')['status']=='passed'
files={str(p.relative_to(HERE)):sha(p) for folder in ['data','tokens','anchors','raw','provenance'] for p in (HERE/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
write(HERE/'DATA_FROZEN.json',{'at':now(),'files':files})
files={str(p.relative_to(HERE)):sha(p) for p in list(HERE.glob('*.py'))+list((HERE/'source').glob('*.py'))+[HERE/'PROTOCOL.md',HERE/'BUDGET_PLAN.json',HERE/'models.json',HERE/'SOURCE_REUSE.json',HERE/'ANCHOR_INDEX.json',HERE/'MODEL_IDENTITY_AUDIT.json',HERE/'ANCHOR_MODEL_IDENTITY.json'] if p.is_file()}
write(HERE/'CODE_FROZEN.json',{'at':now(),'files':files})
import torch,transformers,peft,safetensors,platform
write(HERE/'ENVIRONMENT.json',{'at':now(),'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,'safetensors':safetensors.__version__})
print('data/code frozen',sha(HERE/'DATA_FROZEN.json'),sha(HERE/'CODE_FROZEN.json'))
