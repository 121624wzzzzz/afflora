import ast,platform,sys,importlib.metadata,inspect
from common import *
from design import make
def main():
 assert read(HERE/'DATA_AUDIT.json')['status']=='passed' and read(HERE/'MODEL_IDENTITY_AUDIT.json')['status']=='passed'
 assert read(HERE/'MODEL_FORMULA_AUDIT.json')['status']=='passed'
 assert read(HERE/'CHECKPOINT_RUNTIME_AUDIT.json')['status']=='passed'
 assert read(HERE/'ACCELERATED_KERNEL_AUDIT.json')['status']=='passed'
 assert read(HERE/'INITIAL_REPEAT_GATE.json')['status']=='passed'
 assert not (HERE/'CODE_FROZEN.json').exists() and not (HERE/'checkpoints').exists()
 make();files={}
 for p in sorted(HERE.glob('*.py')):ast.parse(p.read_text());files[p.name]=sha(p)
 for p in sorted((HERE/'source').glob('*.py')):files[str(p.relative_to(HERE))]=sha(p)
 for n in ['models.json','BUDGET_PLAN.json','BUDGET_ALLOCATION.json','PROTOCOL.md','CANDIDATES.json','SEARCH_JOBS.json','SMOKE_JOBS.json','BASE_JOBS.json',
           'SOURCE_REUSE.json','DATA_AUDIT.json','DATA_FROZEN.json','MODEL_IDENTITY_AUDIT.json','MODEL_FORMULA_AUDIT.json','CHECKPOINT_RUNTIME_AUDIT.json','ACCELERATED_KERNEL_AUDIT.json','TECHNICAL_CORRECTIONS.json','INITIAL_REPEAT_GATE.json','DISPATCHER_IDENTITY.json']:
  files[n]=sha(HERE/n)
 if (HERE/'PREFLIGHT_REVISION.json').exists():files['PREFLIGHT_REVISION.json']=sha(HERE/'PREFLIGHT_REVISION.json')
 from transformers.loss.loss_utils import ForCausalLMLoss
 native_sources=read(HERE/'MODEL_FORMULA_AUDIT.json')['native_sources'];p=inspect.getfile(ForCausalLMLoss);native_sources[p]=sha(p)
 import fla,causal_conv1d
 for package in [fla,causal_conv1d]:
  for root in package.__path__:
   for q in Path(root).rglob('*'):
    if q.suffix in ['.py','.so']:native_sources[str(q)]=sha(q)
 import causal_conv1d_cuda
 q=Path(causal_conv1d_cuda.__file__);native_sources[str(q)]=sha(q)
 write(HERE/'RUNTIME.json',{'at':now(),'python':sys.version,'platform':platform.platform(),
   'packages':{p:importlib.metadata.version(p) for p in ['torch','transformers','peft','safetensors','numpy','scipy','babel','matplotlib','triton','flash-linear-attention','fla-core','causal-conv1d']},
   'native_sources':native_sources})
 files['RUNTIME.json']=sha(HERE/'RUNTIME.json')
 write(HERE/'CODE_FROZEN.json',{'at':now(),'files':files})
 write(HERE/'READY.json',{'at':now(),'status':'passed','code_frozen_sha256':sha(HERE/'CODE_FROZEN.json'),'data_frozen_sha256':sha(HERE/'DATA_FROZEN.json')})
 print('frozen',len(files),'files',sha(HERE/'CODE_FROZEN.json'),flush=True)
if __name__=='__main__':main()
