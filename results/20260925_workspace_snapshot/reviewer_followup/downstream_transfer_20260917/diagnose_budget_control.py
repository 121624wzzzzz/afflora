"""Post hoc response to the question about budget-LoRA declines, not a primary test."""
import math
import numpy as np
from scipy.stats import t
from peft import LoraConfig,TaskType
from common import *

out={'at':now(),'scope':'post hoc descriptive budget-minus-ordinary comparison; unadjusted intervals are not new confirmatory tests','groups':[]}
cfg=LoraConfig(task_type=TaskType.CAUSAL_LM,r=8,lora_alpha=16,lora_dropout=.05,target_modules=['q_proj','k_proj','v_proj','o_proj','up_proj','down_proj','gate_proj'],bias='none')
assert not cfg.use_rslora and not cfg.use_dora and cfg.lora_alpha/cfg.r==2.0
out['runtime_lora_configuration']={'use_rslora':cfg.use_rslora,'use_dora':cfg.use_dora,'ordinary_scaling':cfg.lora_alpha/cfg.r,'budget_scaling_in_frozen_code':2.0}
for task in TASKS:
    for model in MODELS:
        seeds=[]
        for seed in range(7100,7105):
            r={'seed':seed};initial=[]
            for arm in ['hidden','hidden_budget']:
                name=f'{task}_{model}_{arm}_s{seed}';cp=HERE/'checkpoints'/name
                tr=read(cp/'TRAINING.json');a=read(cp/'INITIALIZATION.json');initial.append(a)
                r[arm]={'test_primary':read(HERE/'evaluations'/name/'test/SUMMARY.json')['primary'],
                    'mean_last8_training_loss':float(np.mean([v['loss'] for v in tr['history'][-8:]])),
                    'trainable_parameters':a['trainable_parameters']}
            assert initial[0]['shared_hidden_initialization_sha256']==initial[1]['shared_hidden_initialization_sha256']
            r['budget_minus_hidden']=r['hidden_budget']['test_primary']-r['hidden']['test_primary'];seeds.append(r)
        a=np.array([r['budget_minus_hidden'] for r in seeds]);half=float(t.ppf(.975,4)*a.std(ddof=1)/math.sqrt(5))
        out['groups'].append({'task':task,'model':model,'seeds':seeds,'mean_delta':float(a.mean()),
            'descriptive_uncorrected95_seed_interval':[float(a.mean()-half),float(a.mean()+half)]})
write(HERE/'BUDGET_CONTROL_DIAGNOSTIC.json',out)
print(canonical([{k:r[k] for k in ['task','model','mean_delta','descriptive_uncorrected95_seed_interval']} for r in out['groups']]))
