from common import HERE, CHAT, PYTHON, read

ARMS=['none','output','both','hidden_budget']
MODELS={k:v['path'] for k,v in read(CHAT/'manifest.json')['models'].items()}
def jobs(smoke=False):
    return [{'name':('smoke_' if smoke else '')+f'{model}_{arm}_hr8_sd{seed}',
        'model':model,'arm':arm,'seed':seed,'smoke':smoke,
        'checkpoint':str(HERE/'checkpoints'/(('smoke_' if smoke else '')+f'{model}_{arm}_hr8_sd{seed}'))}
        for seed in ([42] if smoke else [42,43,44]) for model in MODELS for arm in ARMS]

def variant(j):return {'none':'hidden_lora','hidden_budget':'hidden_lora',
    'output':'affine_lm_head_plus_hidden_lora','both':'affine_input_lm_head_plus_hidden_lora'}[j['arm']]

def command(j):
    result=[PYTHON,'-u',str(HERE/'train.py'),'--model-path',MODELS[j['model']],
        '--train-data',str(HERE/'data/train.jsonl'),'--output-dir',j['checkpoint'],'--variant',variant(j),
        '--hidden-lora-rank','8','--hidden-lora-alpha','16','--hidden-lora-dropout','0.05',
        '--affine-rank','16','--affine-alpha','128','--affine-dropout','0','--max-seq-len','2048',
        '--per-device-train-batch-size','8','--gradient-accumulation-steps','2','--learning-rate','5e-5',
        '--num-train-epochs','1','--lr-scheduler-type','cosine','--warmup-ratio','0.03',
        '--max-grad-norm','1','--logging-steps','10','--save-strategy','no','--master-dtype','fp32',
        '--base-dtype','bf16','--bf16','--seed',str(j['seed'])]
    if j['smoke']:result+=['--max-train-samples','32','--max-steps','2','--logging-steps','1']
    return result
