"""Two fixed training demonstrations probe prompt/format sensitivity of Base.
Separate diagnostic, never substituted for an adapter's matched zero-shot control.
"""
import os,subprocess,time
from common import *
from transformers import AutoTokenizer

def main():
    assert not (HERE/'FEWSHOT_FROZEN.json').exists()
    train=rows(HERE/'data/cluener_pilot_train.jsonl');dev=rows(HERE/'data/cluener_pilot_dev.jsonl');models=read(HERE/'models.json')
    examples=train[:2];context=''.join('Example '+str(i+1)+':\n'+prompt(r)+canonical(r['target'])+'\n\n' for i,r in enumerate(examples))
    write_rows(HERE/'data/cluener_fewshot_dev.jsonl',dev);specs=[];files=['data/cluener_fewshot_dev.jsonl']
    for model in MODELS:
        tok=AutoTokenizer.from_pretrained(models[model]['path'],local_files_only=True)
        tokens=[{'id':r['id'],'prompt_ids':tok.encode(context+'Now answer the next request:\n'+prompt(r),add_special_tokens=False),
            'target_ids':tok.encode(canonical(r['target']),add_special_tokens=False)+[151643]} for r in dev]
        assert max(len(r['prompt_ids']) for r in tokens)<=1536
        filename=f'tokens/cluener_{model}_fewshot_dev.json';write(HERE/filename,tokens);files.append(filename)
        spec={'name':f'diagnostic_cluener_{model}_two_shot_base','model':model,'task':'cluener','arm':'base','seed':6100,'lr':0.,'eval_split':'fewshot_dev'}
        filename=f"specs/{spec['name']}.json";write(HERE/filename,spec);files.append(filename);specs.append(spec)
    write(HERE/'FEWSHOT_FROZEN.json',{'at':now(),'scope':'Base prompt sensitivity only; not a matched adapter contrast',
        'example_ids':[r['id'] for r in examples],'files':{f:sha(HERE/f) for f in files},'code_sha256':sha(__file__)})
    while not (HERE/'PILOT_COMPLETE.json').exists():time.sleep(10)
    assert read(HERE/'PILOT_COMPLETE.json')['status']=='passed'
    active=[]
    for gpu,spec in zip([1,2],specs):
        log=(HERE/'logs'/f"{spec['name']}.log").open('w');env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
        proc=subprocess.Popen([PYTHON,'-u',str(HERE/'run.py'),'--spec',str(HERE/'specs'/f"{spec['name']}.json")],env=env,stdout=log,stderr=subprocess.STDOUT)
        active.append((proc,log,spec));print('launched',spec['name'],proc.pid,flush=True)
    write(HERE/'FEWSHOT_RUNNING.json',{'at':now(),'processes':[{'pid':p.pid,'name':s['name']} for p,_,s in active]})
    while any(p.poll() is None for p,_,_ in active):time.sleep(10)
    outcomes=[]
    for p,f,s in active:f.close();outcomes.append({'name':s['name'],'code':p.returncode})
    write(HERE/'FEWSHOT_COMPLETE.json',{'at':now(),'status':'passed' if all(r['code']==0 for r in outcomes) else 'failed','outcomes':outcomes})
if __name__=='__main__':main()
