import os,time,subprocess
from common import *
from transformers import AutoTokenizer

def execute(jobs,stage):
    pending=list(jobs);active=[];finished=[];failed=[]
    while pending or active:
        used={r['gpu'] for r in active}
        for gpu in read(HERE/'GPU_POOL.json')['gpus']:
            if not pending:break
            if gpu in used:continue
            j=pending.pop(0);log=(HERE/'logs'/f"{stage.lower()}_{j['name']}.log").open('w')
            cmd=[PYTHON,'-u',str(HERE/'heldout_run.py'),'--checkpoint',str(HERE/'checkpoints'/j['name'])]
            if j.get('probe'):cmd.append('--probe')
            if j.get('fewshot'):cmd.append('--fewshot')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
            active.append({'job':j,'gpu':gpu,'process':proc,'log':log});print(json.dumps({'stage':stage,'launched':j['name'],'pid':proc.pid,'gpu':gpu}),flush=True)
        for r in list(active):
            code=r['process'].poll()
            if code is not None:
                r['log'].close();active.remove(r);result={'name':r['job']['name'],'code':code}
                (finished if code==0 else failed).append(result);print(json.dumps(result),flush=True)
        write(HERE/f'{stage}_STATE.json',{'at':now(),'pending':[r['name'] for r in pending],
            'active':[{'name':r['job']['name'],'gpu':r['gpu'],'pid':r['process'].pid} for r in active],'finished':finished,'failed':failed})
        if pending or active:time.sleep(10)
    write(HERE/f'{stage}_COMPLETE.json',{'at':now(),'status':'passed' if not failed else 'failed','finished':finished,'failed':failed})
    assert not failed,failed

def main():
    assert not (HERE/'HELDOUT_FROZEN.json').exists()
    # The training demonstrations were fixed before the diagnostic, not selected
    # against held-out labels or model outputs.
    demos=rows(HERE/'data/cluener_pilot_train.jsonl')[:2];context=''.join('Example '+str(i+1)+':\n'+prompt(r)+canonical(r['target'])+'\n\n' for i,r in enumerate(demos))
    data=rows(HERE/'data/cluener_test.jsonl');write_rows(HERE/'data/cluener_fewshot_test.jsonl',data)
    for m,cfg in read(HERE/'models.json').items():
        tok=AutoTokenizer.from_pretrained(cfg['path'],local_files_only=True)
        tokens=[{'id':r['id'],'prompt_ids':tok.encode(context+'Now answer the next request:\n'+prompt(r),add_special_tokens=False),
            'target_ids':tok.encode(canonical(r['target']),add_special_tokens=False)+[151643]} for r in data]
        assert all(len(r['prompt_ids'])<=1536 for r in tokens)
        write(HERE/f'tokens/cluener_{m}_fewshot_test.json',tokens)
    datafiles=[f'data/{t}_test.jsonl' for t in TASKS]+[f'tokens/{t}_{m}_test.json' for t in TASKS for m in MODELS]
    datafiles+=['data/cluener_fewshot_test.jsonl']+[f'tokens/cluener_{m}_fewshot_test.json' for m in MODELS]
    codefiles=['HELDOUT_PROTOCOL.md','heldout_run.py','heldout_pipeline.py','audit_heldout.py','analyze_heldout.py','plot_heldout.py','CODE_FROZEN.json','FEWSHOT_FROZEN.json']
    write(HERE/'HELDOUT_FROZEN.json',{'at':now(),'data_files':{f:sha(HERE/f) for f in datafiles},'code_files':{f:sha(HERE/f) for f in codefiles}})
    while not (HERE/'ADAPTER_PILOT_COMPLETE.json').exists():time.sleep(10)
    assert read(HERE/'ADAPTER_PILOT_COMPLETE.json')['status']=='passed';assert read(HERE/'FEWSHOT_COMPLETE.json')['status']=='passed'
    probes=[{'name':f'pilot_{task}_{model}_{arm}','probe':True} for task in TASKS for model in MODELS for arm in ['base','hidden']]
    execute(probes,'HELDOUT_PROBE')
    records=[read(HERE/'heldout_probes'/f"{j['name']}.json") for j in probes]
    flips=sum(len(r['batch32_vs_saved_batch8_token_differences']) for r in records)
    write(HERE/'HELDOUT_BATCH.json',{'at':now(),'batch_size':8 if flips else 32,'raw_sequence_differences':flips,'dev_examples_checked':256,
        'reason':'globally revert to original batch8 on any development token difference; no test output inspected'})
    jobs=[{'name':f'pilot_{task}_{model}_base'} for task in TASKS for model in MODELS]
    for seed in range(6100,6105):
        for task in TASKS:
            for model in MODELS:
                for arm in ['hidden','input','output','hidden_both','hidden_budget']:
                    name=f'pilot_{task}_{model}_hidden' if seed==6100 and arm=='hidden' else f'adapter_{task}_{model}_{arm}_s{seed}'
                    jobs.append({'name':name})
    jobs += [{'name':f'diagnostic_cluener_{m}_two_shot_base','fewshot':True} for m in MODELS]
    assert len(jobs)==106;write(HERE/'HELDOUT_JOBS.json',jobs);execute(jobs,'HELDOUT')
if __name__=='__main__':main()
