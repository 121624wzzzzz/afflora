"""Reverify frozen inputs, all token encodings, pairs and output audits."""
from common import *
from transformers import AutoTokenizer
from safetensors.torch import load_file

def main():
    assert read(HERE/'SCHEDULER_COMPLETE.json')['status']=='passed'
    checked=0
    for fn in ['CODE_FROZEN.json','DATA_FROZEN.json']:
        for rel,h in read(HERE/fn)['files'].items():assert sha(HERE/rel)==h,rel;checked+=1
    token_count=0
    for model,cfg in read(HERE/'models.json').items():
        for file,h in cfg['files'].items():assert sha(Path(file))==h,file
        tok=AutoTokenizer.from_pretrained(cfg['path'],local_files_only=True)
        for task in TASKS:
            for split in ['train','dev','test']:
                data=rows(HERE/f'data/{task}_{split}.jsonl');tokens=read(HERE/f'tokens/{task}_{model}_{split}.json');assert len(data)==len(tokens)
                for r,tr in zip(data,tokens):
                    assert tr=={'id':r['id'],'prompt_ids':tok.encode(prompt(r),add_special_tokens=True),'target_ids':tok.encode(canonical(r['target']),add_special_tokens=False)+[TOKEN_EOS_ID]};token_count+=1
    runs=responses=sql=0
    for job in read(HERE/'SMOKE_JOBS.json')+read(HERE/'FORMAL_JOBS.json'):
        s=job['spec'];name=s['name'];a=read(HERE/'audits'/f'{name}.json');assert a['status']=='passed'
        assert read(HERE/'checkpoints'/name/'spec.json')==s
        for split,h in a['summary_sha256'].items():
            p=HERE/'evaluations'/name/split/'SUMMARY.json';assert sha(p)==h;summary=read(p);assert sha(p.parent/'responses.jsonl')==summary['responses_sha256']
        if s['arm']!='base':
            root=HERE/'checkpoints'/name;tr=read(root/'TRAINING.json');assert sha(root/'adapter.safetensors')==tr['adapter_sha256']
        runs+=1;responses+=a['responses'];sql+=a['official_sql_executions']
    pairs=[]
    for model in MODELS:
        for task in TASKS:
            for seed in range(TASK_SETTINGS[task]['seed_start'],TASK_SETTINGS[task]['seed_start']+3):
                roots=[HERE/'checkpoints'/f'{task}_{model}_{arm}_s{seed}' for arm in ARMS]
                init=[read(p/'INITIALIZATION.json') for p in roots]
                assert len({r['shared_hidden_initialization_sha256'] for r in init})==1
                actual=[]
                for root,record in zip(roots,init):
                    h=hashlib.sha256()
                    for name,value in sorted(load_file(str(root/'initial_adapter.safetensors')).items()):
                        if '.lora_' not in name:continue
                        value=value[:8,:] if '.lora_A.' in name else value[:,:8]
                        h.update(name.encode());h.update(value.float().contiguous().numpy().tobytes())
                    assert h.hexdigest()==record['shared_hidden_initialization_sha256'];actual.append(h.hexdigest())
                assert len(set(actual))==1
                assert len({canonical(read(p/'TRAIN_ORDER.json')) for p in roots})==1
                assert len({r['frozen_before'] for r in init})==1
                counts={arm:r['trainable_parameters'] for arm,r in zip(ARMS,init)};assert counts['hidden_budget']>=counts['hidden_both']
                pairs.append({'task':task,'model':model,'seed':seed,'counts':counts,'status':'passed'})
    write(HERE/'FINAL_AUDIT.json',{'at':now(),'status':'passed','frozen_files_checked':checked,'tokens_reencoded':token_count,'runs':runs,'responses_checked':responses,'official_sql_checks':sql,'paired_groups':pairs})
    print('Final audit passed',runs,'runs',responses,'responses',flush=True)
if __name__=='__main__':main()
