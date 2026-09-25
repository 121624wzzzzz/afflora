"""Same sealed examples/prompts, native Llama BOS/EOS, fixed three-seed main arms."""
import sys
from transformers import AutoTokenizer
from common import *

def official_engine(split):
    sys.path[:0]=[str(HERE/'raw/reference_deps'),str(HERE/'raw/official_wikisql')]
    from lib.dbengine import DBEngine
    return DBEngine(str(HERE/f'raw/wikisql/data/{split}.db'))

def main():
    configs=read(HERE/'models.json');assert set(configs)==set(MODELS)
    audit={'at':now(),'counts':{},'lengths':{},'reused_fits':0,'native_bos':TOKEN_BOS_ID,'native_eos':TOKEN_EOS_ID};budgets={}
    for model,cfg in configs.items():
        c=read(Path(cfg['path'])/'config.json');d=c['hidden_size'];L=c['num_hidden_layers'];hd=c.get('head_dim',d//c['num_attention_heads'])
        q=c['num_attention_heads']*hd;k=c['num_key_value_heads']*hd;ff=c['intermediate_size']
        hidden=8*L*(2*(d+q)+2*(d+k)+3*(d+ff))
        sys.path.insert(0,str(HERE/'source'))
        from budget import make_allocation
        plan=make_allocation(c);assert plan['hidden']==hidden
        budgets[model]={'base':0,'hidden':hidden,'hidden_budget':hidden+65*d,'hidden_both':hidden+65*d,'budget_excess':0,'q_expanded':plan['q_expanded'],'k_expanded':plan['k_expanded'],'down_expanded':plan['down_expanded'],'tied':cfg['tie_word_embeddings']}
    write(HERE/'BUDGET_PLAN.json',budgets)
    for model,cfg in configs.items():
        tok=AutoTokenizer.from_pretrained(cfg['path'],local_files_only=True)
        assert tok.eos_token_id==TOKEN_EOS_ID and tok.bos_token_id==TOKEN_BOS_ID
        for task in TASKS:
            sets=[]
            for split in ['train','dev','test']:
                data=rows(HERE/f'data/{task}_{split}.jsonl')
                expected=({'train':2048,'dev':200,'test':1343} if task=='cluener' else {'train':2048,'dev':256,'test':1024})[split]
                assert len(data)==expected;sets.append({prompt(r).casefold() for r in data});audit['counts'][task+'_'+split]=len(data)
                tokens=[]
                for r in data:
                    p=[TOKEN_BOS_ID]+tok.encode(prompt(r),add_special_tokens=False)
                    assert tok.encode(prompt(r),add_special_tokens=True)==p
                    target=tok.encode(canonical(r['target']),add_special_tokens=False)+[TOKEN_EOS_ID]
                    assert p.count(TOKEN_BOS_ID)==1 and TOKEN_EOS_ID not in p
                    assert target.count(TOKEN_EOS_ID)==1 and TOKEN_BOS_ID not in target
                    assert not (set(tok.all_special_ids)-{TOKEN_BOS_ID,TOKEN_EOS_ID}) & set(p+target)
                    tokens.append({'id':r['id'],'prompt_ids':p,'target_ids':target})
                name=f'{task}_{model}_{split}'
                lengths={'n':len(tokens),'max_prompt':max(len(r['prompt_ids']) for r in tokens),'max_target':max(len(r['target_ids']) for r in tokens),'max_sequence':max(len(r['prompt_ids'])+len(r['target_ids']) for r in tokens)}
                audit['lengths'][name]=lengths;print(name,lengths,flush=True)
                assert lengths['max_sequence']<=4096 and lengths['max_target']<=TASK_SETTINGS[task]['max_new_tokens'],lengths
                write(HERE/f'tokens/{name}.json',tokens)
            assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    jobs=[];smokes=[]
    for model in MODELS:
        for task in TASKS:
            seed0=TASK_SETTINGS[task]['seed_start']
            for arm in ['base']+ARMS:
                for seed in ([seed0] if arm=='base' else range(seed0,seed0+3)):
                    name=f'{task}_{model}_'+('base' if arm=='base' else f'{arm}_s{seed}')
                    spec={'name':name,'task':task,'model':model,'arm':arm,'seed':seed,'lr':2e-4,'microbatch':TASK_SETTINGS[task]['microbatch'],'smoke':False}
                    jobs.append({'spec':spec,'reused':False,'priority':1 if arm=='base' else 2+seed-seed0})
            for arm in ['hidden_both','hidden_budget']:
                name=f'smoke_{task}_{model}_{arm}'
                smokes.append({'spec':{'name':name,'task':task,'model':model,'arm':arm,'seed':seed0-1,'lr':2e-4,'microbatch':TASK_SETTINGS[task]['microbatch'],'smoke':True},'priority':0,'reused':False})
    jobs.sort(key=lambda j:(j['priority'],j['spec']['task'],j['spec']['model'],j['spec']['arm']))
    write(HERE/'ANCHOR_INDEX.json',[]);write(HERE/'SMOKE_JOBS.json',smokes);write(HERE/'FORMAL_JOBS.json',jobs);write(HERE/'INVENTORY.json',jobs)
    audit.update(status='passed',new_formal_jobs=len(jobs),smoke_jobs=len(smokes));write(HERE/'DATA_AND_REUSE_AUDIT.json',audit)
if __name__=='__main__':main()
