"""Re-encode exact datasets, audit accepted historical runs, enumerate fixed jobs."""
import sys,shutil,collections,math
import torch
from transformers import AutoTokenizer
from common import *
from scoring import score,aggregate

def official_engine(split):
    sys.path[:0]=[str(HERE/'raw/reference_deps'),str(HERE/'raw/official_wikisql')]
    from lib.dbengine import DBEngine
    return DBEngine(str(HERE/f'raw/wikisql/data/{split}.db'))

def main():
    configs=read(HERE/'models.json');assert set(configs)==set(MODELS)
    toks={m:AutoTokenizer.from_pretrained(c['path'],local_files_only=True) for m,c in configs.items()}
    audit={'at':now(),'counts':{},'lengths':{},'reused_responses_rescored':0,'source_files_checked':0}
    all_data={};all_tokens={};budgets={}
    for model,cfg in configs.items():
        c=read(Path(cfg['path'])/'config.json');d=c['hidden_size'];layers=c['num_hidden_layers'];hd=c.get('head_dim',d//c['num_attention_heads'])
        q=c['num_attention_heads']*hd;k=c['num_key_value_heads']*hd;ff=c['intermediate_size']
        hidden=8*layers*((d+q)+2*(d+k)+(q+d)+3*(d+ff))
        extra,_,nk,nq=min((nq*(d+q)+nk*(d+k)-65*d,-nq,nk,nq) for nq in range(layers+1) for nk in range(layers+1) if nq*(d+q)+nk*(d+k)>=65*d)
        assert extra<=65*d*.003
        budgets[model]={'base':0,'hidden':hidden,'hidden_budget':hidden+65*d+extra,'hidden_input':hidden+33*d,'hidden_output':hidden+32*d,'hidden_both':hidden+65*d,'input':33*d,'output':32*d,'both':65*d,'budget_excess':extra,'q_expanded':nq,'k_expanded':nk,'tied':cfg['tie_word_embeddings']}
    write(HERE/'BUDGET_PLAN.json',budgets)
    for task in TASKS:
        sets=[]
        for split in ['train','dev','test']:
            data=rows(HERE/f'data/{task}_{split}.jsonl');all_data[task,split]=data
            expected=({'train':2048,'dev':200,'test':1343} if task=='cluener' else {'train':2048,'dev':256,'test':1024})[split]
            assert len(data)==expected;sets.append({prompt(r).casefold() for r in data});audit['counts'][task+'_'+split]=len(data)
            for model,tok in toks.items():
                assert tok.eos_token_id==151643
                tokens=[{'id':r['id'],'prompt_ids':tok.encode(prompt(r),add_special_tokens=False),'target_ids':tok.encode(canonical(r['target']),add_special_tokens=False)+[151643]} for r in data]
                assert all(len(r['prompt_ids'])+len(r['target_ids'])<=4096 and len(r['target_ids'])<=TASK_SETTINGS[task]['max_new_tokens'] for r in tokens)
                assert all(r['target_ids'].count(151643)==1 and not ({151643,151644,151645}&set(r['prompt_ids'])) and not ({151644,151645}&set(r['target_ids'])) for r in tokens)
                name=f'{task}_{model}_{split}';write(HERE/f'tokens/{name}.json',tokens);all_tokens[task,model,split]=tokens
                audit['lengths'][name]={'n':len(tokens),'max_prompt':max(len(r['prompt_ids']) for r in tokens),'max_target':max(len(r['target_ids']) for r in tokens)}
        assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    roots=HERE.parent.parent
    sources=[('model_scaling_20260917','adb6c2f3b26e26208b9bb3b71ecd445cfd525e01bcc7a02ad0f2168594e06ae9','evaluations/*/test/SUMMARY.json'),('downstream_transfer_20260917','ddfcf8b6c43e39303d9900efeac54d68899bf23b51bfb59426a59637ffd1fdc6','evaluations/*/test/SUMMARY.json'),('posttraining_tasks_grounded_20260916','c717690a663b6d58c70224b4b4f798e34343f1e0de49b8e8bf1df7e91d37ab5f','heldout/*/*/SUMMARY.json')]
    anchors=[];seen=set()
    for dirname,manifest_hash,pattern in sources:
        source=roots/dirname;manifest=read(source/'ARTIFACT_MANIFEST.json')
        assert sha(source/'ARTIFACT_MANIFEST.json')==manifest_hash,(dirname,'manifest identity')
        checked={}
        def check(p):
            rel=str(p.relative_to(source));expected=manifest['files'][rel]['sha256'];actual=sha(p);assert actual==expected,p
            checked[rel]=actual;return actual
        for sp in sorted(source.glob(pattern)):
            check(sp);s=read(sp);spec=s['spec'];task=spec['task'];model=spec['model'];arm=spec['arm'];seed=spec['seed']
            if task not in TASKS or model not in MODELS or arm not in ['base']+ARMS:continue
            if (s.get('eval_split') or spec.get('eval_split'))!='test':continue
            key=(task,model,arm,None if arm=='base' else seed)
            if key in seen:continue
            if arm!='base' and seed not in range(TASK_SETTINGS[task]['seed_start'],TASK_SETTINGS[task]['seed_start']+5):continue
            data=all_data[task,'test'];tokens=all_tokens[task,model,'test'];rp=sp.parent/'responses.jsonl';check(rp);rs=rows(rp)
            if len(rs)!=len(data):continue
            assert [r['id'] for r in rs]==[r['id'] for r in data]
            assert sha(rp)==s['responses_sha256'] and s['input_ids_sha256']==htext(canonical([r['prompt_ids'] for r in tokens]))
            ck=source/'checkpoints'/spec['name'];check(ck/'spec.json');assert read(ck/'spec.json')=={k:v for k,v in spec.items() if k not in ['eval_split']}
            init=read(ck/'INITIALIZATION.json');check(ck/'INITIALIZATION.json')
            assert init['trainable_parameters']==budgets[model][arm]
            assert init['preflight']['zero_residual_max_abs_error']==0
            files=[sp,rp,ck/'spec.json',ck/'INITIALIZATION.json'];adapter_checks={}
            if arm!='base':
                tr=read(ck/'TRAINING.json');check(ck/'TRAINING.json');assert tr['steps']==64 and tr['examples']==2048
                assert tr['frozen_before']==tr['frozen_after'] and tr['reload_loss_error']==0 and tr['optimizer_fp32'] and tr['optimizer_whitelist_verified']
                assert check(ck/'adapter.safetensors')==tr['adapter_sha256'];adapter_checks['adapter.safetensors']=tr['adapter_sha256']
                adapter_checks['initial_adapter.safetensors']=check(ck/'initial_adapter.safetensors')
                order=read(ck/'TRAIN_ORDER.json');check(ck/'TRAIN_ORDER.json');train=all_data[task,'train']
                planned=torch.randperm(len(train),generator=torch.Generator().manual_seed(seed)).tolist()
                assert order['indices']==planned and order['ids']==[train[i]['id'] for i in planned]
                assert spec['lr']==2e-4
                files += [ck/'TRAINING.json',ck/'TRAIN_ORDER.json']
            for r,g in zip(rs,data):
                assert toks[model].decode(r['token_ids'],skip_special_tokens=False)==r['text']
                for k,v in score(r['text'],g).items():assert r[k]==v,(dirname,key,r['id'],k)
            got=aggregate(rs,task);assert got=={k:s[k] for k in got}
            dest=HERE/'anchors'/f"{task}_{model}_{arm}"/(str(seed) if arm!='base' else 'base');dest.mkdir(parents=True,exist_ok=False)
            for p in files:shutil.copyfile(p,dest/p.name)
            anchors.append({'task':task,'model':model,'arm':arm,'seed':None if arm=='base' else seed,'directory':str(dest.relative_to(HERE)),'source':str(source),'source_summary':str(sp.relative_to(source)),'source_checkpoint':str(ck),'source_manifest_sha256':manifest_hash,'adapter_hashes':adapter_checks,'summary':s})
            seen.add(key);audit['reused_responses_rescored']+=len(rs)
        audit['source_files_checked']+=len(checked);write(HERE/'provenance'/f'{dirname}_reuse.json',{'status':'passed','source':str(source),'manifest_sha256':manifest_hash,'files_checked':checked})
    write(HERE/'ANCHOR_INDEX.json',anchors)
    jobs=[];inventory=[];smokes=[]
    priority={'qwen25_7b_base':10,'qwen3_8b_base':10,'qwen25_05b_base':15,'qwen3_17b_base':25,'qwen3_4b_base':25,'qwen25_15b_base':30,'qwen25_3b_base':30,'qwen3_06b_base':30}
    for model in MODELS:
        for task in TASKS:
            seed0=TASK_SETTINGS[task]['seed_start'];n=5 if model in ['qwen25_7b_base','qwen3_8b_base'] else 3
            for arm in ['base']+ARMS:
                for seed in ([seed0] if arm=='base' else range(seed0,seed0+n)):
                    name=f'{task}_{model}_'+('base' if arm=='base' else f'{arm}_s{seed}')
                    spec={'name':name,'task':task,'model':model,'arm':arm,'seed':seed,'lr':2e-4,'microbatch':TASK_SETTINGS[task]['microbatch'],'smoke':False}
                    old=next((a for a in anchors if (a['task'],a['model'],a['arm'],a['seed'])==(task,model,arm,None if arm=='base' else seed)),None)
                    record={'spec':spec,'reused':old is not None,'anchor_directory':old['directory'] if old else None,'priority':22 if seed>=seed0+3 else priority[model]}
                    inventory.append(record)
                    if old is None:jobs.append(record)
            spec={'name':f'smoke_{task}_{model}','task':task,'model':model,'arm':'hidden_both' if task=='cluener' else 'hidden_budget','seed':seed0-1,'lr':2e-4,'microbatch':TASK_SETTINGS[task]['microbatch'],'smoke':True}
            smokes.append({'spec':spec,'priority':0,'reused':False})
    jobs.sort(key=lambda r:(r['priority'],r['spec']['seed'],r['spec']['task'],r['spec']['model'],ARMS.index(r['spec']['arm']) if r['spec']['arm'] in ARMS else -1))
    write(HERE/'SMOKE_JOBS.json',smokes);write(HERE/'FORMAL_JOBS.json',jobs);write(HERE/'INVENTORY.json',inventory)
    audit.update(status='passed',reused_runs=len(anchors),new_formal_jobs=len(jobs),smoke_jobs=len(smokes))
    write(HERE/'DATA_AND_REUSE_AUDIT.json',audit);print(canonical(audit),flush=True)
if __name__=='__main__':main()
