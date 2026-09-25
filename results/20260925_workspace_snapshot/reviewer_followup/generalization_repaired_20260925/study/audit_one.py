"""Independent post-run audit, invoked only after a worker exits successfully."""
import argparse,sys
from transformers import AutoTokenizer
from safetensors import safe_open
from common import *
from scoring import score,aggregate
from prepare import official_engine

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);args=ap.parse_args()
    root=HERE/'checkpoints'/args.name;spec=read(root/'spec.json');assert read(root/'COMPLETE.json')['status']=='passed'
    init=read(root/'INITIALIZATION.json');model=spec['model'];arm=spec['arm'];task=spec['task'];budget={arm:spec['expected_total_parameters']}
    assert init['trainable_parameters']==budget[arm]
    assert init['tied_weights']==read(HERE/'models.json')[model]['tie_word_embeddings']
    names=init['trainable_names'];assert all(n.startswith('boundary_') or '.lora_A.' in n or '.lora_B.' in n for n in names)
    assert any('.lora_' in n for n in names)==arm.startswith('hidden')
    placement=spec['placement']
    for side in ['input','output']:
        expected=placement in (['e','eu','shared'] if side=='input' else ['u','eu'])
        assert any(n.startswith('boundary_'+side) for n in names)==expected
    if placement=='shared':
        runtime=read(root/'SHARED_RUNTIME.json');assert min(runtime['calls'].values())>0
        assert init['boundary_sharing']['shared'] and init['tied_weights']
    if placement=='eu':assert init['independent_boundary_storage_verified']
    assert init['preflight']['zero_residual_max_abs_error']==0
    if arm!='base':
        tr=read(root/'TRAINING.json');n=64 if spec['smoke'] else len(spec.get('subset_indices',range(2048)));steps=(n+31)//32
        assert tr['steps']==steps and tr['examples']==n and tr['frozen_before']==tr['frozen_after']
        assert tr['reload_loss_error']==0 and tr['optimizer_whitelist_verified'] and tr['optimizer_fp32']
        assert tr['changed_groups']==tr['present_groups']
        assert sha(root/'adapter.safetensors')==tr['adapter_sha256']
        for file in ['initial_adapter.safetensors','adapter.safetensors']:
            with safe_open(str(root/file),framework='pt',device='cpu') as f:
                assert set(f.keys())==set(names)
                for k in f.keys():
                    if k.startswith('boundary_') and k.endswith('down.weight'):assert f.get_tensor(k).shape[0]==spec['rank']
                    if k.startswith('boundary_') and k.endswith('up.weight'):assert f.get_tensor(k).shape[1]==spec['rank']
                assert sum(f.get_tensor(k).numel() for k in f.keys())==budget[arm]
                assert all(str(f.get_tensor(k).dtype)=='torch.float32' for k in f.keys())
        import torch
        data=rows(HERE/f'data/{task}_train.jsonl');data=[data[i] for i in spec['subset_indices']] if spec.get('subset_indices') else data;order=torch.randperm(len(data),generator=torch.Generator().manual_seed(spec['seed'])).tolist()[:n]
        assert read(root/'TRAIN_ORDER.json')=={'indices':order,'ids':[data[i]['id'] for i in order]}
    for ap in (HERE/'audits').glob('*.json'):
        other=read(HERE/'checkpoints'/ap.stem/'spec.json')
        if other['model']==spec['model'] and other['seed']==spec['seed']:
            oi=read(HERE/'checkpoints'/ap.stem/'INITIALIZATION.json')
            assert oi['shared_hidden_initialization_sha256']==init['shared_hidden_initialization_sha256']
    resource=read(root/'RESOURCES.json')
    assert resource['training_wall_seconds']>0 and resource['peak_allocated_bytes']>0 and resource['peak_reserved_bytes']>=resource['peak_allocated_bytes']
    tok=AutoTokenizer.from_pretrained(read(HERE/'models.json')[model]['path'],local_files_only=True)
    checked=official=0;summaries={}
    for tag in (['smoke_dev'] if spec['smoke'] else spec.get('eval_splits',['dev','test'])):
        split='dev' if tag=='smoke_dev' else tag;p=HERE/'evaluations'/args.name/tag;s=read(p/'SUMMARY.json');rs=rows(p/'responses.jsonl');data=rows(HERE/f'data/{task}_{split}.jsonl');tokens=read(HERE/f'tokens/{task}_{model}_{split}.json')
        if spec['smoke']:data=data[:TASK_SETTINGS[task]['test_batch']];tokens=tokens[:len(data)]
        assert s['responses_sha256']==sha(p/'responses.jsonl')
        assert s['input_ids_sha256']==htext(canonical([r['prompt_ids'] for r in tokens]))
        assert [r['id'] for r in rs]==[r['id'] for r in data]
        if task=='wikisql':
            engine=official_engine(split);from lib.query import Query
            from scoring_sql import parse
        for r,g in zip(rs,data):
            assert tok.decode(r['token_ids'],skip_special_tokens=False)==r['text'] and (128001 if model.startswith('llama') else 151643) not in r['token_ids']
            assert r['length']==len(r['token_ids']) and r['capped']==(not r['native_eos'] and r['length']>=TASK_SETTINGS[task]['max_new_tokens'])
            for k,v in score(r['text'],g).items():assert r[k]==v,(args.name,g['id'],k)
            if task=='wikisql' and r['query_valid']:
                q,_=parse(r['text'])
                try:ref=engine.execute_query(g['table_id'],Query.from_dict(q),lower=True)==g['gold_execution']
                except Exception:ref=False
                assert ref==r['content_correct'] and r['lf_correct']==(Query.from_dict(q)==Query.from_dict(g['target']));official+=1
            checked+=1
        from metric_audit import audit_metrics
        audit_metrics(rs,data,s,read(HERE/f'data/{task}_labels.json'))
        got=aggregate(rs,task);assert got=={k:s[k] for k in got};summaries[tag]=sha(p/'SUMMARY.json')
    write(HERE/'audits'/f'{args.name}.json',{'at':now(),'status':'passed','name':args.name,'responses':checked,'official_sql_executions':official,'parameter_scope':budget[arm],'summary_sha256':summaries})
    print(canonical({'name':args.name,'status':'passed','responses':checked}),flush=True)
if __name__=='__main__':main()
