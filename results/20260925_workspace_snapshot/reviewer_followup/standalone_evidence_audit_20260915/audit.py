"""Read-only audit of existing corrected-SFT standalone evidence."""
import hashlib,json,math
from datetime import datetime
from pathlib import Path
from statistics import mean,stdev
import torch
from safetensors.torch import load_file

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'placement_capacity_corrected_20260911'
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def main():
    manifest=read(OLD/'manifest.json');summary=read(OLD/'summary.json');files={};rows=[];nreports=0
    for p,h in manifest['sha256'].items():assert sha(OLD/p)==h,p
    test_ids=[json.loads(s)['record_id'] for s in (OLD/'data/test.jsonl').read_text().splitlines()]
    for model,cfg in manifest['models'].items():
        config_path=Path(cfg['path'])/'config.json';assert sha(config_path)==cfg['config_sha256']
        config=read(config_path);d=config['hidden_size'];v=config['vocab_size']
        # Original study recorded size and mtime, not a full weight hash.
        for fn,stat in cfg['weight_files'].items():
            current=(Path(cfg['path'])/fn).stat()
            assert (current.st_size,current.st_mtime_ns)==(stat['bytes'],stat['mtime_ns'])
        reference_ids={}
        for placement in ['none','input','output']:
            ces=[];counts=[];digests=[]
            for seed in ([None] if placement=='none' else [42,43,44]):
                name=f'{model}_{placement}_hr0_'+('base' if seed is None else f'sd{seed}')
                for split in ['dev','test']:
                    path=OLD/'reports'/f'{name}.{split}.json';report=read(path);files[str(path)]=sha(path)
                    ex=report['per_example'];assert len(ex)==report['num_examples']==1000
                    ids=[(e['record_id'],e['token_count']) for e in ex]
                    if split in reference_ids:assert ids==reference_ids[split]
                    else:reference_ids[split]=ids
                    if split=='test':assert [e['record_id'] for e in ex]==test_ids
                    total=sum(e['nll_sum'] for e in ex);tokens=sum(e['token_count'] for e in ex)
                    assert tokens==report['supervised_tokens'] and abs(total-report['total_nll'])<1e-8
                    assert abs(total/tokens-report['avg_ce'])<1e-12
                    assert all(abs(e['nll_sum']/e['token_count']-e['mean_ce'])<1e-12 for e in ex)
                    assert report['run_dir']==(None if seed is None else str(OLD/'checkpoints'/name))
                    if split=='test':ces.append(report['avg_ce'])
                    nreports+=1
                if seed is None:counts.append(0);continue
                cp=OLD/'checkpoints'/name;args=read(cp/'run_args.json');init=read(cp/'initialization_audit.json')
                assert args['hidden_lora_rank']==0 and init['trainable']['hidden']==0 and init['hidden_init_sha256'] is None
                assert args['initial_hidden_lora_adapter'] is None and args['initial_affine_adapter'] is None
                assert not (cp/'adapter_config.json').exists()
                tensors=load_file(str(cp/'affine_vocab_adapter.safetensors'))
                assert all('affine' in k and 'lora_' not in k for k in tensors)
                assert all(t.dtype==torch.float32 and torch.isfinite(t).all() for t in tensors.values())
                count=sum(t.numel() for t in tensors.values());expected=2*d*16+(d if placement=='input' else 0)
                assert count==expected==init['total_trainable']==read(cp/'trainable_summary.json')['trainable']
                assert any('.up.weight' in k and t.norm()>0 for k,t in tensors.items())
                counts.append(count);digests.append(sha(cp/'affine_vocab_adapter.safetensors'))
                for fn in ['affine_vocab_adapter.safetensors','run_args.json','initialization_audit.json','trainable_summary.json','affine_vocab_config.json']:files[str(cp/fn)]=sha(cp/fn)
            reported=next(r for r in summary['models'][model]['rows'] if r['rank']==0 and r['placement']==placement)
            assert ces==reported['ce']['values'] and counts==reported['parameters']
            row={'model':model,'placement':placement,'hidden_rank':0,'seeds':([None] if placement=='none' else [42,43,44]),
                 'trainable_parameters':counts[0],'test_ce_mean':mean(ces),'test_ce_per_seed':ces,'checkpoint_current_sha256':digests}
            if len(ces)>1:
                half=4.302652729911275*stdev(ces)/math.sqrt(3);row['nominal_seed_ci95']=[mean(ces)-half,mean(ces)+half]
            if placement=='input':
                row['same_rank16_vocab_input_parameter_formula']=16*(v+d)
                row['same_rank_parameter_ratio_vocab_over_affine']=16*(v+d)/counts[0]
                row['parameter_comparison_note']='Algebraic same-rank, single-matrix count only; no matched-performance comparison.'
            rows.append(row)
    out={'at':datetime.now().astimezone().isoformat(timespec='seconds'),'status':'passed','scope':'Existing artifact consistency and parameter count; no new model inference or training.',
         'original_frozen_source_data_hashes_verified':len(manifest['sha256']),'reports_recomputed':nreports,'standalone_checkpoints_inspected':12,
         'model_weight_provenance':'Original config hashes and weight size/mtime verified; original study did not seal full weight hashes.',
         'rows':rows,'source_artifact_sha256':files}
    write(HERE/'AUDIT.json',out)
    print(json.dumps({k:v for k,v in out.items() if k!='source_artifact_sha256'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
