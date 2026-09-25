"""Independent arithmetic, artifact binding and frozen-parameter audit."""
import numpy as np
import torch
from scipy.special import logsumexp
from safetensors.torch import load_file
from transformers import AutoTokenizer
from settings import *
from run import specs,verify_inputs
from modeling import digest

def main():
    verify_inputs(True);assert not list((HERE/'checkpoints').glob('*/FAILED.json'))
    selection=read(HERE/'SELECTION.json')
    for m,arms in selection['selected'].items():
        for arm,selected in arms.items():
            for candidate in selected['all_candidates']:
                path=Path(candidate['source'])/'validation_metrics.json'
                assert sha(path)==candidate['metrics_sha256']
                metric=read(path)['primary']
                assert metric['accuracy']==candidate['accuracy'] and metric['candidate_nll']==candidate['candidate_nll']
            best=sorted(selected['all_candidates'],key=lambda x:(-x['accuracy'],x['candidate_nll'],x['lr']))[0]
            assert best['lr']==selected['lr'] and best['source']==selected['source']
    copies=read(HERE/'REUSE_AUDIT.json');old=Path(copies['source_root']);seal=read(old/'ARTIFACT_MANIFEST.json')['files']
    for entry in seal:assert sha(old/entry['path'])==entry['sha256'],entry['path']
    tokenizers={m:AutoTokenizer.from_pretrained(c['path'],local_files_only=True) for m,c in read(HERE/'models.json').items()}
    count=gens=frozen=near_flips=gen_flips=0;frozen_hashes={};base_replays={};stop_ids={}
    for model,cfg in read(HERE/'models.json').items():
        native=read(Path(cfg['path'])/'generation_config.json')['eos_token_id']
        stop_ids[model]=sorted(set((native if isinstance(native,list) else [native])+[read(HERE/'DATA_AUDIT.json')['tokenization'][model]['eos_id']]))
    for phase,njobs in [('smoke',6),('tuning',18),('confirmation',30),('reference',2)]:
        assert read(HERE/f'{phase.upper()}_COMPLETE.json')['jobs']==njobs
        for s in specs(phase):
            cp=Path(s['checkpoint']);assert read(cp/'COMPLETE.json')['spec']==s
            assert read(cp/'spec.json')==s
            if phase!='reference':
                train=read(cp/'TRAINING.json');init=read(cp/'INITIALIZATION.json');state=load_file(str(cp/'adapter.safetensors'))
                assert train['adapter_sha256']==sha(cp/'adapter.safetensors') and train['tensor_sha256']==digest(state)
                assert set(state)==set(init['trainable_names'])
                assert all(('.lora_A.' in k or '.lora_B.' in k) if s['arm']=='hidden_r8' else k.startswith('standalone_affine.') for k in state)
                assert sum(t.numel() for t in state.values())==init['trainable_parameters']
                assert all(t.dtype==torch.float32 and torch.isfinite(t).all() for t in state.values())
                assert init['hidden_lora_installed']==(s['arm']=='hidden_r8') and train['optimizer_whitelist_verified']
                assert train['frozen_parameters_before_sha256']==train['frozen_parameters_after_sha256']==init['frozen_parameters_before_sha256']
                frozen_hashes.setdefault(s['model'],set()).add(train['frozen_parameters_before_sha256'])
                assert train['base_gradient_checks']==(8 if phase=='smoke' else 1456)
                assert train['steps']==(2 if phase=='smoke' else 364)
                if s['arm']=='hidden_r8':assert train['hook_counts']=={'input':0,'output':0}
                else:assert train['hook_counts'][s['arm']]>0 and train['hook_counts'][{'input':'output','output':'input'}[s['arm']]]==0
                frozen+=1
            if phase=='smoke':
                assert read(cp/'RELOAD_AUDIT.json')['status']=='passed';continue
            split='validation' if phase=='tuning' else 'test';metrics=read(cp/f'{split}_metrics.json');pp=cp/f'{split}_predictions.jsonl'
            assert sha(pp)==metrics['prediction_sha256'];rr=rows(pp)
            data=read(HERE/'tokens'/f"{s['model']}_{'validation' if phase=='tuning' else 'test_rotations'}.json")
            assert [(r['id'],r['rotation'],r['gold'],r['ambiguous_gold']) for r in rr]==[(r['id'],r['rotation'],r['gold'],r['ambiguous_gold']) for r in data]
            ll=np.array([r['label_logits'] for r in rr]);gold=np.array([r['gold'] for r in rr]);nll=logsumexp(ll,axis=1)-ll[np.arange(len(rr)),gold]
            assert np.max(np.abs(nll-np.array([r['nll'] for r in rr])))<1e-11
            pred=ll.argmax(1);assert np.array_equal(pred,np.array([r['prediction'] for r in rr]))
            correct=pred==gold;assert np.array_equal(correct,np.array([r['correct'] for r in rr]))
            mask=np.array([r['rotation']==0 and not r['ambiguous_gold'] for r in rr]);assert mask.sum()==metrics['primary']['n']
            assert abs(100*correct[mask].mean()-metrics['primary']['accuracy'])<1e-11
            assert abs(nll[mask].mean()-metrics['primary']['candidate_nll'])<1e-11
            if phase!='reference':assert metrics['checkpoint_sha256']==sha(cp/'adapter.safetensors')
            near_flips+=sum(r['prediction_flip'] for r in metrics['near_tie_rechecks']);count+=len(rr)
            if phase=='tuning':continue
            gp=cp/'test_generation.jsonl';assert sha(gp)==metrics['generation_sha256'];gg=rows(gp)
            assert len(gg)==1000 and [(g['id'],g['gold']) for g in gg]==[(r['id'],r['gold']) for r in rr if r['rotation']==0]
            tok=tokenizers[s['model']];total_valid=total_correct=total_stopped=total_capped=0
            for g in gg:
                assert len(g['generated_ids'])<=MAX_NEW_TOKENS
                stop=next((i for i,tokid in enumerate(g['generated_ids']) if tokid in stop_ids[s['model']]),None)
                content=g['generated_ids'] if stop is None else g['generated_ids'][:stop]
                assert content==g['content_ids'] and g['terminated']==(stop is not None)
                assert g['hit_length_cap']==(stop is None and len(g['generated_ids'])>=MAX_NEW_TOKENS)
                assert tok.decode(g['content_ids'],skip_special_tokens=False)==g['text']
                txt=g['text'].strip();valid=txt in ['A','B','C','D'];correct=valid and txt=='ABCD'[g['gold']]
                assert valid==g['valid_answer'] and correct==g['strict_correct']
                if not g['ambiguous_gold']:
                    total_valid+=valid;total_correct+=correct;total_stopped+=g['terminated'];total_capped+=g['hit_length_cap']
            assert abs(metrics['generation']['strict_accuracy']-100*total_correct/998)<1e-11
            assert abs(metrics['generation']['valid_answer_rate']-100*total_valid/998)<1e-11
            assert abs(metrics['generation']['terminated_rate']-100*total_stopped/998)<1e-11
            assert abs(metrics['generation']['length_cap_rate']-100*total_capped/998)<1e-11
            gen_flips+=len(metrics['generation_first_token_disagreements_with_candidate_pass']);gens+=len(gg)
    assert all(len(h)==1 for h in frozen_hashes.values())
    assert count==146000 and gens==32000 and frozen==54
    write(HERE/'FINAL_AUDIT.json',{'at':now(),'status':'passed','training_jobs':frozen,'all_frozen_weights_bitwise_unchanged':True,
        'one_frozen_base_digest_per_model':{m:next(iter(h)) for m,h in frozen_hashes.items()},'prediction_rows_recomputed':count,'generation_rows_redecoded':gens,
        'near_tie_prediction_flips':near_flips,'generation_first_token_shape_disagreements':gen_flips,'generation_stop_ids':stop_ids,
        'original_sealed_files_rechecked':len(seal),'single_side_trainable_whitelist_verified':True,'separate_hidden_lora_control_verified':True})
    print(read(HERE/'FINAL_AUDIT.json'))

if __name__=='__main__':main()
