import numpy as np
from transformers import AutoTokenizer
from settings import *
from pipeline import verify_frozen,verify_sealed_sources
from scoring import decode_output


def main():
    verify_frozen();verify_sealed_sources()
    models=read(HERE/'models.json');tokenizers={m:AutoTokenizer.from_pretrained(c['path'],local_files_only=True) for m,c in models.items()}
    prediction_count=0;generation_count=0;reference_replays=0;invalid={};max_logit_error=0.
    paths=list((HERE/'evaluations').glob('*/test_metrics.json'))+list((HERE/'evaluations').glob('*/validation_metrics.json'))
    assert len(paths)==70,len(paths)
    for path in paths:
        report=read(path);split=report['split'];model=report['spec']['model'];directory=path.parent
        cp=Path(report['spec']['checkpoint']);raw=rows(HERE/'data'/f'{split}.jsonl')
        assert sha(HERE/'tokens'/f'{model}_{split}.json')==report['input_tokens_sha256']
        predpath=directory/f'{split}_predictions.jsonl';assert sha(predpath)==report['prediction_sha256'];pred=rows(predpath)
        assert len(pred)==len(raw)==1000
        for p,r in zip(pred,raw):
            assert all(p[k]==r[k] for k in ['id','gold','ambiguous_gold'])
            assert len(p['label_logits'])==4 and all(np.isfinite(p['label_logits']))
            assert p['prediction']==int(np.argmax(p['label_logits']))
            assert p['correct']==(p['prediction']==p['gold'])
            assert p['unrestricted_correct']==(p['unrestricted_token']==32+p['gold'])
            assert p['valid_label']==(p['unrestricted_token'] in [32,33,34,35])
            assert p['answer_ce']>=0 and p['eos_ce']>=0
        clean=[r for r in pred if not r['ambiguous_gold']]
        assert abs(report['primary']['candidate_accuracy']-100*np.mean([r['correct'] for r in clean]))<1e-10
        prediction_count+=len(pred)
        if 'original_candidate_replay' in report:
            replay=report['original_candidate_replay'];assert replay['prediction_flips']==0
            max_logit_error=max(max_logit_error,replay['max_logit_error']);reference_replays+=1
        assert report['adapter_tensors_unchanged'] and report['original_weights_unchanged_during_eval']
        if split=='test':
            genpath=directory/'test_generation.jsonl';assert sha(genpath)==report['generation_sha256'];gen=rows(genpath)
            assert len(gen)==1000 and not report['first_token_disagreements']
            args=report['generation_settings'];assert args['max_new_tokens']==128 and args['do_sample'] is False
            invalid[report['spec']['name']]=[]
            for g,r in zip(gen,raw):
                assert all(g[k]==r[k] for k in ['id','gold','ambiguous_gold'])
                fresh=decode_output(g['generated_ids'],tokenizers[model],args['eos_token_id'],r['choices'],r['gold'],128)
                assert all(g[k]==v for k,v in fresh.items()),(g['id'],fresh)
                if not fresh['valid_answer']:invalid[report['spec']['name']].append({'id':g['id'],'text':g['text'],'route':g['route']})
            clean_gen=[r for r in gen if not r['ambiguous_gold']]
            for key,field in [('answer_accuracy','correct'),('valid_answer_rate','valid_answer'),('strict_accuracy','strict_correct'),
                              ('terminated_rate','terminated'),('length_cap_rate','hit_length_cap'),('completed_answer_accuracy','completed_correct')]:
                assert abs(report['generation'][key]-100*np.mean([r[field] for r in clean_gen]))<1e-10
            generation_count+=len(gen)
    assert reference_replays==54
    selected=read(HERE/'SELECTION.json')['selected']
    for model,selection in selected.items():
        candidates=selection['candidates']
        assert selection['lr']==sorted(candidates,key=lambda x:(-x['accuracy'],x['candidate_nll'],x['lr']))[0]['lr']
        for candidate in candidates:assert sha(candidate['source'])==candidate['metrics_sha256']
        for seed in CONFIRMATION_SEEDS:
            fits=[read(p) for p in (HERE/'checkpoints').glob('*/spec.json') if read(p)['phase']=='confirmation' and read(p)['model']==model and read(p)['seed']==seed]
            assert len(fits)==1 and fits[0]['lr']==selection['lr']
    trained=[]
    for path in (HERE/'checkpoints').glob('*/TRAINING.json'):
        r=read(path);spec=read(path.parent/'spec.json');init=read(path.parent/'INITIALIZATION.json')
        assert r['frozen_parameters_bitwise_unchanged'] and r['optimizer_whitelist_verified']
        assert r['frozen_parameters_before_sha256']==r['frozen_parameters_after_sha256']
        assert sha(path.parent/'adapter.safetensors')==r['adapter_sha256']
        assert init['target_projection']=='q_proj' and init['zero_start_max_abs_logit_error']==0
        assert 0.96<init['budget_fraction']<=1
        assert r['steps']==(2 if spec['phase']=='smoke' else 364)
        trained.append({'name':spec['name'],'phase':spec['phase'],'parameters':init['trainable_parameters']})
    assert len(trained)==20,len(trained) # two first-attempt smokes, two repaired smokes, six tuning, ten confirmation
    write(HERE/'INVALID_ANSWERS.json',invalid)
    result={'at':now(),'status':'passed','source_sealed_files_verified':sum(r['verified_files'] for r in read(HERE/'REUSE_AUDIT.json')['sealed_sources'].values()),
            'predictions_recomputed':prediction_count,'generations_redecoded':generation_count,'old_candidate_replays':reference_replays,
            'old_prediction_flips':0,'max_old_logit_error':max_logit_error,'new_training_records':trained,
            'validation_selection_recomputed':True,'all_original_weights_frozen':True,'preflight_attempt1_retained':True,
            'invalid_answer_details':'INVALID_ANSWERS.json'}
    write(HERE/'FINAL_AUDIT.json',result);print(json.dumps({k:v for k,v in result.items() if k!='new_training_records'},indent=2))


if __name__=='__main__':main()
