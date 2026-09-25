import numpy as np
from scipy.special import logsumexp
from support import *

def main():
    reuse=read(HERE/'REUSE_AUDIT.json')
    assert sha(OLD/'ARTIFACT_MANIFEST.json')==reuse['source_manifest_sha256']
    for path,h in SEAL.items():assert sha(path)==h,path
    for path,h in reuse['verified_copies'].items():assert sha(HERE/Path(path).relative_to(OLD))==h,path
    model_files=0
    for cfg in read(HERE/'models.json').values():
        for path,h in cfg['files'].items():assert sha(path)==h,path;model_files+=1
    plan=read(HERE/'ABLATION_PLAN.json');ep=read(HERE/'EOS_FROZEN_PLAN.json')
    for path,h in [('ablate.py',plan['script_sha256']),('DESIGN.md',plan['design_sha256']),('eos_probe.py',ep['script_sha256']),('EOS_PROBE_PLAN.md',ep['plan_sha256'])]:assert sha(HERE/path)==h
    assert read(HERE/'ABLATIONS_COMPLETE.json')['jobs']==10
    assert read(HERE/'EOS_COMPLETE.json')['jobs']==10
    count_label=count_eos=replays=0;error_eos_prefix=0.
    for s in plan['tasks']:
        tokens=read(HERE/'tokens'/f"{s['model']}_test_rotations.json")
        clean=np.array([not t['ambiguous_gold'] for t in tokens]);canonical=clean&np.array([t['rotation']==0 for t in tokens])
        gold=np.array([t['gold'] for t in tokens]);labelids=np.array([32,33,34,35])
        a=read(HERE/'ablations'/s['name']/'RESULTS.json');e=read(HERE/'eos'/s['name']/'RESULTS.json')
        assert a['adapter_tensors_unchanged'] and e['adapter_tensors_unchanged']
        assert a['full_replay_max_abs_logit_error']==0 and a['full_replay_prediction_flips']==0
        assert a['checkpoint_sha256']==sha(Path(s['checkpoint'])/'adapter.safetensors')
        assert e['prefix_label_replay_max_abs_error']<1e-4
        error_eos_prefix=max(error_eos_prefix,e['prefix_label_replay_max_abs_error'])
        for directory,r in [('ablations',a),('eos',e)]:assert sha(HERE/directory/s['name']/'scores.npz')==r['scores_sha256']
        original=np.array([r['label_logits'] for r in load_predictions(s)],dtype=np.float32)
        with np.load(HERE/'ablations'/s['name']/'scores.npz') as ar,np.load(HERE/'eos'/s['name']/'scores.npz') as er:
            assert np.array_equal(ar['full'],original);replays+=len(original)
            for mode,metrics in a['metrics'].items():
                scores=ar[mode].astype(np.float64);assert scores.shape==(4000,4) and np.isfinite(scores).all()
                nll=logsumexp(scores,axis=-1)-scores[np.arange(4000),gold]
                logmass=ar[mode+'_log_label_mass'];unrestricted=ar[mode+'_unrestricted'];pred=scores.argmax(-1)
                assert np.isfinite(logmass).all() and (logmass<=1e-10).all()
                for split,mask in [('canonical',canonical),('rotations',clean)]:
                    assert metrics[split]['n']==int(mask.sum())
                    check={'accuracy':100*np.mean(pred[mask]==gold[mask]),'candidate_nll':nll[mask].mean(),
                        'label_mass_penalty':-logmass[mask].mean(),'full_vocab_label_nll':(nll-logmass)[mask].mean(),
                        'valid_first_label_percent':100*np.isin(unrestricted[mask],labelids).mean(),
                        'unrestricted_first_token_accuracy':100*(unrestricted[mask]==labelids[gold[mask]]).mean()}
                    for k,v in check.items():assert abs(v-metrics[split][k])<1e-11,(s['name'],mode,split,k)
                count_label+=4000
                if mode in er:
                    eos=er[mode];assert eos.shape==(4000,) and np.isfinite(eos).all() and (eos>=-1e-10).all()
                    for split,mask in [('canonical',canonical),('rotations',clean)]:assert abs(eos[mask].mean()-e['eos_nll'][mode][split])<1e-12
                    count_eos+=4000
    result={'at':now(),'status':'passed','sealed_original_files_rechecked':len(SEAL),'copied_files_rechecked':len(reuse['verified_copies']),
        'model_files_rechecked':model_files,'inference_jobs':20,'label_score_rows_audited':count_label,'eos_score_rows_audited':count_eos,
        'original_full_path_rows_replayed_bitwise':replays,'max_eos_prefix_label_replay_error':error_eos_prefix,
        'new_training_jobs':0,'original_sealed_study_unchanged':True,'notes':'New analyses are post hoc; no selected test configuration or changed primary test.'}
    assert count_label==168000 and count_eos==160000 and replays==40000
    write(HERE/'FINAL_AUDIT.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
