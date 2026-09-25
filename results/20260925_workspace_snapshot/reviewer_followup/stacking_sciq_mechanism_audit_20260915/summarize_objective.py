import numpy as np
from support import *
from analyze_predictions import interval

def main():
    rr=[read(p) for p in sorted((HERE/'eos').glob('*/RESULTS.json'))]
    assert len(rr)==10
    out={'at':now(),'scope':'Post hoc teacher-forced test objective after gold label; not free generation.','models':{}}
    for model in read(HERE/'models.json'):
        records=[r for r in rr if r['spec']['model']==model]
        assert [r['spec']['seed'] for r in records]==list(range(2002,2007))
        values={m:{s:{k:[] for k in ['eos_nll','full_vocab_label_nll','two_token_mean_nll']} for s in ['canonical','rotations']}
                for m in ['full','input_only','output_only','both_off']}
        for r in records:
            a=read(HERE/'ablations'/r['spec']['name']/'RESULTS.json')
            assert sha(HERE/'eos'/r['spec']['name']/'scores.npz')==r['scores_sha256']
            for mode in values:
                for split in values[mode]:
                    eos=r['eos_nll'][mode][split];label=a['metrics'][mode][split]['full_vocab_label_nll']
                    for key,v in [('eos_nll',eos),('full_vocab_label_nll',label),('two_token_mean_nll',.5*(eos+label))]:values[mode][split][key].append(v)
        m={'modes':{mode:{split:{key:interval(v) for key,v in metrics.items()} for split,metrics in splits.items()} for mode,splits in values.items()},'effects':{}}
        for effect,left,right in [('output_effect_input_on','full','input_only'),('full_minus_both_off','full','both_off')]:
            m['effects'][effect]={split:{key:interval(np.array(values[left][split][key])-values[right][split][key]) for key in values[left][split]} for split in values[left]}
        out['models'][model]=m
    write(HERE/'OBJECTIVE_ANALYSIS.json',out)
    for model,m in out['models'].items():
        print(model)
        print('output on effect',m['effects']['output_effect_input_on']['rotations'])
        print('EOS', {mode:r['rotations']['eos_nll']['mean'] for mode,r in m['modes'].items()})

if __name__=='__main__':main()
