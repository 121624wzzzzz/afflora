"""Check actual saved shuffle orders and fresh randomizations across seeds."""
import torch
from collections import defaultdict
from settings import *
from run import specs

def main():
    assert read(HERE/'CONFIRMATION_COMPLETE.json')['jobs']==30
    lengths={m:len(read(HERE/'tokens'/f'{m}_train.json')) for m in read(HERE/'models.json')}
    groups=defaultdict(list);paired=defaultdict(set);checked=0
    for phase in ['smoke','tuning','confirmation']:
        for s in specs(phase):
            cp=Path(s['checkpoint']);expected=torch.randperm(lengths[s['model']],generator=torch.Generator().manual_seed(s['seed'])).tolist()
            if phase=='smoke':expected=expected[:64]
            assert read(cp/'TRAIN_ORDER.json')['indices']==expected,s['name']
            checked+=1
            init=read(cp/'INITIALIZATION.json')
            if s['arm'] in ['input','output']:paired[(s['model'],s['seed'])].add(init['linear_init_sha256'])
            if phase=='confirmation':groups[(s['model'],s['arm'])].append((s['seed'],init['adapter_init_sha256'],read(cp/'TRAINING.json')['tensor_sha256']))
    assert all(len(v)==1 for v in paired.values()),'Input/output linear initialization mismatch'
    for k,rr in groups.items():
        assert sorted(x[0] for x in rr)==CONFIRMATION_SEEDS
        assert len({x[1] for x in rr})==len({x[2] for x in rr})==5,k
    write(HERE/'INDEPENDENCE_AUDIT.json',{'at':now(),'status':'passed','saved_training_orders_reproduced':checked,
        'same_shuffle_across_arms_models_within_seed':True,'input_output_linear_initialization_paired_within_seed':True,
        'five_distinct_initial_and_final_adapter_states_per_model_arm':True,'groups':{f'{m}/{a}':[{'seed':s,'initial_tensor_sha256':i,'final_tensor_sha256':f} for s,i,f in v] for (m,a),v in groups.items()}})
    print('Randomization audit passed:',checked,'training orders and',len(groups),'five-seed groups')
if __name__=='__main__':main()
