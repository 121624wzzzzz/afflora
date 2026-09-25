"""Read-only checks of independent seeds and data order, including paired controls."""
import torch
from settings import *


def main():
    source=HERE.parent/'standalone_base_native_sciq_20260916'
    n=len(rows(HERE/'data/train.jsonl'));checks=[];states={}
    for file in (HERE/'checkpoints').glob('*/TRAINING.json'):
        cp=file.parent;spec=read(cp/'spec.json');init=read(cp/'INITIALIZATION.json');train=read(file)
        order=torch.randperm(n,generator=torch.Generator().manual_seed(spec['seed'])).tolist()
        if spec['phase']=='smoke':order=order[:64]
        assert read(cp/'TRAIN_ORDER.json')['indices']==order
        if spec['phase'] in ['confirmation','tuning']:
            candidates=list((source/'checkpoints').glob(f"{spec['phase']}_{spec['model']}_input_sd{spec['seed']}_*/TRAIN_ORDER.json"))
            assert candidates
            for reference in candidates:
                assert read(reference)['indices']==order
                original_init=read(reference.parent/'INITIALIZATION.json')
                assert original_init['frozen_parameters_before_sha256']==init['frozen_parameters_before_sha256']
        if spec['phase']=='confirmation':
            states.setdefault(spec['model'],[]).append({'seed':spec['seed'],'initial':init['adapter_init_sha256'],'final':train['tensor_sha256']})
        checks.append({'name':spec['name'],'order_regenerated':True,'source_input_order_matched':spec['phase']!='smoke'})
    for model,records in states.items():
        assert sorted(r['seed'] for r in records)==CONFIRMATION_SEEDS
        assert len({r['initial'] for r in records})==len({r['final'] for r in records})==5
    assert len(checks)==20 and len(states)==2
    write(HERE/'INDEPENDENCE_AUDIT.json',{'at':now(),'status':'passed','training_orders':checks,'confirmation_states':states,
                                       'same_data_order_as_paired_alora':True,'all_five_initial_and_final_states_distinct':True})
    print('20 training orders verified; both five-seed controls independent and paired to the original input adapters.')


if __name__=='__main__':main()
