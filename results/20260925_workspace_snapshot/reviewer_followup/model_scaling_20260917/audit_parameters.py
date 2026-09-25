"""Independent scope and provenance checks; no model execution or source mutation."""
from common import *

def main():
    models=read(HERE/'models.json');frozen={};runs=[]
    for m,cfg in models.items():
        for p,h in cfg['files'].items():assert sha(p)==h,p
    for p in sorted((HERE/'checkpoints').glob('*/INITIALIZATION.json')):
        spec=read(p.parent/'spec.json');a=read(p);arm=spec['arm'];d=models[spec['model']]['hidden_size']
        names=a['trainable_names'];expected_boundary=(33*d if arm in ['input','both','hidden_both'] else 0)+(32*d if arm in ['output','both','hidden_both','hidden_output'] else 0)
        assert a['parameter_groups']['boundary']==expected_boundary
        if not arm.startswith('hidden'):assert a['parameter_groups']['hidden']==0
        if arm=='base':assert not names and a['trainable_parameters']==0
        if arm=='input':assert all(n.startswith('boundary_input.') for n in names)
        if arm=='output':assert all(n.startswith('boundary_output.') for n in names)
        if arm in ['hidden','hidden_budget']:assert all('.lora_A.' in n or '.lora_B.' in n for n in names)
        assert a['trainable_parameters']==sum(a['parameter_groups'].values())
        assert a['trainable_parameters']==read(HERE/'BUDGET_PLAN.json')[spec['model']][arm]
        assert a['tied_weights']==models[spec['model']]['tie_word_embeddings']
        assert frozen.setdefault(spec['model'],a['frozen_before'])==a['frozen_before'],(spec['name'],'different original weights')
        assert a['preflight']['zero_residual_max_abs_error']==0
        runs.append({'name':spec['name'],'arm':arm,'trainable_parameters':a['trainable_parameters']})
    files=0;examples=0
    for p in sorted((HERE/'tokens').glob('*.json')):
        for r in read(p):
            assert r['target_ids'][-1]==151643 and r['target_ids'].count(151643)==1,(p,r['id'])
            assert not ({151644,151645}&set(r['prompt_ids']+r['target_ids'])),(p,r['id'])
            examples+=1
        files+=1
    write(HERE/'PARAMETER_AUDIT.json',{'at':now(),'status':'passed','model_files_verified':sum(len(c['files']) for c in models.values()),
        'original_weight_tensor_digests':frozen,'runs':runs,'token_files_checked':files,'token_records_checked':examples})
    print('passed',len(runs),'parameter scopes;',examples,'token records')
if __name__=='__main__':main()
