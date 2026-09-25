"""Use completion prompts and a uniquely represented native EOS for Base SFT."""
import numpy as np
from transformers import AutoTokenizer
from common import *

def main():
    old=HERE.parent/'standalone_single_boundary_sciq_20260915'
    aborted=HERE.parent/'standalone_base_sciq_20260916'
    seal=read(old/'ARTIFACT_MANIFEST.json')
    for r in seal['files']:assert sha(old/r['path'])==r['sha256']
    copied={};bindings={}
    for name in ['common.py','generation.py','evaluate.py','source/affine_adapter.py']:
        assert sha(HERE/name)==sha(old/name);copied[name]=sha(HERE/name);bindings[name]=name
    for name in ['train','validation','test']:
        rel=f'data/{name}.jsonl';assert sha(HERE/rel)==sha(old/rel);copied[rel]=sha(HERE/rel);bindings[rel]=rel
    assert sha(HERE/'source/PRIOR_DATA_AUDIT.json')==sha(old/'DATA_AUDIT.json')
    copied['source/PRIOR_DATA_AUDIT.json']=sha(HERE/'source/PRIOR_DATA_AUDIT.json');bindings['source/PRIOR_DATA_AUDIT.json']='DATA_AUDIT.json'
    audit=read(old/'DATA_AUDIT.json');audit['tokenization']={};audit['model_files_verified']={};encodings={}
    duplicate=read(HERE/'BASE_SPECIAL_TOKEN_DIAGNOSTIC.json')
    for model,cfg in read(HERE/'models.json').items():
        for p,h in cfg['files'].items():assert sha(p)==h
        audit['model_files_verified'].update(cfg['files'])
        tok=AutoTokenizer.from_pretrained(cfg['path'],local_files_only=True)
        eos=read(Path(cfg['path'])/'config.json')['eos_token_id'];assert eos==tok.eos_token_id==151643
        assert duplicate[model]['native_eos_identical_rows']==[eos]
        assert duplicate[model]['im_end_identical_count']>1
        labels=[tok.encode(letter,add_special_tokens=False) for letter in 'ABCD'];assert all(len(x)==1 for x in labels)
        meta={'label_ids':[x[0] for x in labels],'eos_id':eos,'pad_id':eos,'splits':{},'serialization':'plain task completion: common.prompt + newline newline Answer: newline; no chat template or chat special tokens'}
        def encode_native(row,rotation=0):
            text=prompt(row,rotation)+'\n\nAnswer:\n';ids=tok.encode(text,add_special_tokens=False)
            gold=(row['gold']-rotation)%4
            assert tok.encode(text+'ABCD'[gold],add_special_tokens=False)==ids+[meta['label_ids'][gold]]
            assert not any(x>=151643 for x in ids),'No special tokens in plain prompts'
            return {k:row[k] for k in ['id','ambiguous_gold']}|{'input_ids':ids,'gold':gold,'rotation':rotation}
        for split in ['train','validation','test']:
            rr=rows(HERE/'data'/f'{split}.jsonl');ee=[encode_native(r) for r in rr]
            write(HERE/'tokens'/f'{model}_{split}.json',ee)
            meta['splits'][split]={'rows':len(ee),'max_prompt_tokens':max(len(r['input_ids']) for r in ee)}
            if split=='test':write(HERE/'tokens'/f'{model}_test_rotations.json',[encode_native(r,i) for i in range(4) for r in rr])
        audit['tokenization'][model]=meta
        encodings[model]={'example_training_prompt':tok.decode(read(HERE/'tokens'/f'{model}_train.json')[0]['input_ids']),
            'native_eos_id':eos,'unique_native_eos_row_verified':True,'full_prompt_answer_boundary_roundtrip':True}
    audit['native_base_protocol']={'reason':'Avoid indistinguishable unused ChatML target rows in frozen Base vocabulary','encodings':encodings,
        'previous_chat_tokens_reused':False,'data_split_and_labels_reused':True,'prior_attempt':str(aborted)}
    write(HERE/'DATA_AUDIT.json',audit)
    write(HERE/'REUSE_AUDIT.json',{'source_root':str(old),'original_manifest_sha256':sha(old/'ARTIFACT_MANIFEST.json'),
        'original_files_verified':len(seal['files']),'copied_source_sha256':copied,'source_bindings':bindings,
        'model_provenance_sha256':sha(HERE/'MODEL_PROVENANCE.json'),'generated_token_sha256':{str(p.relative_to(HERE)):sha(p) for p in (HERE/'tokens').glob('*.json')},
        'aborted_protocol_sha256':sha(aborted/'ABORTED_PROTOCOL.json'),'special_token_diagnostic_sha256':sha(HERE/'BASE_SPECIAL_TOKEN_DIAGNOSTIC.json'),
        'new_adapter_initialization':'fresh zero residual; no aborted/previous fitted adapters reused'})
    print('NATIVE BASE PREPARATION PASSED',flush=True)
if __name__=='__main__':main()
