"""Verify immutable reuse and official Base checkpoints before training."""
import hashlib,json,shutil,requests
from pathlib import Path
from transformers import AutoTokenizer

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'standalone_single_boundary_sciq_20260915'
ROOT=HERE.parents[2]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
    seal=read(OLD/'ARTIFACT_MANIFEST.json')
    for r in seal['files']:assert sha(OLD/r['path'])==r['sha256'],r['path']
    copied={};bindings={};derived={}
    def copy(src,dst,immutable=True):
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        assert sha(src)==sha(dst)
        (copied if immutable else derived)[str(dst.relative_to(HERE))]=sha(src)
        bindings[str(dst.relative_to(HERE))]=str(src.relative_to(OLD))
    for name in ['common.py','generation.py','evaluate.py','source/affine_adapter.py']:
        copy(OLD/name,HERE/name)
    for name in ['modeling.py','train.py','run.py','analyze.py','final_audit.py']:
        copy(OLD/name,HERE/name,False)
    for src in (OLD/'data').glob('*.jsonl'):copy(src,HERE/'data'/src.name)
    copy(OLD/'DATA_AUDIT.json',HERE/'source/PRIOR_DATA_AUDIT.json')
    models={};provenance={};token_records={};session=requests.Session();session.trust_env=False
    configs=[('qwen3_06b_base','qwen3_06b_chat','Qwen3-0.6B-Base','Qwen3-0.6B-Base'),
             ('qwen25_15b_base','qwen25_15b_chat','Qwen2.5-1.5B','Qwen2.5-1.5B-Base')]
    for key,oldkey,repo,localdir in configs:
        meta=read(HERE/f'hub_metadata_{repo}.json');target=HERE/'verified_models'/repo;target.mkdir(parents=True,exist_ok=True)
        records={};local=ROOT/'models'/localdir
        for entry in meta['siblings']:
            name=entry['rfilename']
            if '/' in name or not name.endswith(('.json','.txt','.safetensors','.jinja')):continue
            dest=target/name;source=local/name
            def matches(p):
                if not p.exists():return False
                if entry.get('lfs'):return sha(p)==entry['lfs']['sha256']
                b=p.read_bytes();return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==entry['blobId']
            accepted=matches(source)
            if not dest.exists():
                if accepted:
                    if name.endswith('.safetensors'):dest.symlink_to(source)
                    else:shutil.copy2(source,dest)
                else:
                    assert not name.endswith('.safetensors'),'Unexpected weight mismatch; inspect before downloading'
                    r=session.get(f"https://huggingface.co/Qwen/{repo}/resolve/{meta['sha']}/{name}",timeout=30);r.raise_for_status();dest.write_bytes(r.content)
            assert matches(dest),(repo,name)
            records[name]={'sha256':sha(dest),'bytes':dest.stat().st_size,'official_blob':entry['blobId'],'local_reused':accepted}
            print(repo,name,'verified',flush=True)
        cfg=read(target/'config.json');models[key]={'path':str(target),'hidden_size':cfg['hidden_size'],'tie_word_embeddings':cfg['tie_word_embeddings'],
            'files':{str(target/k):v['sha256'] for k,v in records.items()},'repo':'Qwen/'+repo,'revision':meta['sha'],'training_stage':'pretraining_only','previous_chat_model_key':oldkey}
        provenance[key]={'repo':'Qwen/'+repo,'revision':meta['sha'],'official_metadata_url':'https://huggingface.co/api/models/Qwen/'+repo+'?blobs=true','files':records}
        tok=AutoTokenizer.from_pretrained(target,local_files_only=True)
        oldtok=AutoTokenizer.from_pretrained(read(OLD/'models.json')[oldkey]['path'],local_files_only=True)
        assert tok.get_vocab()==oldtok.get_vocab(),'Token IDs must have identical meaning'
        checked=0
        for split in ['train','validation','test','test_rotations']:
            src=OLD/'tokens'/f'{oldkey}_{split}.json';dst=HERE/'tokens'/f'{key}_{split}.json';copy(src,dst)
            for r in read(dst):
                text=oldtok.decode(r['input_ids'],skip_special_tokens=False)
                assert tok.encode(text,add_special_tokens=False)==r['input_ids'],(key,split,r['id'])
                checked+=1
        token_records[key]={'previous_model_key':oldkey,'identical_vocabulary_mapping':True,'all_reused_prompt_sequences_roundtrip':checked,
            'serialization':'Exact previous post-trained-model task serialization, explicitly taught to Base during SFT; not assumed to be Base-native behavior.'}
    write(HERE/'models.json',models);write(HERE/'MODEL_PROVENANCE.json',{'status':'passed','models':provenance})
    audit=read(OLD/'DATA_AUDIT.json');oldmeta=audit['tokenization'];audit['tokenization']={k:oldmeta[v['previous_chat_model_key']] for k,v in models.items()}
    audit['model_files_verified']={p:h for cfg in models.values() for p,h in cfg['files'].items()};audit['base_tokenization_verification']=token_records
    write(HERE/'DATA_AUDIT.json',audit)
    write(HERE/'REUSE_AUDIT.json',{'source_root':str(OLD),'original_manifest_sha256':sha(OLD/'ARTIFACT_MANIFEST.json'),
        'original_files_verified':len(seal['files']),'copied_source_sha256':copied,'source_bindings':bindings,'derived_code_original_sha256':derived,
        'model_provenance_sha256':sha(HERE/'MODEL_PROVENANCE.json'),'new_adapter_initialization':'fresh zero residual; no trained adapters reused'})
    print('PREPARATION PASSED',flush=True)
if __name__=='__main__':main()
