"""Compare local model files with Meta's pinned public git/LFS identities."""
import requests
from concurrent.futures import ThreadPoolExecutor
from common import *

ROOT=HERE.parents[2]
CONFIGS={'llama32_3b_base':('Llama-3.2-3B','13afe5124825b4f3751f836b40dafda64c1ed062'),
         'llama31_8b_base':('Llama-3.1-8B','d04e592bb4f6aa9cfee91e2e20afa771667e1d4b')}

def verify(item):
    alias,(name,rev)=item;path=ROOT/'models'/(name+'-Base')
    session=requests.Session();session.trust_env=False
    response=session.get(f'https://huggingface.co/api/models/meta-llama/{name}/revision/{rev}',params={'blobs':'true'},timeout=30)
    response.raise_for_status();metadata=response.json();assert metadata['sha']==rev
    write(HERE/'provenance'/f'{alias}_official_metadata.json',metadata)
    files={};checked={}
    for record in metadata['siblings']:
        fn=record['rfilename']
        if not (fn in ['config.json','generation_config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json','special_tokens_map.json'] or fn.endswith('.safetensors')):continue
        p=path/fn;assert p.is_file() and p.stat().st_size==record['size'],p
        h=sha(p);files[str(p)]=h
        if 'lfs' in record:
            assert h==record['lfs']['sha256'],p;checked[fn]={'kind':'LFS SHA256','value':h}
        else:
            content=p.read_bytes();git=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
            assert git==record['blobId'],p;checked[fn]={'kind':'git blob SHA1','value':git,'sha256':h}
    cfg=read(path/'config.json');index=read(path/'model.safetensors.index.json')
    assert set(index['weight_map'].values())<={Path(f).name for f in files}
    result={'path':str(path),'repo':'meta-llama/'+name,'revision':rev,'files':files,'hidden_size':cfg['hidden_size'],
            'tie_word_embeddings':cfg['tie_word_embeddings'],'training_stage':'pretrained_base','bos_token_id':cfg['bos_token_id'],'eos_token_id':cfg['eos_token_id']}
    return alias,result,checked

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:verified=list(pool.map(verify,CONFIGS.items()))
    write(HERE/'models.json',{a:r for a,r,c in verified})
    write(HERE/'MODEL_IDENTITY_AUDIT.json',{'at':now(),'status':'passed','official_sources':{a:c for a,r,c in verified}})
    print('Official model identities verified',list(CONFIGS),flush=True)
