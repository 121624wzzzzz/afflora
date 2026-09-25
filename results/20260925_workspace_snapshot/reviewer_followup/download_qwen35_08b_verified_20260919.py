"""Download public Qwen ModelScope distribution; verify against pinned Qwen HF identities."""
import concurrent.futures,hashlib,json,time,math,shutil
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'lora/reviewer_followup/qwen35_08b_20260919'
MODEL=ROOT/'models/Qwen3.5-0.8B-Base'
REV='dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68'
def session():
    s=requests.Session();s.trust_env=False;return s
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def verify(p,r):
    assert p.stat().st_size==r['size'],p.name
    h=sha(p)
    if r.get('lfs'):assert h==r['lfs']['sha256'],p.name
    else:
        raw=p.read_bytes();assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==r['blobId'],p.name
    return h
def main():
    STUDY.mkdir(exist_ok=True);MODEL.mkdir(exist_ok=True)
    s=session();r=s.get(f'https://huggingface.co/api/models/Qwen/Qwen3.5-0.8B-Base/revision/{REV}',params={'blobs':'true'},timeout=40);r.raise_for_status();official=r.json();assert official['sha']==REV
    r=s.get('https://www.modelscope.cn/api/v1/models/Qwen/Qwen3.5-0.8B-Base/repo/files',params={'Revision':'master','Recursive':'true'},timeout=40);r.raise_for_status();mirror=r.json();assert mirror['Success']
    write(STUDY/'provenance/qwen35_08b_official_metadata.json',official)
    write(STUDY/'provenance/qwen35_08b_public_distribution_metadata.json',mirror)
    index={r['Path']:r for r in mirror['Data']['Files'] if r['Type']=='blob'}
    names={'config.json','model.safetensors.index.json','tokenizer.json','tokenizer_config.json','merges.txt','vocab.json','preprocessor_config.json','video_preprocessor_config.json','README.md','LICENSE'}
    records=[r for r in official['siblings'] if r['rfilename'] in names or r['rfilename'].endswith('.safetensors')]
    assert len([r for r in records if r['rfilename'].endswith('.safetensors')])==1
    for r in records:
        m=index[r['rfilename']];assert m['Size']==r['size']
        if r.get('lfs'):assert m['Sha256']==r['lfs']['sha256'],r['rfilename']
    def download(r):
        fn=r['rfilename'];dest=MODEL/fn;m=index[fn]
        if dest.exists():return fn,verify(dest,r)
        part=dest.with_suffix(dest.suffix+'.partial')
        url=f"https://www.modelscope.cn/models/Qwen/Qwen3.5-0.8B-Base/resolve/{m['Revision']}/{fn}"
        # Independent verified byte ranges avoid a single slow connection becoming
        # the tail of the download. Preserve the previously downloaded prefix.
        start=part.stat().st_size if part.exists() else 0
        planpath=MODEL/(fn+'.ranges.json')
        if planpath.exists() or (fn.endswith('.safetensors') and r['size']-start>256*1024*1024):
            if planpath.exists():
                plan=json.loads(planpath.read_text());assert plan['start']==start and plan['total']==r['size']
            else:
                chunk=math.ceil((r['size']-start)/4)
                plan={'start':start,'total':r['size'],'ranges':[[a,min(a+chunk,r['size'])-1] for a in range(start,r['size'],chunk)]}
                write(planpath,plan)
            def fetch_range(pair):
                lo,hi=pair;q=MODEL/(fn+f'.range_{lo}_{hi}');wanted=hi-lo+1
                for attempt in range(6):
                    offset=q.stat().st_size if q.exists() else 0;assert offset<=wanted
                    if offset==wanted:break
                    try:
                        response=session().get(url,headers={'Range':f'bytes={lo+offset}-{hi}'},stream=True,timeout=(10,30))
                        response.raise_for_status()
                        assert response.status_code==206 and response.headers['Content-Range']==f'bytes {lo+offset}-{hi}/{r["size"]}'
                        with q.open('ab') as f:
                            for block in response.iter_content(8*1024*1024):f.write(block)
                        assert q.stat().st_size==wanted
                    except (requests.RequestException,OSError) as e:
                        print(json.dumps({'file':fn,'range':[lo,hi],'attempt':attempt+1,'network_error':type(e).__name__}),flush=True)
                        if attempt==5:raise
                assert q.stat().st_size==wanted
                print(json.dumps({'file':fn,'range_completed':[lo,hi]}),flush=True)
                return q
            futures=[ranges_pool.submit(fetch_range,pair) for pair in plan['ranges']]
            pieces=[future.result() for future in futures]
            assembled=dest.with_suffix(dest.suffix+'.assembled')
            with assembled.open('wb') as out:
                for q in ([part] if start else [])+pieces:
                    with q.open('rb') as src:shutil.copyfileobj(src,out,8*1024*1024)
            h=verify(assembled,r);assembled.replace(dest)
            for q in pieces:q.unlink()
            if part.exists():part.unlink()
            planpath.unlink();print(json.dumps({'file':fn,'verified':True,'bytes':r['size'],'segmented':True}),flush=True)
            return fn,h
        for attempt in range(3):
            try:
                start=part.stat().st_size if part.exists() else 0
                if start==r['size']:break
                response=session().get(url,headers={'Range':f'bytes={start}-'} if start else {},stream=True,timeout=(30,120));response.raise_for_status()
                if start and response.status_code!=206:start=0
                if response.status_code==206:assert response.headers['Content-Range'].startswith(f'bytes {start}-')
                done=start;last=time.monotonic()
                with part.open('ab' if start else 'wb') as f:
                    for block in response.iter_content(8*1024*1024):
                        f.write(block);done+=len(block)
                        if time.monotonic()-last>30:
                            print(json.dumps({'file':fn,'bytes':done,'total':r['size']}),flush=True);last=time.monotonic()
                break
            except (requests.RequestException,OSError) as e:
                print(json.dumps({'file':fn,'attempt':attempt+1,'network_error':type(e).__name__}),flush=True)
                if attempt==2:raise
        h=verify(part,r);part.replace(dest);print(json.dumps({'file':fn,'verified':True,'bytes':r['size']}),flush=True)
        return fn,h
    small=[r for r in records if not r['rfilename'].endswith('.safetensors')]
    big=[r for r in records if r['rfilename'].endswith('.safetensors')]
    checked=dict(map(download,small))
    # One global cap avoids multiplying file concurrency by range concurrency.
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ranges_pool:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:checked.update(pool.map(download,big))
    cfg=json.loads((MODEL/'config.json').read_text());weight_index=json.loads((MODEL/'model.safetensors.index.json').read_text())
    assert set(weight_index['weight_map'].values())<=set(checked)
    from transformers import AutoConfig
    resolved=AutoConfig.from_pretrained(MODEL,local_files_only=True); cfg=cfg['text_config']; resolved=resolved.text_config
    result={'status':'passed','repo':'Qwen/Qwen3.5-0.8B-Base','revision':REV,'path':str(MODEL),'training_stage':'pretrained_base',
            'distribution':'public Qwen/Qwen3.5-0.8B-Base on ModelScope; every runtime file verified against official Qwen HF revision',
            'files':{str(MODEL/k):v for k,v in checked.items()},'linear_attention_layers':cfg['layer_types'].count('linear_attention'),'hidden_size':cfg['hidden_size'],'tie_word_embeddings':resolved.tie_word_embeddings,
            'tying_declaration':'Qwen3.5 text config; checked again by actual parameter pointer at runtime',
            'bos_token_id':cfg.get('bos_token_id'),'eos_token_id':cfg['eos_token_id'],'pad_token_id':cfg.get('pad_token_id')}
    write(STUDY/'models.json',{'qwen35_08b_base':result});write(STUDY/'MODEL_IDENTITY_AUDIT.json',result)
    print(json.dumps({'status':'passed','files':len(checked),'revision':REV}),flush=True)
if __name__=='__main__':main()
