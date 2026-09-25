import argparse,time
import torch
from transformers import AutoTokenizer
from common import *
from modeling import build,base_precision,load_adapter,frozen_digest,adapter_state,digest
from scoring import score,aggregate

def generate(model,tokens,tok,batch_size):
    from modeling import batch
    ids,mask,_=batch(tokens,False)
    with torch.inference_mode():
        out=model.generate(input_ids=ids,attention_mask=mask,do_sample=False,max_new_tokens=512,
            eos_token_id=151643,pad_token_id=151643,use_cache=True)
    results=[]
    for seq in out[:,ids.shape[1]:].tolist():
        ended=151643 in seq;seq=seq[:seq.index(151643)] if ended else seq
        results.append({'token_ids':seq,'text':tok.decode(seq,skip_special_tokens=False),'native_eos':ended,
            'capped':not ended and len(seq)>=512,'length':len(seq)})
    return results

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--probe',action='store_true');p.add_argument('--fewshot',action='store_true');a=p.parse_args()
    cp=Path(a.checkpoint);spec=read(cp/'spec.json');assert read(cp/'COMPLETE.json')['status']=='passed'
    for rel,h in read(HERE/'HELDOUT_FROZEN.json')['code_files'].items():assert sha(HERE/rel)==h,rel
    for rel,h in read(HERE/'CODE_FROZEN.json')['files'].items():assert sha(HERE/rel)==h,rel
    model,audit=build(spec);before=frozen_digest(model);expected=read(cp/'INITIALIZATION.json');assert before==expected['frozen_before']
    reuse={'checkpoint':str(cp),'spec_sha256':sha(cp/'spec.json'),'original_tensor_digest':before,'initialization_sha256':sha(cp/'INITIALIZATION.json')}
    if spec['arm']!='base':
        training=read(cp/'TRAINING.json');assert sha(cp/'adapter.safetensors')==training['adapter_sha256']
        load_adapter(model,cp/'adapter.safetensors');assert digest(adapter_state(model))==training['adapter_tensor_sha256']
        reuse['adapter_sha256']=training['adapter_sha256'];reuse['adapter_tensor_sha256']=training['adapter_tensor_sha256']
    base_precision(model,torch.float32);model.eval();tok=AutoTokenizer.from_pretrained(read(HERE/'models.json')[spec['model']]['path'],local_files_only=True)
    if a.probe:
        assert not a.fewshot
        tokens=read(HERE/f"tokens/{spec['task']}_{spec['model']}_pilot_dev.json")[:32]
        tag='baseline' if spec['arm']=='base' else 'final'
        ref=rows(HERE/'evaluations'/spec['name']/tag/'responses.jsonl')[:32]
        got=generate(model,tokens,tok,32);assert [r['id'] for r in tokens]==[r['id'] for r in ref]
        diffs=[tokens[i]['id'] for i,(x,y) in enumerate(zip(ref,got)) if x['token_ids']!=y['token_ids'] or x['native_eos']!=y['native_eos']]
        write(HERE/'heldout_probes'/f"{spec['name']}.json",{'at':now(),'status':'passed','reuse':reuse,'examples':32,'batch32_vs_saved_batch8_token_differences':diffs})
        print(spec['name'],'probe differences',len(diffs),flush=True);return
    decision=read(HERE/'HELDOUT_BATCH.json');bs=decision['batch_size'];split='fewshot_test' if a.fewshot else 'test'
    filename=f"tokens/{spec['task']}_{spec['model']}_{split}.json";datafile=f"data/{spec['task']}_{split}.jsonl"
    manifest=read(HERE/'HELDOUT_FROZEN.json')['data_files'];assert sha(HERE/filename)==manifest[filename];assert sha(HERE/datafile)==manifest[datafile]
    tokens=read(HERE/filename);data=rows(HERE/datafile);assert [r['id'] for r in tokens]==[r['id'] for r in data]
    tag='baseline' if spec['arm']=='base' else 'final';outdir=HERE/'heldout'/spec['name']/tag;outdir.mkdir(parents=True,exist_ok=False)
    write(outdir/'REUSE.json',reuse);records=[];started=time.monotonic()
    for start in range(0,len(tokens),bs):
        generated=generate(model,tokens[start:start+bs],tok,bs)
        for row,pred in zip(data[start:start+bs],generated):records.append(dict(pred,id=row['id'],**score(pred['text'],row)))
        write_rows(outdir/'responses.jsonl',records);progress={'done':len(records),'total':len(data),'seconds':time.monotonic()-started}
        write(outdir/'PROGRESS.json',progress);print(json.dumps(progress),flush=True)
    summary=aggregate(records,spec['task']);summary.update(native_eos_pct=100*sum(r['native_eos'] for r in records)/len(records),
        capped_pct=100*sum(r['capped'] for r in records)/len(records),mean_generated_tokens=sum(r['length'] for r in records)/len(records),
        seconds=time.monotonic()-started,spec=spec,tag=tag,eval_split=split,batch_size=bs,
        input_ids_sha256=htext(canonical([r['prompt_ids'] for r in tokens])),responses_sha256=sha(outdir/'responses.jsonl'))
    write(outdir/'SUMMARY.json',summary)
if __name__=='__main__':main()
