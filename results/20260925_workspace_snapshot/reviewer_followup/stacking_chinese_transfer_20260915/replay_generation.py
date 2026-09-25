"""Reproduce the first four original CMRC batches without changing main outputs."""
import argparse
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from common import HERE, checkpoint_hashes, now, prompt_ids, read, rows, sha, write
from evaluate import legacy, batches, generation, torch, set_seed

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);a=p.parse_args()
    manifest=read(HERE/'manifest.json');j=next(x for x in manifest['endpoints'] if x['name']==a.name)
    assert checkpoint_hashes(j['checkpoint'])==j['checkpoint_hashes']
    for path,h in manifest['local_sha256'].items():assert sha(path)==h,path
    torch.set_num_threads(4);set_seed(0)
    model,tokenizer,model_path,_=legacy.load_model(argparse.Namespace(run_dir=j['checkpoint'],model_path=None,affine_ablation='none',device='cuda'))
    data=rows(HERE/'data/cmrc_eval.jsonl')
    items=[{'ids':prompt_ids(tokenizer,'cmrc',r),'row':r} for r in data]
    first=[]
    for i,b in enumerate(batches(items,16,8192)):
        if i==4:break
        first.extend(x['row'] for x in b)
    config=read(Path(model_path)/'generation_config.json');eos=config.get('eos_token_id',tokenizer.eos_token_id)
    eos=sorted(set((eos if isinstance(eos,list) else [eos])+tokenizer.encode('<|im_end|>',add_special_tokens=False)))
    out=HERE/'generation_replay'/a.name;out.mkdir(parents=True,exist_ok=True)
    got=generation(model,tokenizer,first,out,eos,False)
    old_path=HERE/'outputs'/a.name/'cmrc_generation.jsonl';old=rows(old_path)[:len(got)]
    assert len(old)==len(got)
    assert all(x==y for x,y in zip(old,got)),'Generation replay mismatch'
    write(out/'AUDIT.json',{'status':'passed','checked_at':now(),'examples':len(got),'original_batches':4,
        'all_token_ids_text_and_scores_identical':True,'original_output_sha256':sha(old_path),
        'replay_output_sha256':sha(out/'cmrc_generation.jsonl'),'checkpoint_hashes':j['checkpoint_hashes'],
        'implementation_sha256':sha(HERE/'replay_generation.py')})
    print(a.name,'exact generation replay passed',len(got),flush=True)

if __name__=='__main__':main()
