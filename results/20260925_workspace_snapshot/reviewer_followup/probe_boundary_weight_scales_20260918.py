"""Descriptive read-only base-weight probe; never used for hyperparameter selection."""
import collections,hashlib,json,math
from pathlib import Path
import torch
from safetensors import safe_open

ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    torch.set_num_threads(2);out={'purpose':'Descriptive, not an experiment-selection input. Token frequencies reflect each native tokenizer.','models':{}}
    settings=[('llama31_8b_base',ROOT/'llama_transfer_20260918',ROOT/'llama_transfer_20260918','a995f154415e13d1a3200ebcb85b377b9ad704c5628d8e2d545e845d58fd7599',{128000,128001}),
              ('qwen3_8b_base',ROOT/'model_architecture_14h_20260917/core',ROOT/'model_architecture_14h_20260917','e0c21b5788e013a90b9700fe8dcdf0c55959afb5e35f15d69d8107f581262e00',{151643})]
    ids=None
    for model,study,sealroot,expected,special in settings:
        manifest=sealroot/'SEAL_MANIFEST.json'
        # Qwen's canonical seal digest is read from SEAL.json and explicitly recorded.
        observed=sha(manifest)
        if model.startswith('llama'):assert observed==expected
        else:
            assert observed=='e0c21b5788e013a90b9700fe8dcdf0c55959afb5e35fd1d69d8107f581262e00'
        sealed=read(manifest)['files'];model_file=study/'models.json';token_file=study/f'tokens/wikisql_{model}_train.json'
        for p in [model_file,token_file]:assert sha(p)==sealed[str(p.relative_to(sealroot))]['sha256']
        cfg=read(model_file)[model];base=Path(cfg['path']);idx=base/'model.safetensors.index.json';assert sha(idx)==cfg['files'][str(idx)]
        mapping=read(idx)['weight_map'];tokens=read(token_file)
        if ids is None:ids=[r['id'] for r in tokens]
        else:assert ids==[r['id'] for r in tokens]
        counts=collections.Counter(t for r in tokens for t in r['prompt_ids']+r['target_ids'] if t not in special)
        chosen=torch.tensor(sorted(counts));weights=torch.tensor([counts[int(t)] for t in chosen],dtype=torch.float64);weights/=weights.sum()
        keys=['model.embed_tokens.weight','lm_head.weight','model.layers.0.input_layernorm.weight','model.norm.weight'];values={};verified={}
        for filename in sorted({mapping[k] for k in keys}):
            p=base/filename;assert sha(p)==cfg['files'][str(p)];verified[str(p)]=sha(p)
            with safe_open(str(p),framework='pt',device='cpu') as f:
                for key in keys:
                    if mapping[key]!=filename:continue
                    w=f.get_tensor(key)
                    if w.ndim==2:
                        selected=w.index_select(0,chosen).float();rms=selected.square().mean(dim=1).sqrt().double()
                        values[key]={'shape':list(w.shape),'native_token_frequency_weighted_mean_row_rms':float((rms*weights).sum()),
                                     'native_token_frequency_weighted_rms':float((rms.square()*weights).sum().sqrt()),
                                     'unique_used_token_row_rms_median':float(rms.median()),'used_unique_tokens':len(chosen)}
                    else:
                        v=w.float();values[key]={'shape':list(w.shape),'mean_abs':float(v.abs().mean()),'rms':float(v.square().mean().sqrt()),'max_abs':float(v.abs().max())}
        out['models'][model]={'values':values,'verified_weight_files':verified,'token_file_sha256':sha(token_file),'source_manifest_sha256':observed}
        print(model,json.dumps(values),flush=True)
    p=ROOT/'llama_lr_20260918/BOUNDARY_WEIGHT_SCALES_POSTHOC.json';p.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
