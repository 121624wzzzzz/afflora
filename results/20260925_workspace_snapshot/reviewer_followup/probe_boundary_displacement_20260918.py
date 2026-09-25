"""Posthoc E displacement on training tokens; no forward pass or model selection."""
import collections,json,math
from pathlib import Path
import torch
from safetensors import safe_open
from probe_boundary_weight_scales_20260918 import ROOT,read,sha

def main():
    torch.set_num_threads(2);out={'purpose':'Posthoc FP32 boundary-map evaluation on native training token frequencies. Not downstream evidence.','models':{}}
    scales=read(ROOT/'llama_lr_20260918/BOUNDARY_WEIGHT_SCALES_POSTHOC.json')
    for model,study,sealroot,special in [('llama31_8b_base',ROOT/'llama_transfer_20260918',ROOT/'llama_transfer_20260918',{128000,128001}),
                                     ('qwen3_8b_base',ROOT/'model_architecture_14h_20260917/core',ROOT/'model_architecture_14h_20260917',{151643})]:
        assert sha(sealroot/'SEAL_MANIFEST.json')==scales['models'][model]['source_manifest_sha256']
        manifest=read(sealroot/'SEAL_MANIFEST.json')['files'];cfg=read(study/'models.json')[model];base=Path(cfg['path'])
        index=read(base/'model.safetensors.index.json')['weight_map'];ep=base/index['model.embed_tokens.weight']
        assert sha(ep)==cfg['files'][str(ep)]
        tf=study/f'tokens/wikisql_{model}_train.json';assert sha(tf)==scales['models'][model]['token_file_sha256']
        counts=collections.Counter(t for r in read(tf) for t in r['prompt_ids']+r['target_ids'] if t not in special)
        idx=torch.tensor(sorted(counts));freq=torch.tensor([counts[int(t)] for t in idx],dtype=torch.float64);freq/=freq.sum()
        with safe_open(str(ep),framework='pt',device='cpu') as f:x=f.get_tensor('model.embed_tokens.weight').index_select(0,idx).float()
        def weighted_rms(v):return float((v.square().mean(dim=1).double()*freq).sum().sqrt())
        base_rms=weighted_rms(x);rs=[]
        for seed in [7100,7101,7102]:
            p=study/'checkpoints'/f'wikisql_{model}_hidden_both_s{seed}'/'adapter.safetensors';h=sha(p);assert h==manifest[str(p.relative_to(sealroot))]['sha256']
            with safe_open(str(p),framework='pt',device='cpu') as f:s={k:f.get_tensor(k) for k in f.keys() if k.startswith('boundary_')}
            bias=s['boundary_input.bias'];a=s['boundary_input.down.weight'];b=s['boundary_input.up.weight'];low=8*((x@a.T)@b.T)
            delta=low+bias;bias_rms=float(bias.square().mean().sqrt())
            ua=s['boundary_output.down.weight'];ub=s['boundary_output.up.weight'];ufrob=8*float(((ub.T@ub)*(ua@ua.T)).sum().sqrt())/math.sqrt(ua.shape[1])
            rs.append({'seed':seed,'adapter_sha256':h,'embedding_rms':base_rms,'input_bias_rms':bias_rms,'input_bias_rms_over_embedding_rms':bias_rms/base_rms,
                       'input_low_rank_delta_rms_over_embedding_rms':weighted_rms(low)/base_rms,
                       'input_total_delta_rms_over_embedding_rms':weighted_rms(delta)/base_rms,
                       'output_linear_delta_frobenius_over_identity_frobenius':ufrob})
        out['models'][model]=rs;print(model,json.dumps(rs),flush=True)
    (ROOT/'llama_lr_20260918/BOUNDARY_DISPLACEMENT_POSTHOC.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
