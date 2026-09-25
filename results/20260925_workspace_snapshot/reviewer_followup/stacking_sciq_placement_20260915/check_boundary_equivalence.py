"""Check custom placement hooks against the verified original A-LoRA wrappers."""
import argparse,json
import torch
from common import *
from modeling import load_checkpoint,batch,forward_selected
from affine_adapter import AffineEmbedding,AffineLMHead,AffineVocabConfig

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);a=p.parse_args()
    cp=HERE/'checkpoints'/f'smoke_{a.model}_both_sd2001_lr5e-05'
    report={}
    for precision in ['native_bf16_amp','fp32']:
        model,spec,_=load_checkpoint(cp,fp32=precision=='fp32')
        meta=read(HERE/'DATA_AUDIT.json')['tokenization'][a.model]
        items=read(HERE/'tokens'/f'{a.model}_validation.json')[:3];ids,mask,lengths=batch(items,meta)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='native_bf16_amp'):
            custom=forward_selected(model,ids,mask,lengths)
            for handle in model._placement_handles:handle.remove()
            base=model.get_base_model();cfg=AffineVocabConfig(base.config.hidden_size,rank=16,alpha=128,use_lm_head=True)
            base.set_input_embeddings(AffineEmbedding(base.get_input_embeddings(),cfg,base.placement_maps['input']))
            base.lm_head=AffineLMHead(base.lm_head,cfg,base.placement_maps['output'])
            original=forward_selected(model,ids,mask,lengths)
        error=float((custom-original).abs().max());assert error==0,error
        report[precision]={'max_abs_logit_error':error,'bitwise_equal':bool(torch.equal(custom,original))}
        del model,custom,original;torch.cuda.empty_cache()
    write(HERE/f'BOUNDARY_EQUIVALENCE_{a.model}.json',{'status':'passed','trained_nonzero_smoke_maps':True,'paths':report})
    print(json.dumps(report))

if __name__=='__main__':main()
