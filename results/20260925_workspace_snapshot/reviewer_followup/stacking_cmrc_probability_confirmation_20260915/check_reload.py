"""Compare independently loaded adapters against saved in-memory smoke logits."""
import argparse
from pathlib import Path
import sys
import hashlib
import torch
import common as support
sys.path.insert(0,str(support.HERE/'source/corrected_sft_experiment'))
import evaluate_corrected_sft as ce


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);a=p.parse_args()
    torch.set_num_threads(4);cp=Path(a.checkpoint)
    saved=torch.load(cp/'smoke_forward.pt',weights_only=True,map_location='cpu')
    model,_,_,_=ce.load_model(argparse.Namespace(run_dir=str(cp),model_path=None,affine_ablation='none',device='cuda'))
    with torch.inference_mode():
        got=model(input_ids=saved['input_ids'].to('cuda'),attention_mask=saved['attention_mask'].to('cuda')).logits[:,-3:,:].float().cpu()
    error=float((got-saved['last_logits']).abs().max())
    assert error<=1e-5,error
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        amp=model(input_ids=saved['input_ids'].to('cuda'),attention_mask=saved['attention_mask'].to('cuda')).logits[:,-3:,:].float().cpu()
    amp_error=float((amp-saved['amp_logits']).abs().max());assert amp_error<=1e-5,amp_error
    digest=hashlib.sha256()
    for name,value in sorted(model.named_parameters()):
        if 'lora_' in name or '.affine.' in name:
            digest.update(name.encode());digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
    assert digest.hexdigest()==saved['adapter_tensor_sha256']
    report={'status':'passed','max_abs_logit_difference':error,'amp_max_abs_logit_difference':amp_error,
            'adapter_tensor_sha256':digest.hexdigest(),'adapter_tensors_bitwise_identical':True}
    support.write_json(cp/'RELOAD_AUDIT.json',report)
    print(report)


if __name__=='__main__':main()
