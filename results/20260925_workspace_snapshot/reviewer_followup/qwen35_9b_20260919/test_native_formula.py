"""Synthetic CPU hybrid decoder: boundary equivalence, native loss and text/VLM parity."""
import copy,inspect,torch
from transformers import Qwen3_5TextConfig,Qwen3_5ForCausalLM,Qwen3_5Config,Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5 import modeling_qwen3_5,configuration_qwen3_5
from transformers import conversion_mapping
from common import *
from modeling import install_boundaries,loss,digest

def main():
 assert not (HERE/'CODE_FROZEN.json').exists()
 for name in ['causal_conv1d_fn','causal_conv1d_update','chunk_gated_delta_rule','fused_recurrent_gated_delta_rule','FusedRMSNormGated']:setattr(modeling_qwen3_5,name,None)
 torch.set_num_threads(2);torch.manual_seed(314159)
 cfg=Qwen3_5TextConfig(vocab_size=47,hidden_size=32,intermediate_size=64,num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
    linear_key_head_dim=8,linear_value_head_dim=8,linear_num_key_heads=2,linear_num_value_heads=4,linear_conv_kernel_dim=4,
    layer_types=['linear_attention']*3+['full_attention'],max_position_embeddings=128,bos_token_id=None,eos_token_id=1,pad_token_id=0,tie_word_embeddings=False,
    rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':1.,'mrope_section':[1,1,2],'mrope_interleaved':True})
 cfg._attn_implementation='eager';model=Qwen3_5ForCausalLM(cfg).eval().requires_grad_(False)
 reference=copy.deepcopy(model);before=digest(dict(model.named_parameters()));install_boundaries(model,'hidden_both',314159)
 ids=torch.tensor([[4,8,11,12],[7,3,9,0]]);mask=(ids!=0).long();labels=ids.clone();labels[:,0]=-100;labels[ids==0]=-100
 with torch.inference_mode():
  zero=model(input_ids=ids,attention_mask=mask,use_cache=False).logits;native=reference(input_ids=ids,attention_mask=mask,use_cache=False).logits
  assert torch.equal(zero,native)
 # Native multimodal wrapper, text only: share the unmodified text modules exactly.
 vc={'depth':1,'hidden_size':32,'intermediate_size':64,'num_heads':4,'out_hidden_size':32,'patch_size':2,'spatial_merge_size':2,'temporal_patch_size':2,'num_position_embeddings':16}
 full_cfg=Qwen3_5Config(text_config=cfg.to_dict(),vision_config=vc,image_token_id=43,video_token_id=44,vision_start_token_id=45,vision_end_token_id=46,tie_word_embeddings=False)
 full_cfg._attn_implementation='eager';full=Qwen3_5ForConditionalGeneration(full_cfg).eval();full.model.language_model=reference.model;full.lm_head=reference.lm_head
 with torch.inference_mode():
  full_out=full(input_ids=ids,attention_mask=mask,use_cache=False).logits
  vlm_error=float((full_out[mask.bool()]-native[mask.bool()]).abs().max());assert vlm_error<2e-5,vlm_error
 with torch.no_grad():
  for side in ['input','output']:
   aff=getattr(model,'boundary_'+side);aff.up.weight.normal_(std=.03)
   if aff.bias is not None:aff.bias.normal_(std=.01)
  e=model.boundary_input;u=model.boundary_output;d=cfg.hidden_size
  te=torch.eye(d)+e.scale*(e.up.weight@e.down.weight);tu=torch.eye(d)+u.scale*(u.up.weight@u.down.weight)
  ee=model.get_input_embeddings().weight@te.T+e.bias_scale*e.bias;uu=model.lm_head.weight@tu
  reference.config.tie_word_embeddings=False
  reference.get_input_embeddings().weight=torch.nn.Parameter(ee.clone(),requires_grad=False);reference.lm_head.weight=torch.nn.Parameter(uu.clone(),requires_grad=False)
 with torch.inference_mode():
  got=model(input_ids=ids,attention_mask=mask,use_cache=False).logits;expected=reference(input_ids=ids,attention_mask=mask,use_cache=False).logits
  error=float((got-expected).abs().max());assert torch.allclose(got,expected,atol=2e-4,rtol=2e-5),error
  native_loss=model(input_ids=ids,attention_mask=mask,labels=labels,use_cache=False).loss;selected_loss=loss(model,ids,mask,labels)[0]
  loss_error=float(abs(native_loss-selected_loss));assert loss_error<2e-5
 model.train();model.zero_grad();loss(model,ids,mask,labels)[0].backward()
 assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith('boundary_'))
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for n,p in model.named_parameters() if n.startswith('boundary_'))
 assert model.get_input_embeddings().weight.data_ptr()!=model.lm_head.weight.data_ptr()
 assert digest({n:p for n,p in model.named_parameters() if not n.startswith('boundary_')})==before
 assert all(model._boundary_counts[s]>0 for s in ['input','output'])
 result={'at':now(),'status':'passed','scope':'Synthetic tiny CPU Qwen3.5 hybrid decoder; no pretrained benchmark outputs.',
  'zero_adapter_logits_exact':True,'nonzero_effective_vocabulary_logits_max_abs_error':error,'native_selected_target_loss_error':loss_error,
  'native_text_vs_vlm_valid_logit_error':vlm_error,'finite_boundary_gradients':True,'base_parameters_unchanged':True,'physical_base_tying_preserved':True,'native_tied':False,
  'input_oracle':'E_eff = E @ (I + scale * up @ down).T + bias', 'output_oracle':'U_eff = U @ (I + scale * up @ down)',
  'native_sources':{inspect.getfile(m):sha(inspect.getfile(m)) for m in [modeling_qwen3_5,configuration_qwen3_5,conversion_mapping]}}
 write(HERE/'MODEL_FORMULA_AUDIT.json',result);print(canonical(result),flush=True)
if __name__=='__main__':main()
