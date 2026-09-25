"""Independent effective-vocabulary oracle on a synthetic CPU Gemma, no pretrained outputs."""
import copy,inspect,torch
from transformers import Gemma2Config,Gemma2ForCausalLM
from transformers.models.gemma2 import modeling_gemma2,configuration_gemma2
from common import *
from modeling import install_boundaries,loss,digest
def main():
 assert not (HERE/'CODE_FROZEN.json').exists()
 torch.set_num_threads(2);torch.manual_seed(314159)
 cfg=Gemma2Config(vocab_size=47,hidden_size=24,intermediate_size=48,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
       max_position_embeddings=64,sliding_window=16,query_pre_attn_scalar=8,attn_logit_softcapping=50.,final_logit_softcapping=30.,bos_token_id=2,eos_token_id=1,pad_token_id=0,tie_word_embeddings=True)
 cfg._attn_implementation='eager';model=Gemma2ForCausalLM(cfg).eval().requires_grad_(False)
 with torch.no_grad():model.get_input_embeddings().weight.normal_(std=2.)
 reference=copy.deepcopy(model);before=digest(dict(model.named_parameters()));install_boundaries(model,'hidden_both',314159)
 ids=torch.tensor([[2,4,8,11],[2,7,3,0]]);mask=(ids!=0).long();labels=ids.clone();labels[:,0]=-100;labels[ids==0]=-100
 with torch.inference_mode():
  zero=model(input_ids=ids,attention_mask=mask,use_cache=False).logits;native=reference(input_ids=ids,attention_mask=mask,use_cache=False).logits
  assert torch.equal(zero,native)
 with torch.no_grad():
  for side in ['input','output']:
   aff=getattr(model,'boundary_'+side);aff.up.weight.normal_(std=.03)
   if aff.bias is not None:aff.bias.normal_(std=.01)
  e=model.boundary_input;u=model.boundary_output;d=cfg.hidden_size
  te=torch.eye(d)+e.scale*(e.up.weight@e.down.weight);tu=torch.eye(d)+u.scale*(u.up.weight@u.down.weight)
  ee=model.get_input_embeddings().weight@te.T+e.bias_scale*e.bias
  uu=model.lm_head.weight@tu
  reference.config.tie_word_embeddings=False
  reference.get_input_embeddings().weight=torch.nn.Parameter(ee.clone(),requires_grad=False)
  reference.lm_head.weight=torch.nn.Parameter(uu.clone(),requires_grad=False)
 with torch.inference_mode():
  got=model(input_ids=ids,attention_mask=mask,use_cache=False).logits;expected=reference(input_ids=ids,attention_mask=mask,use_cache=False).logits
  embedding_error=float((model.get_input_embeddings()(ids)-reference.get_input_embeddings()(ids)).abs().max())
  error=float((got-expected).abs().max());assert torch.allclose(got,expected,atol=2e-4,rtol=2e-5),(error,embedding_error)
  native_loss=model(input_ids=ids,attention_mask=mask,labels=labels,use_cache=False).loss;selected_loss=loss(model,ids,mask,labels)[0]
  loss_error=float(abs(native_loss-selected_loss));assert loss_error<2e-5
 assert model.get_input_embeddings().weight.data_ptr()==model.lm_head.weight.data_ptr()
 assert digest({n:p for n,p in model.named_parameters() if not n.startswith('boundary_')})==before
 assert all(model._boundary_counts[s]>0 for s in ['input','output'])
 result={'at':now(),'status':'passed','scope':'Synthetic tiny CPU Gemma; no pretrained benchmark response or model selection.',
   'zero_adapter_logits_exact':True,'nonzero_effective_vocabulary_logits_max_abs_error':error,'nonzero_embedding_max_abs_error':embedding_error,
   'native_selected_target_loss_error':loss_error,'base_parameters_unchanged':True,'physical_base_tying_preserved':True,
   'input_oracle':'E_eff = E @ (I + scale * up @ down).T + bias; then native embedding scaling',
   'output_oracle':'U_eff = U @ (I + scale * up @ down); then native final logit softcap',
   'reference_tying':'effective E/U are independent; no claim they remain tied after folding',
   'native_sources':{inspect.getfile(m):sha(inspect.getfile(m)) for m in [modeling_gemma2,configuration_gemma2]}}
 write(HERE/'MODEL_FORMULA_AUDIT.json',result);print(canonical(result),flush=True)
if __name__=='__main__':main()
