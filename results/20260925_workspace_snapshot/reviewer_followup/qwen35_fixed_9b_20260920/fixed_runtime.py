"""Explicit process-local numerical policy; does not edit installed libraries."""
import os,importlib,sys
import torch
from triton.runtime.autotuner import Autotuner

# Architecture/data/score-independent choices, all drawn from the installed
# native FLA kernel's own admissible configuration list. No timing autotuning.
POLICY={
 'fla.ops.utils.cumsum:chunk_local_cumsum_scalar_kernel':({},4,2),
 'fla.ops.utils.solve_tril:merge_16x16_to_64x64_inverse_kernel':({'DOT_PRECISION':'ieee'},4,3),
 'fla.modules.l2norm:l2norm_fwd_kernel':({'BT':32},4,2),
 'fla.modules.l2norm:l2norm_bwd_kernel':({'BT':32},4,2),
 'fla.ops.common.chunk_delta_h:chunk_gated_delta_rule_fwd_kernel_h_blockdim64':({'BV':32},4,2),
 'fla.ops.common.chunk_delta_h:chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64':({'BV':32},4,2),
 'fla.ops.common.chunk_o:chunk_fwd_kernel_o':({'BK':64,'BV':64},4,3),
 'fla.ops.common.chunk_o:chunk_bwd_kernel_dqkwg':({},4,3),
 'fla.ops.common.chunk_o:chunk_bwd_kernel_dv_local':({},4,3),
 'fla.ops.common.chunk_scaled_dot_kkt:chunk_scaled_dot_kkt_fwd_kernel':({'BK':64},4,3),
 'fla.ops.gated_delta_rule.wy_fast:recompute_w_u_fwd_kernel':({},4,3),
 'fla.ops.gated_delta_rule.wy_fast:prepare_wy_repr_bwd_kernel':({},4,3),
}
_installed={}
def unwrap(obj):
 while not isinstance(obj,Autotuner):obj=obj.fn
 return obj
def describe(c):return {'kwargs':c.kwargs,'num_warps':c.num_warps,'num_stages':c.num_stages,'num_ctas':c.num_ctas}
def install():
 assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','Set before process startup'
 torch.use_deterministic_algorithms(True)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
 torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
 for name,(kwargs,warps,stages) in POLICY.items():
  module,attr=name.split(':');obj=unwrap(getattr(importlib.import_module(module),attr))
  candidates=[c for c in obj.configs if c.kwargs==kwargs and c.num_warps==warps and c.num_stages==stages and c.num_ctas==1]
  assert len(candidates)==1,(name,[describe(c) for c in obj.configs])
  if name not in _installed:assert not obj.cache,(name,'Already autotuned before fixed policy was installed')
  obj.configs=candidates;obj.cache.clear();_installed[name]=obj
 return audit()
def audit():
 assert torch.are_deterministic_algorithms_enabled()
 assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
 assert not torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
 assert not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
 assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
 known=set(map(id,_installed.values()));executed=[];seen=set()
 for module_name,module in list(sys.modules.items()):
  if not module_name.startswith('fla.') or module is None:continue
  for attr,obj in vars(module).items():
   chain=set()
   while hasattr(obj,'fn') and id(obj) not in chain:
    chain.add(id(obj))
    if isinstance(obj,Autotuner):
     if id(obj) not in seen and hasattr(obj,'best_config'):
      seen.add(id(obj));assert id(obj) in known,('Unpinned active kernel',module_name,attr)
      assert len(obj.configs)==1 and not obj.cache
      assert describe(obj.best_config)==describe(obj.configs[0])
      executed.append(next(n for n,o in _installed.items() if o is obj))
     break
    obj=obj.fn
 return {'policy':{name:describe(obj.configs[0]) for name,obj in _installed.items()},'executed_kernels':sorted(executed),
  'torch_deterministic_algorithms':True,'cublas_workspace_config':os.environ['CUBLAS_WORKSPACE_CONFIG'],
  'cudnn_deterministic':True,'tf32':False,'reduced_precision_gemm_reduction':False,
  'scope':'Pinned official FLA configs and explicit PyTorch math settings; does not imply a universal cross-device determinism guarantee.'}
