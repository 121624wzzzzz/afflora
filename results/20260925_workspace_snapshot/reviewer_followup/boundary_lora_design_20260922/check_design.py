from pathlib import Path
import json,torch
R=Path(__file__).resolve().parent;O=R.parent/'qwen_shared_rank8_20260921/study';models=json.loads((O/'models.json').read_text());records=[]
for m in ['qwen3_06b_base','qwen25_15b_base','qwen25_7b_base','qwen3_8b_base']:
 c=json.loads((Path(models[m]['path'])/'config.json').read_text());d=c['hidden_size'];v=c['vocab_size']
 records.append(dict(model=m,d=d,V=v,tied=c['tie_word_embeddings'],standard_E_or_U_rank1=v+d,standard_independent_EU_rank1=2*(v+d),standard_shared_rank1=v+d,affine_E_rank16=33*d,affine_U_rank16=32*d,affine_independent_EU_rank16=65*d,affine_shared_rank16=33*d,affine_shared_rank32=65*d))
# Algebra check using row-vector conventions. E:Vxd; U:dxV.
g=torch.Generator().manual_seed(922);rand=lambda *shape:torch.randn(*shape,generator=g,dtype=torch.float64)
v,d,r=29,11,3;E=rand(v,d);U=rand(d,v);A=rand(d,r);B=rand(r,d);b=rand(d);h=rand(5,d);s=8.
Eeff=E+s*(E@A)@B+b;expected_E=E@(torch.eye(d,dtype=torch.float64)+s*A@B)+b
Ueff=U+s*A@(B@U);expected_logits=(h+s*(h@A)@B)@U
errors=dict(E_factorization=float((Eeff-expected_E).abs().max()),U_factorization=float((h@Ueff-expected_logits).abs().max()))
assert max(errors.values())<1e-10
# The inherited-vocabulary factor is constrained to the base matrix's subspace.
P=rand(v,r);projected=E@torch.linalg.lstsq(E,P).solution
residual=float(torch.linalg.vector_norm(P-projected));assert residual>1e-3
out=dict(status='passed',scope='Design-only parameter accounting and CPU algebra; no new training launched',budgets=records,algebra_errors=errors,free_LoRA_factor_outside_embedding_span_residual=residual,interpretation='Without E bias, affine E update=(E A)B and U update=A(B U), rank<=r. These are constrained effective low-rank weight updates, not identical optimization parameterizations to unrestricted vocabulary-matrix LoRA. Input bias adds a possible rank-one term.')
(R/'DESIGN_CHECK.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False,indent=2))
