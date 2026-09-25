"""Predeclared coverage and exact integer budget for the hybrid text decoder."""
TARGETS=['q_proj','k_proj','v_proj','o_proj','up_proj','down_proj','gate_proj',
         'in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','out_proj']
def architecture_plan(cfg):
 d=cfg.hidden_size; f=cfg.intermediate_size;q=cfg.num_attention_heads*cfg.head_dim;k=cfg.num_key_value_heads*cfg.head_dim
 lk=cfg.linear_num_key_heads*cfg.linear_key_head_dim;lv=cfg.linear_num_value_heads*cfg.linear_value_head_dim
 modules={};groups={'query':[],'key':[],'output':[]}
 for i,kind in enumerate(cfg.layer_types):
  stem=f'model.layers.{i}'
  for n,ins,outs in [('gate_proj',d,f),('up_proj',d,f),('down_proj',f,d)]:modules[f'{stem}.mlp.{n}']=[ins,outs]
  if kind=='full_attention':
   block='self_attn';sizes={'q_proj':(d,2*q),'k_proj':(d,k),'v_proj':(d,k),'o_proj':(q,d)}
   grouped={'query':'q_proj','key':'k_proj','output':'o_proj'}
  else:
   assert kind=='linear_attention';block='linear_attn'
   sizes={'in_proj_qkv':(d,2*lk+lv),'in_proj_z':(d,lv),'in_proj_b':(d,cfg.linear_num_value_heads),'in_proj_a':(d,cfg.linear_num_value_heads),'out_proj':(lv,d)}
   grouped={'query':'in_proj_qkv','output':'out_proj'}
  for n,shape in sizes.items():modules[f'{stem}.{block}.{n}']=list(shape)
  for group,n in grouped.items():groups[group].append(f'{stem}.{block}.{n}')
 costs={g:{sum(modules[n]) for n in names} for g,names in groups.items()};assert all(len(s)==1 for s in costs.values());costs={g:s.pop() for g,s in costs.items()}
 extra=65*d
 choices=[(a*costs['query']+b*costs['key']+c*costs['output']-extra,-a,-b,c,a,b) for a in range(len(groups['query'])+1) for b in range(len(groups['key'])+1) for c in range(len(groups['output'])+1) if a*costs['query']+b*costs['key']+c*costs['output']>=extra]
 excess,_,_,no,nq,nk=min(choices);assert excess==0,(extra,costs)
 counts={'query':nq,'key':nk,'output':no};chosen=[]
 for group,count in counts.items():
  names=groups[group];chosen += [names[i*len(names)//count] for i in range(count)]
 assert len(set(chosen))==len(chosen) and sum(sum(modules[n]) for n in chosen)==extra
 return {'hidden_size':d,'hidden_rank':8,'boundary_rank':16,'input_bias':True,'output_bias':False,'extra':extra,
         'hidden_parameters':8*sum(sum(v) for v in modules.values()),'strict_match':True,'modules':modules,
         'rank_pattern':{n:9 for n in chosen},'group_costs':costs,'extra_rank_counts':counts,
         'rule':'Minimize nonnegative excess, maximize query-like then key extra ranks; at most one per module, evenly spaced within group in layer order.',
         'scope':'All 12 named dense linear projections in the text decoder only; no vision, lm_head, convolution, norm or recurrent scalar adaptation.'}
