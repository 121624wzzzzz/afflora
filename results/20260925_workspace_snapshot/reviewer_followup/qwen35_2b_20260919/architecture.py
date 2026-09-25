"""Architecture-derived exact budget, fixed without observing task outcomes."""
import itertools
TARGETS=['q_proj','k_proj','v_proj','o_proj','up_proj','down_proj','gate_proj',
         'in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','out_proj']
def architecture_plan(cfg):
 d=cfg.hidden_size; f=cfg.intermediate_size;q=cfg.num_attention_heads*cfg.head_dim;k=cfg.num_key_value_heads*cfg.head_dim
 lk=cfg.linear_num_key_heads*cfg.linear_key_head_dim;lv=cfg.linear_num_value_heads*cfg.linear_value_head_dim
 modules={};groups={}
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
  for group,n in grouped.items():
   name=f'{stem}.{block}.{n}';groups.setdefault((group,sum(modules[name])),[]).append(name)
 keys=sorted(groups,key=lambda x:(['query','key','output'].index(x[0]),x[1]));extra=65*d;choices=[]
 for counts in itertools.product(*(range(len(groups[k])+1) for k in keys)):
  if sum(n*k[1] for n,k in zip(counts,keys))!=extra:continue
  nq=sum(n for n,k in zip(counts,keys) if k[0]=='query');nk=sum(n for n,k in zip(counts,keys) if k[0]=='key');no=sum(n for n,k in zip(counts,keys) if k[0]=='output')
  choices.append(((-nq,-nk,no,tuple(-n for n in counts)),counts))
 assert choices,('No exact active-rank budget',extra,keys)
 counts=min(choices)[1];chosen=[]
 for k,count in zip(keys,counts):
  names=groups[k];chosen += [names[i*len(names)//count] for i in range(count)]
 assert len(set(chosen))==len(chosen) and sum(sum(modules[n]) for n in chosen)==extra
 return {'hidden_size':d,'hidden_rank':8,'boundary_rank':16,'input_bias':True,'output_bias':False,'extra':extra,
  'hidden_parameters':8*sum(sum(v) for v in modules.values()),'strict_match':True,'modules':modules,
  'rank_pattern':{n:9 for n in chosen},'group_costs':{f'{g}:{c}':c for g,c in keys},
  'extra_rank_counts':{f'{g}:{c}':n for (g,c),n in zip(keys,counts)},
  'rule':'Exact budget, at most one added rank/module. Maximize query count, then key count, minimize output count; ties maximize counts in query/key/output then ascending-cost order. Even spacing by layer within each semantic/cost group.',
  'scope':'All 12 named dense projections in text decoder; no vision, lm_head, convolution, norm or recurrent scalar adaptation.'}
