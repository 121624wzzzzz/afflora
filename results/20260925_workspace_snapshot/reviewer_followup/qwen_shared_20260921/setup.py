from pathlib import Path
import shutil,json,hashlib
R=Path(__file__).resolve().parent; O=R.parent/'model_architecture_14h_20260917/core'; S=R/'study'
assert not S.exists();S.mkdir()
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
seal=read(O.parent/'SEAL_MANIFEST.json')['files'];verified={}
for p in list(O.glob('*.py'))+list((O/'source').glob('*.py')):
 rel=str(p.relative_to(O));assert sha(p)==seal['core/'+rel]['sha256'];dest=S/rel;dest.parent.mkdir(exist_ok=True);shutil.copy2(p,dest);verified[rel]=sha(p)
for name in ['raw','data','tokens']:(S/name).symlink_to(O/name,target_is_directory=True)
for name in ['models.json','BUDGET_PLAN.json']:
 assert sha(O/name)==seal['core/'+name]['sha256'];shutil.copy2(O/name,S/name)
for name in ['checkpoints','evaluations','audits','logs','specs']:(S/name).mkdir()
p=S/'common.py';p.write_text(p.read_text().replace("ARMS = [", "ARMS = ['hidden_shared16','hidden_shared32',"))
p=S/'modeling.py';s=p.read_text().replace('from affine_adapter import LowRankAffineMap','from affine_adapter import LowRankAffineMap\nfrom shared_boundary import SharedTransposeHead')
s=s.replace("n.replace('.base_layer.','.')", "n.replace('.base_layer.','.').replace('lm_head.base_head.','lm_head.')")
pos="    for n,p in model.named_parameters():\n        assert p.requires_grad==is_adapter(n),(arm,n)"
insert="""    shared=arm in ['hidden_shared16','hidden_shared32']
    if shared:
        rank=int(arm.removeprefix('hidden_shared'))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(spec['seed']+1000003)
            aff=LowRankAffineMap(model.config.hidden_size,rank,8*rank,0,True)
        model.add_module('boundary_input',aff.float())
        def post(module,args,output):counts['input']+=1;return aff(output)
        hooks.append(model.get_input_embeddings().register_forward_hook(post))
        model.lm_head=SharedTransposeHead(model.lm_head,aff,counts)
"""
assert pos in s;s=s.replace(pos,insert+pos)
s=s.replace("    assert audit['parameter_groups']['boundary']==expect", "    if shared:expect=(2*rank+1)*d\n    audit['boundary_sharing']={'shared':shared,'rank':rank if shared else None,'output_transpose':shared,'unique_parameter_ids':len({id(p) for p in trainable.values()}),'active_sides':['input','output'] if shared else list(audit['boundary_initialization_sha256'])}\n    assert audit['boundary_sharing']['unique_parameter_ids']==len(trainable)\n    assert audit['parameter_groups']['boundary']==expect")
p.write_text(s)
p=S/'audit_one.py';s=p.read_text();s=s.replace("(arm in allowed)","(arm in allowed or (side=='input' and arm in ['hidden_shared16','hidden_shared32']))")
s=s.replace("    assert init['preflight']", "    if arm in ['hidden_shared16','hidden_shared32']:\n        assert init['boundary_sharing']['shared'] and init['boundary_sharing']['output_transpose']\n        assert init['boundary_sharing']['active_sides']==['input','output']\n    assert init['preflight']",1);p.write_text(s)
# Shared output parameters are stored once under boundary_input. Runtime calls separately audited.
p=S/'run.py';s=p.read_text().replace("    after=frozen_digest(model)", "    if spec['arm'] in ['hidden_shared16','hidden_shared32']:\n        assert model._boundary_counts['input']>0 and model._boundary_counts['output']>0\n        write(root/'SHARED_RUNTIME.json',{'calls':model._boundary_counts,'stored_once':True,'output_transpose':True})\n    after=frozen_digest(model)");p.write_text(s)
models=['qwen25_15b_base','qwen3_06b_base','qwen25_7b_base','qwen3_8b_base'];bud=read(S/'BUDGET_PLAN.json');cfg=read(S/'models.json')
for m in models:
 for r in [16,32]:bud[m]['hidden_shared'+str(r)]=bud[m]['hidden']+(2*r+1)*cfg[m]['hidden_size']
 assert bud[m]['hidden_shared32']==bud[m]['hidden_both']
write(S/'BUDGET_PLAN.json',bud)
jobs=[];smokes=[]
for seedidx in range(3):
 for m in models:
  for task,start,micro in [('cluener',6100,2),('wikisql',7100,4)]:
   for r in [16,32]:
    arm='hidden_shared'+str(r);n=f'{task}_{m}_{arm}_s{start+seedidx}';spec=dict(name=n,model=m,task=task,arm=arm,seed=start+seedidx,lr=2e-4,microbatch=micro,smoke=False);jobs.append(spec)
    if seedidx==0:
     smoke=dict(spec,name=n+'_smoke',smoke=True);smokes.append(smoke)
write(S/'FORMAL_JOBS.json',jobs);write(S/'SMOKE_JOBS.json',smokes);write(S/'SOURCE_REUSE.json',dict(source=str(O),manifest_sha256=sha(O.parent/'SEAL_MANIFEST.json'),code_hashes=verified))
print(S)
