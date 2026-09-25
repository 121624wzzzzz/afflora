"""Finish numerical admission and native tying checks before any freeze or fit."""
import ast,json,hashlib
from pathlib import Path
PARENT=Path(__file__).resolve().parent
def edit(p,a,b):
 s=p.read_text();assert a in s,(p,a);p.write_text(s.replace(a,b))
def main():
 for key in ['08b','2b','4b','9b']:
  root=PARENT/f'qwen35_fixed_{key}_20260920';assert not (root/'CODE_FROZEN.json').exists()
  p=root/'test_native_formula.py'
  edit(p,"torch.set_num_threads(2);torch.manual_seed(314159)","torch.set_num_threads(2);torch.manual_seed(314159)\n tied=read(HERE/'models.json')[MODEL]['tie_word_embeddings']")
  edit(p,'tie_word_embeddings=True','tie_word_embeddings=tied')
  edit(p,'assert model.get_input_embeddings().weight.data_ptr()==model.lm_head.weight.data_ptr()',
   'assert (model.get_input_embeddings().weight.data_ptr()==model.lm_head.weight.data_ptr())==tied')
  edit(p,"'physical_base_tying_preserved':True,","'physical_base_tying_preserved':True,'native_tie_word_embeddings':tied,")
  p=root/'initial_repeat.py'
  edit(p,'from modeling import batch,loss,is_adapter','from modeling import batch,loss,is_adapter,digest')
  edit(p,"'gradients_bitwise_equal':exact,","'gradients_bitwise_equal':exact,'gradient_sha256':[digest(g) for g in snapshots],")
  edit(p,"result['status']='passed' if diff<2e-4 and rms<1e-4 else 'failed'", "result['status']='passed' if diff==0 and rms==0 and exact and contributions[0]==contributions[1] else 'failed'")
  edit(p,"'loss_tolerance':2e-4,'gradient_relative_rms_tolerance':1e-4,", "'loss_tolerance':0,'gradient_relative_rms_tolerance':0,")
  p=root/'test_initial_repeat.py'
  edit(p,"records[-1]['cross_process_initial_reference']=check_reference(r,task,7600)",
   "records[-1]['cross_process_initial_reference']=check_reference(r,task,7600)\n   if arm=='hidden':assert r['gradient_sha256']==read(HERE/'INITIAL_REFERENCES.json')['records'][f'{task}:7600']['repeat']['gradient_sha256']")
  p=root/'audit_one.py'
  edit(p,"'summary_sha256':summaries", "'numerical_policy_final_sha256':sha(root/'NUMERICAL_POLICY_FINAL.json'),'summary_sha256':summaries")
  p=root/'final_audit.py'
  edit(p,"assert a['status']=='passed';responses+=a['responses'];", "assert a['status']=='passed'\n  assert a['numerical_policy_final_sha256']==sha(root/'NUMERICAL_POLICY_FINAL.json')\n  if s['arm']!='base':\n   from reference_check import check_reference\n   initial=read(root/'INITIALIZATION.json');repeat=read(root/'INITIAL_REPEAT.json')\n   assert check_reference(repeat,s['task'],s['seed'])==initial['cross_process_initial_reference']\n   assert read(root/'TRAINING.json')['history'][0]['loss']==repeat['losses'][0]\n  responses+=a['responses'];")
  for p in root.glob('*.py'):ast.parse(p.read_text())
  dispatcher=PARENT/'schedule_qwen35_fixed_20260920.py'
  (root/'DISPATCHER_IDENTITY.json').write_text(json.dumps(dict(path=str(dispatcher),sha256=hashlib.sha256(dispatcher.read_bytes()).hexdigest()),indent=2)+'\n')
 print('All four sources finalized before technical gates.',flush=True)
if __name__=='__main__':main()
