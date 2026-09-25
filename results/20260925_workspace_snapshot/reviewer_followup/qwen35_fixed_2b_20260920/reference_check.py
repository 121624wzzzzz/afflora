"""Enforce independent-process references before any optimizer update."""
from common import HERE,read,sha
def check_reference(repeat,task,seed):
 path=HERE/'INITIAL_REFERENCES.json';refs=read(path);assert refs['status']=='passed'
 ref=refs['records'][f'{task}:{seed}']['repeat']
 assert repeat['status']=='passed' and repeat['gradients_bitwise_equal']
 assert repeat['loss_absolute_difference']==0 and repeat['gradient_relative_rms']==0
 assert repeat['order_sha256']==ref['order_sha256']
 for values in repeat['microbatch_losses']:assert values==ref['microbatch_losses'][0],(task,seed,values,ref['microbatch_losses'][0])
 return dict(status='passed',reference_sha256=sha(path),reference_key=f'{task}:{seed}',exact_microbatch_match=True,
  scope='Independent-process first-batch loss parity only; not proof of a whole-training-trajectory invariant.')
