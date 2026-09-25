from common import *
import argparse,numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);name=ap.parse_args().name;root=HERE/'results'/name;complete=read(root/'COMPLETE.json');assert complete['status']=='passed' and sha(root/'SUMMARY.json')==complete['summary_sha256'];job=read(root/'JOB.json');summaries=read(root/'SUMMARY.json')
 for task in ['sciq','anli_r1']:
  s=summaries[task];assert sha(root/f'{task}.jsonl')==s['responses_sha256'];data=read(HERE/f"tokens/{job['model']}_{task}.json")['rows'];rs=rows(root/f'{task}.jsonl');assert [r['id'] for r in rs]==[r['id'] for r in data]
  correct=[];rot=[];mass=[];goldp=[]
  for r,g in zip(rs,data):
   v=r['variants'];n=len(v);assert r['gold']==g['gold']
   probs=np.asarray([x['candidate_logprobs'] for x in v],dtype=np.float64);assert np.isfinite(probs).all();mass.append(np.exp(probs).sum(axis=1).mean());probs=np.exp(probs-probs.max(axis=1,keepdims=True));probs/=probs.sum(axis=1,keepdims=True);semantic=np.zeros(n);rc=[]
   for k,x in enumerate(v):
    assert x['rotation']==k and x['order']==g['variants'][k]['order']
    semantic[np.asarray(x['order'])]+=probs[k]/n;rc.append(x['order'][int(probs[k].argmax())]==g['gold'])
   pred=int(semantic.argmax());assert pred==r['prediction'] and np.allclose(semantic,r['semantic_probabilities'],atol=1e-12,rtol=0)
   assert r['correct']==(pred==g['gold']) and rc==r['rotation_correct'];assert abs(mass[-1]-r['candidate_mass'])<1e-10
   correct.append(pred==g['gold']);rot.append(rc);goldp.append(semantic[g['gold']])
  assert abs(s['accuracy']-100*np.mean(correct))<1e-9 and np.allclose(s['rotation_accuracy'],100*np.mean(rot,axis=0),atol=1e-9)
  assert abs(s['candidate_mass']-np.mean(mass))<1e-10 and abs(s['correct_probability']-np.mean(goldp))<1e-10
 write(root/'AUDIT.json',dict(at=now(),status='passed',responses=2000,summary_sha256=sha(root/'SUMMARY.json'),note='Independent numpy recomputation from all rotation candidate logprobs; technical checks passed.'))
if __name__=='__main__':main()
