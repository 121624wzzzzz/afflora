"""Operational resumption only; frozen fitting/scoring/statistical code is unchanged.

Original five allocated GPUs may be shared, with at least 48 GiB free before
starting one worker. No external process is stopped or modified.
"""
import subprocess
import pipeline
from common import *

POOL={2,3,4,5,6}
MIN_FREE_MIB=48*1024

def available_gpus():
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used',
                                 '--format=csv,noheader,nounits'],text=True)
    return {i for line in out.splitlines() for i,total,used in [map(int,line.split(','))]
            if i in POOL and total-used>=MIN_FREE_MIB}

def main():
    amendment=read(HERE/'RESOURCE_AMENDMENT.json')
    assert amendment['status']=='recorded_before_resumption'
    assert amendment['new_scheduler_sha256']==sha(HERE/'pipeline_shared.py')
    assert not Path(f"/proc/{amendment['previous_scheduler_pid']}").exists(), 'Old scheduler still alive'
    state=read(HERE/'FORMAL_STATE.json')
    assert not state['active'] and not state['failed']
    assert state==read(HERE/'SCHEDULER_PAUSE_SNAPSHOT.json')
    smoke,formal=pipeline.make_jobs();completed=[];adapters=0;evals=0
    for spec in smoke+formal:
        p=HERE/'checkpoints'/spec['name']
        if not p.exists():continue
        assert read(p/'COMPLETE.json')['status']=='passed'
        assert read(p/'spec.json')==spec
        if spec['arm']!='base':
            t=read(p/'TRAINING.json')
            assert sha(p/'adapter.safetensors')==t['adapter_sha256']
            assert t['frozen_before']==t['frozen_after'] and t['reload_loss_error']==0
            adapters+=1
        for e in (HERE/'evaluations'/spec['name']).glob('*/SUMMARY.json'):
            assert sha(e.parent/'responses.jsonl')==read(e)['responses_sha256'];evals+=1
        completed.append(spec['name'])
    write(HERE/'OPERATIONAL_RESUME_CHECK.json',{'at':now(),'status':'passed','completed_runs':completed,
        'adapter_files_rehashed':adapters,'evaluation_response_files_rehashed':evals,
        'gpu_pool':sorted(POOL),'minimum_free_memory_before_launch_MiB':MIN_FREE_MIB})
    pipeline.free_gpus=available_gpus
    pipeline.main()

if __name__=='__main__':main()
