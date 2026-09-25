from common import *

def main():
    for stage in ['SMOKE','PILOT','ADAPTER_PILOT','CONFIRMATION']:
        p=HERE/f'{stage}_STATE.json'
        if p.exists():
            state=read(p);print(stage,'finished',len(state['finished']),'failed',len(state['failed']),'active',len(state['active']),'pending',len(state['pending']))
            for r in state['active']:
                train=HERE/'checkpoints'/r['name']/'PROGRESS.json'
                ev=sorted((HERE/'evaluations'/r['name']).glob('*/PROGRESS.json'),key=lambda x:x.stat().st_mtime)
                print(' ',r['name'],'GPU',r['gpu'], 'train '+str(read(train)) if train.exists() else '', 'eval '+ev[-1].parent.name+' '+str(read(ev[-1])) if ev else '')
    for p in sorted((HERE/'evaluations').glob('*/*/SUMMARY.json')):
        s=read(p);print('RESULT',p.parent.parent.name,p.parent.name,{k:round(s[k],3) for k in ['primary','json_valid','strict_json','native_eos_pct','capped_pct','answer_ce']},'text_f1',s.get('text_micro_f1'))
if __name__=='__main__':main()
