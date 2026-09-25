"""Compact read-only polling of this follow-up's own state and logs."""
from datetime import datetime
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent


def main():
    print(datetime.now().astimezone().isoformat(timespec='seconds'))
    for kind in ['calibration','chat']:
        root=HERE/kind;p=root/'state.json'
        if not p.exists():print(kind,'waiting');continue
        state=json.loads(p.read_text())
        if kind=='calibration':
            completed=len([p for p in (root/'reports').glob('*.json') if not p.name.endswith('.selection.json')])
            total=24
        else:
            completed=sum(state['jobs'].get(j['name'],{}).get('status')=='complete' for j in state.get('matrix',[]))
            total=24
        print(kind,state.get('phase'),str(completed)+'/'+str(total))
        if kind=='chat':
            refs=state.get('references',[])
            print('unadapted references',sum(state['jobs'].get(j['name'],{}).get('status')=='complete' for j in refs),'/',len(refs))
        for name,job in state.get('jobs',{}).items():
            if job['status'] in ['complete','smoke_complete']:continue
            log=Path(job.get('log',''))
            lines=log.read_text(errors='replace').splitlines() if log.is_file() else []
            progress=[x for x in lines if x.startswith(name+' dev ') or x.startswith(name+' test ') or x.startswith('generated ') or ('%|' in x and re.search(r'\d+/1424',x))]
            print('gpu',job.get('gpu'),name,job['status'],progress[-1][-180:] if progress else '',job.get('error',''))
        audit=root/'FINAL_AUDIT.json'
        if audit.exists():print(kind,'audit',json.loads(audit.read_text()).get('status'))


if __name__=='__main__':main()
