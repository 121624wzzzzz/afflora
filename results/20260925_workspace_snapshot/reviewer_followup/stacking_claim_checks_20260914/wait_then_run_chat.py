"""Launch the fixed chat matrix once the earlier diagnostic releases its GPUs."""
import json
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
PYTHON='/home/wz/anaconda3/envs/torch24/bin/python'


def main():
    print('Waiting for all 24 calibration checkpoints and final audit.',flush=True)
    while not (HERE/'calibration/FINAL_AUDIT.json').exists():
        state=HERE/'calibration/state.json'
        if state.exists():
            s=json.loads(state.read_text())
            failed=[n for n,j in s.get('jobs',{}).items() if j.get('status')=='failed']
            if failed:raise RuntimeError('Calibration requires attention: '+str(failed))
        time.sleep(10)
    assert json.loads((HERE/'calibration/FINAL_AUDIT.json').read_text())['status']=='passed'
    print('Calibration audit passed; starting preflight, smokes and all chat cells.',flush=True)
    subprocess.run([PYTHON,'-u',str(HERE/'chat/run.py'),'run'],check=True)


if __name__=='__main__':main()
