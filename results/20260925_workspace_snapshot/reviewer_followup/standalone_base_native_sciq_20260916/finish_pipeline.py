"""Sequential execution of the already frozen phases after tuning finishes."""
import os,subprocess,time
from settings import *

def main():
    start=time.monotonic()
    while not (HERE/'TUNING_COMPLETE.json').exists():
        assert not list((HERE/'checkpoints').glob('*/FAILED.json')),'Tuning failure requires inspection'
        assert time.monotonic()-start<3600,'Tuning wait timed out'
        time.sleep(5)
    assert not (HERE/'SELECTION.json').exists(),'Do not overwrite a selection'
    for phase in ['select','reference','confirmation']:
        write(HERE/'PIPELINE_PROGRESS.json',{'at':now(),'stage':phase,'status':'running'})
        subprocess.run([PYTHON,'-u',str(HERE/'run.py'),phase],check=True)
    for name in ['analyze.py','final_audit.py','report.py']:
        write(HERE/'PIPELINE_PROGRESS.json',{'at':now(),'stage':name,'status':'running'})
        subprocess.run([PYTHON,'-u',str(HERE/name)],check=True)
    write(HERE/'PIPELINE_PROGRESS.json',{'at':now(),'status':'computed_and_audited_awaiting_human_readable_review'})
if __name__=='__main__':main()
