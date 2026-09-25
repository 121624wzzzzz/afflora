"""Preserve failed preflight; revise diagnostics only after exact controlled checks."""
import hashlib,json,shutil
from pathlib import Path
import torch
from safetensors.torch import load_file

ROOT=Path(__file__).resolve().parent
DST=ROOT/'llama_lr_20260918';ARCHIVE=ROOT/'llama_lr_20260918_preflight_v1';SEALED=ROOT/'llama_transfer_20260918'
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')

def main():
    torch.set_num_threads(2)
    assert read(DST/'SCHEDULER_COMPLETE.json')['status']=='failed'
    assert not ARCHIVE.exists() and not (DST/'SELECTION.json').exists()
    assert all(j['state']=='pending' for j in read(DST/'STATE.json')['jobs'] if j['spec']['stage']=='search')
    assert sha(SEALED/'SEAL_MANIFEST.json')=='a995f154415e13d1a3200ebcb85b377b9ad704c5628d8e2d545e845d58fd7599'
    sealed=read(SEALED/'SEAL_MANIFEST.json')['files'];old=SEALED/'checkpoints/wikisql_llama31_8b_base_hidden_budget_s7100'
    for filename in ['TRAINING.json','TRAIN_ORDER.json','initial_adapter.safetensors','adapter.safetensors']:
        p=old/filename;assert sha(p)==sealed[str(p.relative_to(SEALED))]['sha256']
    a=load_file(str(old/'adapter.safetensors'));initial=load_file(str(old/'initial_adapter.safetensors'));history=read(old/'TRAINING.json')['history'];checks=[]
    for name in ['diagnostic_original_hidden_budget_s7100_gpu1','diagnostic_original_hidden_budget_s7100_gpu7','diagnostic_cpu_instrumentation_hidden_budget_s7100']:
        p=DST/'checkpoints'/name;assert read(p/'COMPLETE.json')['status']=='passed'
        b=load_file(str(p/'adapter.safetensors'));bi=load_file(str(p/'initial_adapter.safetensors'))
        assert a.keys()==b.keys() and all(torch.equal(v,b[k]) for k,v in a.items()),name
        assert initial.keys()==bi.keys() and all(torch.equal(v,bi[k]) for k,v in initial.items()),name
        tr=read(p/'TRAINING.json');assert tr['frozen_before']==tr['frozen_after'] and tr['reload_loss_error']==0
        assert read(p/'TRAIN_ORDER.json')==read(old/'TRAIN_ORDER.json')
        assert [(r['loss'],r['grad_norm']) for r in history]==[(r['loss'],r['grad_norm']) for r in tr['history']],name
        checks.append({'name':name,'initial_final_tensors_and_history_exact':True,'adapter_sha256':sha(p/'adapter.safetensors')})
    assert read(DST/'COMPATIBILITY_DIFFERENCE.json')['hidden_both']['n_changed_tensors']==0
    diagnostic={'status':'passed','checks':checks,'finding':'Unmodified original training exactly reproduced on GPUs 1 and 7; CPU instrumentation exactly reproduced it. GPU instrumentation changed budget-H trajectory. Precise CUDA allocation/kernel mechanism not identified.',
                'original_script_sha256':sha(DST/'run_original_compatibility_diagnostic.py'),'cpu_probe_script_sha256':sha(DST/'run_cpu_diagnostics_probe.py'),
                'scope':'No search fits or confirmation responses have been generated; only preflight/diagnostic fits.'}
    write(DST/'REPRODUCTION_DIAGNOSTIC_RESULT.json',diagnostic)
    frozen=read(DST/'CODE_FROZEN.json')
    for rel,h in frozen['files'].items():assert sha(DST/rel)==h,rel
    preserved={str(p.relative_to(DST)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(DST.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    write(DST/'PREFLIGHT_ARCHIVE_MANIFEST.json',{'status':'preserved_failed_preflight','files':preserved})
    DST.rename(ARCHIVE);DST.mkdir()
    for rel in frozen['files']:
        p=ARCHIVE/rel;q=DST/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    for directory in ['source','raw','data','tokens']:
        shutil.copytree(ARCHIVE/directory,DST/directory,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for filename in ['README.md','BOUNDARY_SCALE_DIAGNOSIS_ZH.md','BOUNDARY_DISPLACEMENT_POSTHOC.json','BOUNDARY_WEIGHT_SCALES_POSTHOC.json']:
        shutil.copy2(ARCHIVE/filename,DST/filename)
    s=(DST/'run.py').read_text()
    s=s.replace('return float(torch.linalg.vector_norm(torch.stack(torch._foreach_norm(tensors)))) if tensors else 0.',
                'return float(torch.linalg.vector_norm(torch.stack(torch._foreach_norm([t.detach().cpu() for t in tensors])))) if tensors else 0.')
    s=s.replace('p.detach().clone() for p in ps','p.detach().cpu().clone() for p in ps')
    s=s.replace('[p-b for p,b in zip(ps,before_update[k])]','[p.detach().cpu()-b for p,b in zip(ps,before_update[k])]')
    assert s.replace(';validate_spec(spec)','')==(ARCHIVE/'run_cpu_diagnostics_probe.py').read_text()
    (DST/'run.py').write_text(s)
    revision={'revision':2,'archive':str(ARCHIVE),'archive_manifest_sha256':sha(ARCHIVE/'PREFLIGHT_ARCHIVE_MANIFEST.json'),
              'diagnosis':diagnostic,'unchanged':'Candidates, search/confirmation seeds, datasets, optimizer equations, clipping, parameter counts, selection and analysis rules.',
              'changed':'Diagnostic gradient norms and parameter-update snapshots are computed on CPU, avoiding added CUDA tensor allocations.',
              'validation':'Repeat all six smoke fits and both full exact-compatibility fits before search admission.'}
    write(DST/'PREFLIGHT_REVISION.json',revision)
    p=DST/'PROTOCOL.md';p.write_text(p.read_text()+'''\n## Preflight-only implementation revision (v2)\n\nThe preserved v1 preflight failed exact final-tensor reproduction for budget-H,\nwhile H+E+U reproduced exactly. Two unmodified-original repetitions on GPUs 1/7\nand a CPU-instrumented repetition all reproduced the original budget-H initial\nand final tensors and every step's loss/norm exactly. Extra GPU diagnostic\nallocations affected the numerical training trajectory; the specific CUDA kernel\nmechanism was not identified. V2 moves only diagnostic copies/norm calculations\nto CPU. See PREFLIGHT_REVISION.json and the preserved preflight archive.\nNo search fit or confirmation response preceded this revision. The grid, seeds,\ndata, optimizer update, clipping and statistical rules remain unchanged. Repeat\nall smoke and exact compatibility gates before admitting formal work.\n''')
    p=DST/'freeze.py';s=p.read_text().replace("'SOURCE_REUSE.json','DATA_AUDIT.json'","'SOURCE_REUSE.json','PREFLIGHT_REVISION.json','DATA_AUDIT.json'");p.write_text(s)
    print('preserved',ARCHIVE,'created CPU-instrumented revision',DST,flush=True)
if __name__=='__main__':main()
