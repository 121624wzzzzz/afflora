"""Seal/independently verify the completed LR follow-up; never rewrite a seal."""
import argparse,hashlib,json
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parent/'llama_lr_20260918'
EXCLUDE={'SEAL.json','SEAL_MANIFEST.json'}
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def now():return datetime.now().astimezone().isoformat()
def files(root):return sorted(p for p in root.rglob('*') if p.is_file() and p.name not in EXCLUDE and '__pycache__' not in p.parts and p.suffix!='.pyc')
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')

def integrity():
    assert read(ROOT/'SCHEDULER_COMPLETE.json')['status']=='passed'
    audit=read(ROOT/'FINAL_AUDIT.json');assert audit['status']=='passed'
    assert audit['analysis_sha256']==sha(ROOT/'ANALYSIS.json') and audit['selection_sha256']==sha(ROOT/'SELECTION.json')
    for f in ['CODE_FROZEN.json','DATA_FROZEN.json']:
        for rel,h in read(ROOT/f)['files'].items():assert sha(ROOT/rel)==h,rel
    revision=read(ROOT/'PREFLIGHT_REVISION.json');archive=Path(revision['archive'])
    assert sha(archive/'PREFLIGHT_ARCHIVE_MANIFEST.json')==revision['archive_manifest_sha256']
    for rel,meta in read(archive/'PREFLIGHT_ARCHIVE_MANIFEST.json')['files'].items():
        p=archive/rel;assert p.stat().st_size==meta['bytes'] and sha(p)==meta['sha256'],rel
    assert read(ROOT/'COMPATIBILITY.json')['status']=='passed'
    assert (ROOT/'FINAL_INTERPRETATION_ZH.md').exists()
    assert '此处由完成后的' not in (ROOT/'FINAL_INTERPRETATION_ZH.md').read_text()
    probes=[read(ROOT/'mechanism_probe'/f'{m}.json') for m in ['llama31_8b_base','qwen3_8b_base']]
    assert all(p['status']=='passed' and p['frozen_and_adapter_parameters_unchanged'] for p in probes)
    assert probes[0]['example_ids']==probes[1]['example_ids']
    assert probes[0]['boundary_initialization_sha256']==probes[1]['boundary_initialization_sha256']
    assert read(ROOT/'TABLE_BOOTSTRAP.json')['status']=='passed'
    for plan_name,script in [('MECHANISM_PROBE_PLAN.json','probe_boundary_initial_gradients_20260918.py'),('TABLE_BOOTSTRAP_PLAN.json','bootstrap_llama_lr_tables_20260918.py')]:
        snapshot=ROOT/'provenance'/(script+'.txt')
        assert sha(snapshot if snapshot.exists() else ROOT.parent/script)==read(ROOT/plan_name)['script_sha256']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--verify',action='store_true');args=ap.parse_args()
    if args.verify:
        seal=read(ROOT/'SEAL.json');manifest=read(ROOT/'SEAL_MANIFEST.json')
        assert seal['manifest_sha256']==sha(ROOT/'SEAL_MANIFEST.json')
        assert set(manifest['files'])=={str(p.relative_to(ROOT)) for p in files(ROOT)}
        for rel,meta in manifest['files'].items():
            p=ROOT/rel;assert p.stat().st_size==meta['bytes'] and sha(p)==meta['sha256'],rel
        integrity();print(json.dumps({'at':now(),'status':'passed','file_count':len(manifest['files']),'manifest_sha256':sha(ROOT/'SEAL_MANIFEST.json')}),flush=True);return
    assert not (ROOT/'SEAL.json').exists() and not (ROOT/'SEAL_MANIFEST.json').exists()
    integrity()
    provenance=ROOT/'provenance';provenance.mkdir(exist_ok=True)
    for name in ['seal_llama_lr_20260918.py','bootstrap_llama_lr_20260918.py','revise_llama_lr_20260918_preflight.py',
                 'probe_boundary_weight_scales_20260918.py','probe_boundary_displacement_20260918.py','probe_boundary_initial_gradients_20260918.py','bootstrap_llama_lr_tables_20260918.py','report_llama_lr_20260918.py']:
        p=ROOT.parent/name
        if p.exists():(provenance/(name+'.txt')).write_bytes(p.read_bytes())
    entries={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files(ROOT)}
    write(ROOT/'SEAL_MANIFEST.json',{'at':now(),'status':'sealed','excluded':sorted(EXCLUDE)+['**/__pycache__/**','**/*.pyc'],
                                  'file_count':len(entries),'total_bytes':sum(v['bytes'] for v in entries.values()),'files':entries})
    write(ROOT/'SEAL.json',{'at':now(),'status':'sealed','manifest_sha256':sha(ROOT/'SEAL_MANIFEST.json'),'file_count':len(entries),
                         'preflight_archive_manifest_sha256':read(ROOT/'PREFLIGHT_REVISION.json')['archive_manifest_sha256']})
    print(json.dumps(read(ROOT/'SEAL.json')),flush=True)
if __name__=='__main__':main()
