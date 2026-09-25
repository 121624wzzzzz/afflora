"""Independent read-only verifier; its output must not be redirected into archive."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while True:
            data=f.read(4*1024*1024)
            if not data:break
            h.update(data)
    return h.hexdigest()
def main():
    seal=json.loads((ROOT/'SEAL.json').read_text());manifest_path=ROOT/'SEAL_MANIFEST.json'
    assert digest(manifest_path)==seal['manifest_sha256']
    manifest=json.loads(manifest_path.read_text());expected=manifest['files'];actual={}
    for p in ROOT.rglob('*'):
        if '__pycache__' in p.parts or p.suffix=='.pyc':continue
        assert not p.is_symlink(),p
        if p.is_file() and p.relative_to(ROOT).as_posix() not in ['SEAL_MANIFEST.json','SEAL.json']:
            actual[p.relative_to(ROOT).as_posix()]=p
    assert set(actual)==set(expected),{'missing':sorted(set(expected)-set(actual)),'extra':sorted(set(actual)-set(expected))}
    for i,(rel,p) in enumerate(sorted(actual.items())):
        r=expected[rel];assert p.stat().st_size==r['bytes'],rel;assert digest(p)==r['sha256'],rel
        if (i+1)%500==0:print(f'Verified {i+1}/{len(actual)} files',flush=True)
    assert len(actual)==manifest['file_count']==seal['file_count']
    assert sum(v['bytes'] for v in expected.values())==manifest['total_bytes']
    verified=datetime.now(timezone.utc);deadline=datetime.fromisoformat(json.loads((ROOT/'WINDOW.json').read_text())['deadline_utc'])
    print(json.dumps({'status':'passed','verified_at':verified.isoformat(),'within_requested_deadline':verified<=deadline,
                      'file_count':len(actual),'manifest_sha256':seal['manifest_sha256']}))
if __name__=='__main__':main()
