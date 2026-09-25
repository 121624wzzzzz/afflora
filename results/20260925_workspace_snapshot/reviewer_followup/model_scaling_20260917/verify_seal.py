"""Read-only verification, separate from the sealing process."""
from common import *

manifest=read(HERE/'ARTIFACT_MANIFEST.json')
actual={str(p.relative_to(HERE)) for p in HERE.rglob('*')
        if p.is_file() and '__pycache__' not in p.parts and p.name!='ARTIFACT_MANIFEST.json'}
assert actual==set(manifest['files']),(actual-set(manifest['files']),set(manifest['files'])-actual)
for rel,expected in manifest['files'].items():
    p=HERE/rel
    assert p.stat().st_size==expected['bytes'] and sha(p)==expected['sha256'],rel
print(canonical({'status':'passed','files':len(actual),'manifest_sha256':sha(HERE/'ARTIFACT_MANIFEST.json')}))
