"""Read-only verification of this completed archive and its original sources."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def check(root):
    manifest=read(root/'ARTIFACT_MANIFEST.json')
    for row in manifest['files']:
        path=root/row['path']
        assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],path
    return len(manifest['files'])


if __name__=='__main__':
    assert read(HERE/'COMPLETION.json')['status']=='complete'
    count=check(HERE);sources=0
    for name,record in read(HERE/'REUSE_AUDIT.json')['sealed_sources'].items():
        source=HERE.parent/name;assert sha(source/'ARTIFACT_MANIFEST.json')==record['manifest_sha256']
        sources+=check(source)
    files={p:h for model in read(HERE/'models.json').values() for p,h in model['files'].items()}
    for path,digest in files.items():assert sha(path)==digest,path
    print(json.dumps({'status':'passed','archive_files':count,'source_files':sources,'model_files':len(files),
                      'manifest_sha256':sha(HERE/'ARTIFACT_MANIFEST.json'),'writes_performed':False},indent=2))
