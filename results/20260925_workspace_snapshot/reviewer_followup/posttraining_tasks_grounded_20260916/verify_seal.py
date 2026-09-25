"""Read-only verification of this archive after sealing."""
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    root = Path(__file__).resolve().parent
    manifest_path = root / 'ARTIFACT_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['status'] == 'sealed'
    expected = set(manifest['files'])
    actual = {str(p.relative_to(root)) for p in root.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p != manifest_path}
    assert actual == expected, {'added': sorted(actual - expected), 'missing': sorted(expected - actual)}
    for relative, record in manifest['files'].items():
        path = root / relative
        assert path.stat().st_size == record['bytes'], relative
        assert sha(path) == record['sha256'], relative
    print(f'PASS: {len(expected)} files; manifest SHA256 {sha(manifest_path)}')


if __name__ == '__main__':
    main()
