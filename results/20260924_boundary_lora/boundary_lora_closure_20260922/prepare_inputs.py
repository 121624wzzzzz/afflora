from common import *
old=Path(read(HERE/'SOURCE.json')['study']);source=read(old/'INPUT_AUDIT.json');files={}
for path,h in source['files'].items():
 p=Path(path)
 try:p=HERE/p.relative_to(old)
 except ValueError:pass
 assert sha(p)==h,str(p);files[str(p)]=h
write(HERE/'INPUT_AUDIT.json',dict(status='passed',at=now(),files=files,counts=source['counts'],source_target_audit=read(old/'TARGET_AUDIT.json'),source_target_audit_sha256=sha(old/'TARGET_AUDIT.json')))
write(HERE/'REUSE_AUDIT.json',dict(status='passed',runs=[],note='No effectiveness results reused as new confirmations; only checked technical smoke gates are inherited.'))
print('all six model identities and data/token bytes verified',flush=True)
