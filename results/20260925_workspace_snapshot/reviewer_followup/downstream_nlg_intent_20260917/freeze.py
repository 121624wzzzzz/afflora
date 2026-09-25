from common import *
assert not (HERE/'checkpoints').exists()
files={str(p.relative_to(HERE)):sha(p) for parent in ['raw','data','tokens'] for p in (HERE/parent).rglob('*') if p.is_file() and '__pycache__' not in str(p)}
write(HERE/'DATA_FROZEN.json',{'at':now(),'files':files})
files={str(p.relative_to(HERE)):sha(p) for p in list(HERE.glob('*.py'))+list((HERE/'source').glob('*.py'))+list((HERE/'vendor').rglob('*'))+[HERE/'PROTOCOL.md',HERE/'SLOT_SCORER_PATCH.md',HERE/'models.json',HERE/'SOURCE_REUSE.json'] if p.is_file() and '__pycache__' not in str(p) and p.suffix!='.pyc'}
write(HERE/'CODE_FROZEN.json',{'at':now(),'files':files})
print('frozen',len(files),'code/dependency files')
