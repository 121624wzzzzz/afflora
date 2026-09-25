import hashlib
from concurrent.futures import ThreadPoolExecutor
import requests
from common import HERE, now, sha, write

SPECS = {
 'cmrc2018': ('ymcui/cmrc2018', 'c0eb1b6ba219847457e6af3180da722bbeb656af', [
     'README.md', 'LICENCE', 'squad-style-data/cmrc2018_train.json',
     'squad-style-data/cmrc2018_dev.json', 'squad-style-data/cmrc2018_evaluate.py']),
 'c3': ('nlpdata/c3', '1f41682ebb684d63afe7c583c8f38e137fc8a826', [
     'README.md','license.txt']+[f'data/c3-{part}-{split}.json' for part in ['d','m'] for split in ['train','dev','test']])}

def fetch(spec):
    name,(repo,commit,paths)=spec
    session=requests.Session();session.trust_env=False
    tree=session.get(f'https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1',timeout=30)
    tree.raise_for_status(); entries={x['path']:x for x in tree.json()['tree']}
    records={}
    for path in paths:
        url=f'https://raw.githubusercontent.com/{repo}/{commit}/{path}'
        response=session.get(url,timeout=60);response.raise_for_status();data=response.content
        blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        assert blob==entries[path]['sha'],path
        target=HERE/'vendor'/name/path;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists(): assert target.read_bytes()==data
        else: target.write_bytes(data)
        records[str(target)]={'url':url,'git_blob_sha1':blob,'sha256':sha(target),'bytes':len(data)}
    print(name,'downloaded',len(records),'verified blobs',flush=True)
    return name,{'repository':repo,'commit':commit,'files':records}

if __name__=='__main__':
    with ThreadPoolExecutor(2) as pool: repos=dict(pool.map(fetch,SPECS.items()))
    write(HERE/'DATA_PROVENANCE.json',{'downloaded_at':now(),'repositories':repos})
