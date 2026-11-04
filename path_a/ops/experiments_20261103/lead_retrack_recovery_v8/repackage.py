import json, hashlib, tarfile, sys
from pathlib import Path
X=Path(sys.argv[1]); sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
members=[]
for p in sorted(X.rglob('*')):
    if not p.is_file() or p.name in ('deploy.tar.gz','PACKAGE_SHA256.json') or p.name.startswith('build_cases') or p.suffix=='.pyc': continue
    rp=p.relative_to(X)
    if len(rp.parts)>=4 and rp.parts[0]=='cases' and rp.parts[2]=='payload' and rp.parts[3] in ('rgb','depth','masks'): continue
    if rp.parts[0]=='run_v1': continue
    members.append(p)
(X/'PACKAGE_SHA256.json').write_text(json.dumps({str(p.relative_to(X)):sha(p) for p in members},indent=2)+'\n')
pkg=X/'deploy.tar.gz'; pkg.unlink(missing_ok=True)
with tarfile.open(pkg,'w:gz') as t:
    for p in members+[X/'PACKAGE_SHA256.json']: t.add(p,arcname=str(p.relative_to(X)))
print('PACKAGE',pkg.stat().st_size,len(members))
