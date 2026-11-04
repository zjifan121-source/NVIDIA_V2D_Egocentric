from pathlib import Path
import json,tarfile,hashlib
P=Path(__file__).resolve().parent;batch=json.loads((P/'FROZEN_BATCH.json').read_text());files=[];done=[]
for c in batch['cases']:
 case=P/'cases'/c['case'];r=case/'output/track/RECEIPT.json'
 if not (case/'LAUNCH.json').exists():continue
 done.append(c['case']);files+=sorted(p for p in (case/'output').rglob('*') if p.is_file());files+=[case/'GPU.log',case/'LAUNCH.json']
for extra in ['BATCH_RUNTIME.json','CLOUD_INVENTORY.txt']:
 if (P/extra).exists():files.append(P/extra)
name=P/f'snapshot_{len(done):02d}.tar.gz'
if not name.exists():
 hashes={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
 meta=P/f'SNAPSHOT_{len(done):02d}.json';meta.write_text(json.dumps(dict(cases=done,files=hashes),indent=2))
 with tarfile.open(name,'w:gz') as t:
  for p in [*files,meta]:t.add(p,arcname=str(p.relative_to(P)))
print(json.dumps(dict(cases=done,archive=str(name),sha256=hashlib.sha256(name.read_bytes()).hexdigest())))
