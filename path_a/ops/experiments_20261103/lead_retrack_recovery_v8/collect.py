"""Mac side: snapshot outputs on the host, scp back, verify SHA, extract without overwriting existing files.
Usage: V2D_GPU_IP=<ip> V2D_HOSTKEY_ALIAS=<alias> python collect.py"""
from pathlib import Path
import subprocess,json,hashlib,tarfile,os
W=Path(__file__).resolve().parent;remote='/var/tmp/v2d_lead_retrack_recovery_v7'
ip=os.environ['V2D_GPU_IP'];alias=os.environ['V2D_HOSTKEY_ALIAS']
opts=['-i',str(Path.home()/'.ssh/hyperstack_v2d'),'-o','StrictHostKeyChecking=yes','-o','HostKeyAlias='+alias,'-o','BatchMode=yes','-o','ConnectTimeout=12'];host='ubuntu@'+ip
subprocess.run(['scp',*opts,str(W/'snapshot.py'),host+':'+remote+'/snapshot.py'],check=True)
r=json.loads(subprocess.check_output(['ssh',*opts,host,'python3 '+remote+'/snapshot.py'],text=True));dest=W/'run_v1'/Path(r['archive']).name;dest.parent.mkdir(exist_ok=True)
if not dest.exists():subprocess.run(['scp',*opts,host+':'+r['archive'],str(dest)],check=True)
assert hashlib.sha256(dest.read_bytes()).hexdigest()==r['sha256']
with tarfile.open(dest) as t:
 for member in t.getmembers():
  p=(W/'run_v1'/member.name).resolve();assert p.is_relative_to(W/'run_v1') and member.isfile()
  data=t.extractfile(member).read()
  if p.exists() and p.read_bytes()!=data:
   k=1
   while p.with_name(p.name+f'.superseded{k}').exists():k+=1
   p.rename(p.with_name(p.name+f'.superseded{k}'));print('SUPERSEDED',member.name,k,flush=True)
  if not p.exists():p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
m=json.loads((W/'run_v1'/f"SNAPSHOT_{len(r['cases']):02d}.json").read_text());assert all(hashlib.sha256((W/'run_v1'/k).read_bytes()).hexdigest()==v for k,v in m['files'].items())
print('SHA_VERIFIED',len(m['files']),r['cases'],flush=True)
