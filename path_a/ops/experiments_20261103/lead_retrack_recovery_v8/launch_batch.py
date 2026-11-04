"""Host-side launcher for lead_retrack_recovery_v1. Same container/mounts as scored11-fulltrack-20261103.
Runs enabled cases in FROZEN_BATCH order; per-case timeout; stops on first non-zero exit; no retry."""
from pathlib import Path
import subprocess,json,os,sys,time
P=Path(__file__).resolve().parent;repo=Path('/mnt/v2d/code/NVIDIA_V2D_Egocentric')
sys.path.insert(0,str(repo/'path_a/ops'));import storage
batch=json.loads((P/'FROZEN_BATCH.json').read_text());start=time.monotonic();rows=[];failed=[]
def save(state='RUNNING'):
 (P/'BATCH_RUNTIME.json').write_text(json.dumps(dict(state=state,seconds=time.monotonic()-start,cases=rows),indent=2))
only=set(sys.argv[1:])
for c in batch['cases']:
 if not c['enabled'] or (only and c['case'] not in only):continue
 case=P/'cases'/c['case'];out=case/'output';out.mkdir(exist_ok=False)
 for n in ['rgb','depth','masks']:(case/'payload'/n).mkdir(exist_ok=True)
 (case/'payload/K.json').touch(exist_ok=True)
 volumes=storage.volumes();assert all(v['verified'] and v['free_gib']>=40 for v in volumes)
 assert not subprocess.check_output(['docker','ps','-q'],text=True).strip()
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 if time.monotonic()-start>=batch['total_budget_seconds']:raise TimeoutError('batch GPU budget reached')
 ep=c['episode'];name=f"v2d-lead-{c['case']}";cache=f'/var/tmp/v2d_scored20_dual_anchor_v1/ep{ep:02d}'
 cmd=['docker','run','--rm','--pull=never','--network','none','--name',name,'--gpus','device=0','--user',f'{os.getuid()}:{os.getgid()}',
 '-e','HOME=/tmp','-e','HF_HUB_OFFLINE=1','-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True','--tmpfs','/tmp:rw,size=2g',
 '-v',str(case/'payload')+':/payload:ro',
 '-v',cache+'/rgb:/payload/rgb:ro','-v',cache+'/depth:/payload/depth:ro','-v',cache+'/masks:/payload/masks:ro','-v',cache+'/K.json:/payload/K.json:ro',
 '-v','/mnt/v2d-data1:/mnt/v2d-data1:ro','-v',str(out)+':/output','-v',str(P)+':/package:ro',
 '-v',str(repo/'baseline/reconstruction/modules')+':/workspace:ro',
 '-v',str(P/'foundation_pose_tracker.py')+':/workspace/v2d_foundation_pose/lib/foundation_pose_tracker.py:ro',
 '-v',str(repo/'baseline/reconstruction/data/weights')+':/weights:ro',
 '-v','/workspace/v2d_foundation_pose/lib/FoundationPose/mycpp/build','-v','/workspace/v2d_foundation_pose/lib/FoundationPose/bundlesdf/mycuda',
 'v2d_foundation_pose:afe7dad-trt10.7','python','-u','/package/driver.py','/package/cases/'+c['case']]
 tick=time.monotonic();print('START',c['case'],c['frames'],flush=True)
 try:
  with (case/'GPU.log').open('x') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=min(c['timeout_seconds'],batch['total_budget_seconds']-(time.monotonic()-start)))
  code=r.returncode
 except subprocess.TimeoutExpired:
  subprocess.run(['docker','stop','--time','10',name],check=False);code=124
 rec=dict(case=c['case'],exit_code=code,seconds=time.monotonic()-tick,volumes=volumes)
 (case/'LAUNCH.json').write_text(json.dumps(rec,indent=2));rows.append(rec);save();print('DONE',rec,flush=True)
 if code:failed.append(c['case']);print('FAILED',c['case'],'exit',code,'(continuing; no retry)',flush=True)
subprocess.run(['sync'],check=True)
rows_state='COMPLETE' if not failed else 'COMPLETE_WITH_FAILURES:'+','.join(failed)
(P/'BATCH_RUNTIME.json').write_text(json.dumps(dict(state=rows_state,seconds=time.monotonic()-start,cases=rows,
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip(),
 containers=subprocess.check_output(['docker','ps','-q'],text=True).strip(),volumes=storage.volumes()),indent=2));print('BATCH_COMPLETE',flush=True)
