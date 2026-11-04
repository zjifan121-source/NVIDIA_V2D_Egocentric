"""Cloud-host (A100) orchestrator for an object update: take the winning candidate bundles of a GPU batch (full bundles
exported by evaluate_remote.py on the A6000: current bundle with one slot's mesh+poses swapped), build a mini stage
(scenes/episode_X/{bundle,run.json,support_v1=fb11 unseated spec or the v2 table cylinders}), re-seat the supports
(seat_supports.py rule), then run the hand stage against the NEW objects into a separate output root, and compare.
Usage: V2D_ROOT=/root/v2d apply_winners_remote.py <winners.json> <mini_stage_out>
winners.json: {"ep": {"case": "<case name>", "candidate_bundle": "<path on this host>", "object": ..., "slot": ...}, ...}"""
from pathlib import Path
import os, sys, json, shutil, subprocess
ROOT=Path(os.environ.get('V2D_ROOT','/root/v2d')); PY=os.environ.get('PY','/root/venv-cpu/bin/python')
H=ROOT/'NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261103/lead_hand_reattach_v1'; SEATS=ROOT/'NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261103/lead_support_seating_v1'
V2=ROOT/'work/lead-sprint-release-v2-20261103/stage'; V1=ROOT/'work/lead-sprint-release-20261103/stage'
winners=json.loads(Path(sys.argv[1]).read_text()); M=Path(sys.argv[2]).resolve()
read=lambda p:json.loads(Path(p).read_text())
def put(p,x): Path(p).parent.mkdir(parents=True,exist_ok=True); Path(p).write_text(json.dumps(x,indent=2)+'\n')
if M.exists(): shutil.rmtree(M)
(M/'tools').mkdir(parents=True); shutil.copy2(V1/'tools/install_support.py',M/'tools/install_support.py')
inv={int(r['episode']):r for r in read(V1/'inventory.json')}
eps=[]
for ep,w in winners.items():
    ep=int(ep); sd=M/'scenes'/f'episode_{ep:06d}'; shutil.copytree(w['candidate_bundle'],sd/'bundle',symlinks=False)
    put(sd/'run.json',dict(episode=ep,objects=inv[ep]['objects'],base='lead_sprint_v1 + v8 winner',changed=[dict(object=w['object'],slot=w['slot'],source=w['case'])]))
    uns=V2/'scenes'/f'episode_{ep:06d}'/'support_v1_fb11_unseated'
    if uns.exists(): shutil.copytree(uns,sd/'support_v1')
    else:  # no fb11 support (ep10/ep38): start from the v2 seated table cylinders only (drop the per-object seat disks)
        spec=read(V2/'scenes'/f'episode_{ep:06d}'/'support_v1/support_in_bundle_world.json'); spec['cylinders']=[c for c in spec['cylinders'] if c.get('kind','table')=='table']
        for k in ['seating']: spec.pop(k,None)
        put(sd/'support_v1/support_in_bundle_world.json',spec)
    eps.append(ep); print('MINI_STAGE',ep,w['case'],flush=True)
r=subprocess.run([PY,str(SEATS/'seat_supports.py'),str(M),str(M/'seated_out')],capture_output=True,text=True); print(r.stdout[-3000:],r.stderr[-1500:],flush=True); assert r.returncode==0,'seating failed'
env=dict(os.environ,V2D_ROOT=str(ROOT),LEAD_BUNDLES_ROOT=str(M/'scenes'),LEAD_SEAT_ROOT=str(M/'seated_out'),LEAD_HAND_OUT=str(M/'hand_stage'),PY=PY)
(M/'hand_stage').mkdir(exist_ok=True)
r=subprocess.run(['bash',str(H/'run_hand_stage_remote.sh'),*[str(e) for e in eps]],cwd=H,env=env,capture_output=True,text=True); print(r.stdout[-6000:],r.stderr[-1500:],flush=True)
print('APPLY_WINNERS_DONE',eps,flush=True)
