"""Revision r2 of the hands overlay: only ep28 changes (the hand now releases when the pot is set down). Same checks as r1. Adds new paths only."""
from pathlib import Path
import json, shutil, hashlib, datetime
import numpy as np
W=Path(__file__).resolve().parent; SRC=W/'overlay_r1_src'; S=W/'overlay_r2_stage'; V3=W/'stage/scenes'; P28=SRC/'ep28pot_parent'
REPO='rose-tea-cafe/v2d-t3-recon'; PREFIX='experiments/wentacc/scored20_lead_v3_hands_r1_20261104'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest(); read=lambda p:json.loads(Path(p).read_text())
def put(p,x): Path(p).parent.mkdir(parents=True,exist_ok=True); Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
LOCAL=['/root/v2d/lead_v3/handspin28oldr2','/root/v2d/lead_v3/handspin28r2','/root/v2d/lead_winners_v8','/root/v2d/lead_v3/handspin','/root/v2d/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261103/lead_hand_reattach_v1','/root/v2d','/Volumes/E/Code/V2D']
def scrub(t):
    for r in LOCAL: t=t.replace(r,'<local>')
    return t
if S.exists(): shutil.rmtree(S)
ITEMS=[('r2new','episode_000028_r2',P28/'scenes/episode_000028','新锅，右手在锅放下时松开'),('r2old','episode_000028_oldpot_r2',V3/'episode_000028','旧锅，右手在锅放下时松开')]
for src,name,parent,why in ITEMS:
    sb,ss=SRC/src/'bundles/ep28_hold',SRC/src/'scenes/ep28_hold'; pb=parent/'bundle'
    a=np.load(sb/'result.npz',allow_pickle=False); b=np.load(pb/'result.npz',allow_pickle=False)
    for k in ['object_to_world_transform','object_1_to_world_transform','camera_to_world_transform']: assert np.allclose(a[k],b[k],atol=1e-6),(name,k)
    for m in [f for f in pb.iterdir() if f.suffix in ('.obj','.mtl','.png')]: assert sha(sb/m.name)==sha(m),(name,m.name)
    d=S/'scenes'/name; shutil.copytree(sb,d/'bundle',symlinks=False); shutil.copytree(ss,d/'scene',symlinks=False,ignore=shutil.ignore_patterns('*.generated_by_loader','relocate.log','.relocated','*.orig'))
    for f in list(d.rglob('*.json'))+list(d.rglob('*.log')): f.write_text(scrub(f.read_text()))
    shutil.copytree(parent/'support_v1',d/'support_v1')
    att=read(sb/'ATTACH_HOLD.json'); chk=read(ss/'OUTPUT_CHECK.json'); ct=read(ss/'HAND_CONTACT.json')
    put(d/'PROVENANCE.json',json.loads(scrub(json.dumps(dict(episode=28,change=why,supersedes=name.replace('_r2',''),objects_unchanged=True,holds=att['holds_final'],gaps_filled=att['gaps_filled'],hand_moved_cm=att['hand_moved_cm'],scene_check={k:chk.get(k) for k in ('task_error_cm','penetration_frames','frame0_seating_cm')},robot_contact=ct.get('objects'),policy_tested=False),ensure_ascii=False))))
    print('OK',name,att['holds_final'])
(S/'tools').mkdir(); shutil.copy2(W/'attach_tools/attach_hold.py',S/'tools/attach_hold_r2.py')
files=[p for p in sorted(S.rglob('*')) if p.is_file()]
leak=[str(p) for p in files if p.suffix in ('.json','.md','.csv','.log') and any(x in p.read_text(errors='ignore') for x in ['/Volumes/','/root/v2d','185.216.','223.109.'])]; assert not leak,leak[:5]
put(W/'UPLOAD_DRY_RUN_overlay_r2.json',dict(state='DRY_RUN_READY',repo=REPO,prefix=PREFIX,stage=str(S),files=[dict(relative=str(p.relative_to(S)),sha256=sha(p),bytes=p.stat().st_size) for p in files]))
print('files',len(files),'bytes',sum(p.stat().st_size for p in files))
