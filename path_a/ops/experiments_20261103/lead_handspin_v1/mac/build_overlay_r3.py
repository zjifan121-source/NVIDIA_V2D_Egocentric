"""Revision r3 of the hands overlay. Adds new paths only: COM-fixed ep28 r2 scenes, ep33 and ep32 spinroll with COM, and the spin-only ep33 with COM as the control.
Same checks as r1 and r2. Nothing here has been through a policy yet, and every PROVENANCE.json says so."""
from pathlib import Path
import json, shutil, hashlib, re
import numpy as np
W=Path(__file__).resolve().parent; R3=W/'overlay_r3_src'; SRC=W/'overlay_r1_src'; S=W/'overlay_r3_stage'; V3=W/'stage/scenes'; P28=SRC/'ep28pot_parent'
REPO='rose-tea-cafe/v2d-t3-recon'; PREFIX='experiments/wentacc/scored20_lead_v3_hands_r1_20261104'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest(); read=lambda p:json.loads(Path(p).read_text())
def put(p,x): Path(p).parent.mkdir(parents=True,exist_ok=True); Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
m=re.search(r"^LOCAL=(\[.*\])$",(W/'build_overlay_r2.py').read_text(),re.M); LOCAL=eval(m.group(1))+['/root/v2d/lead_v3/handspinroll','/root/v2d/lead_v3/handspin33com','/root/v2d/lead_v3/handspin28r2com','/root/v2d/lead_v3/handspin28oldr2com','/root/v2d/lead_v3','/root/lead_current','/root/v2d']
LOCAL=sorted(set(LOCAL),key=len,reverse=True)
def scrub(t):
    for r in LOCAL: t=t.replace(r,'<local>')
    return re.sub(r'/root/[^\s"\',\]]+','<local>',t)
SKIP=shutil.ignore_patterns('*.generated_by_loader','relocate.log','.relocated','*.orig','com_backup','._*','__pycache__')
COM_RULE='URDF inertial origin moved to the area-weighted mesh shell centroid; mass, inertia, mesh, collision and trajectories unchanged'
SPIN_RULE='cup: axial spin replaced by the holding wrist azimuth, constant outside holds, frame 0 unchanged'
ROLL_RULE='spoon: roll about the long axis frozen at the frame-0 roll through the hold, exact at release, frame 0 unchanged; spoon hold window taken from the resting track'
ITEMS=[
 dict(name='episode_000028_r3',ep=28,role='candidate, new pot',parent=P28/'scenes/episode_000028',bundle=SRC/'r2new/bundles/ep28_hold',scene=R3/'handspin28r2com/scenes/ep28_hold',was=SRC/'r2new/scenes/ep28_hold',attach='ATTACH_HOLD.json',supersedes='episode_000028_r2',change='r2 new pot plus COM fix',rules=[COM_RULE]),
 dict(name='episode_000028_oldpot_r3',ep=28,role='candidate, old pot',parent=V3/'episode_000028',bundle=SRC/'r2old/bundles/ep28_hold',scene=R3/'handspin28oldr2com/scenes/ep28_hold',was=SRC/'r2old/scenes/ep28_hold',attach='ATTACH_HOLD.json',supersedes='episode_000028_oldpot_r2',change='r2 old pot plus COM fix',rules=[COM_RULE]),
 dict(name='episode_000033_r3_spinroll',ep=33,role='primary',parent=V3/'episode_000033',bundle=R3/'handspinroll/bundles/ep33_rawspinroll',scene=R3/'handspinroll/scenes/ep33_rawspinroll',attach='ATTACH_RESPIN.json',supersedes='episode_000033',change='cup respin, spoon roll frozen, spoon COM',rules=[SPIN_RULE,ROLL_RULE,COM_RULE]),
 dict(name='episode_000033_r3_spin',ep=33,role='control, spoon roll left as tracked',parent=V3/'episode_000033',bundle=R3/'handspin/bundles/ep33_rawspin',scene=R3/'handspin33com/scenes/ep33_rawspin',attach='ATTACH_RESPIN.json',supersedes='episode_000033',change='cup respin, spoon COM',rules=[SPIN_RULE,COM_RULE]),
 dict(name='episode_000032_r3_spinroll',ep=32,role='primary',parent=V3/'episode_000032',bundle=R3/'handspinroll/bundles/ep32_rawspinroll',scene=R3/'handspinroll/scenes/ep32_rawspinroll',attach='ATTACH_RESPIN.json',supersedes='episode_000032',change='cup respin, spoon roll frozen, spoon COM, spoon hold window fixed',rules=[SPIN_RULE,ROLL_RULE,COM_RULE]),
]
if S.exists(): shutil.rmtree(S)
for it in ITEMS:
    name,sb,ss,parent=it['name'],it['bundle'],it['scene'],it['parent']; pb=parent/'bundle'; d=S/'scenes'/name
    a=np.load(sb/'result.npz',allow_pickle=False); b=np.load(pb/'result.npz',allow_pickle=False)
    assert np.allclose(a['camera_to_world_transform'],b['camera_to_world_transform'],atol=1e-6),name
    for k in ['object_to_world_transform','object_1_to_world_transform']:
        if it['ep']==28: assert np.allclose(a[k],b[k],atol=1e-6),(name,k)
        else: assert np.allclose(a[k][0],b[k][0],atol=1e-6),(name,k,'frame 0')
    for f in [x for x in pb.iterdir() if x.suffix in ('.obj','.mtl','.png')]: assert sha(sb/f.name)==sha(f),(name,f.name)
    com=read(ss/'SCENE_COM_FIX.json'); patched=[o['body'] for o in com['objects'] if o.get('patched')]; assert patched,name
    if 'was' in it:
        for u in ss.glob('assets/human_motion_data/ego_recon/processed/*_rigid.urdf'):
            x,y=u.read_text(),(it['was']/u.relative_to(ss)).read_text(); strip=lambda t:re.sub(r'<inertial>\s*<origin xyz="[^"]*"','',re.sub(r'/[^"]*/assets/','assets/',t)); assert strip(x)==strip(y),(name,u.name)
    shutil.copytree(sb,d/'bundle',symlinks=False,ignore=SKIP); shutil.copytree(ss,d/'scene',symlinks=False,ignore=SKIP); shutil.copytree(parent/'support_v1',d/'support_v1',ignore=SKIP)
    for f in list(d.rglob('*.json'))+list(d.rglob('*.log')): f.write_text(scrub(f.read_text()))
    att=read(sb/it['attach']); chk=read(ss/'OUTPUT_CHECK.json'); ct=read(ss/'HAND_CONTACT.json')
    rec=dict(episode=it['ep'],revision='r3',role=it['role'],change=it['change'],supersedes=it['supersedes'],rules=it['rules'],meshes_unchanged=True,policy_tested=False,com_fix=[dict(body=o['body'],offset_cm=o['offset_cm']) for o in com['objects'] if o.get('patched')],scene_check={k:chk.get(k) for k in ('task_error_cm','penetration_frames','frame0_seating_cm')},robot_contact=ct.get('objects'))
    if it['attach']=='ATTACH_HOLD.json': rec.update(holds=att['holds_final'],gaps_filled=att['gaps_filled'],hand_moved_cm=att['hand_moved_cm'])
    else:
        rec.update(attached=att['attached'],respin={k:{x:v[x] for x in v if x not in ('cov_eigs','axis','center_cm')} for k,v in att['respin'].items()})
    if it['ep']==32:
        z=a['object_1_to_world_transform'][:,2,3].astype(float); rest=float(np.median(z[1:11])); up=np.flatnonzero(z[1:]>z[0]+0.02)
        rec['defects_logged']=[dict(what='frame 0 sits below the resting track, so the spoon read as lifted from frame 1 and the right wrist was dragged 21 cm over frames 1-35',frame0_z_m=round(float(z[0]),4),resting_z_m=round(rest,4),offset_cm=round((rest-float(z[0]))*100,2),hold_before=[1,259],hold_after=att['attached']['object_1'][0][1:3],first_frame_read_as_lifted=int(up[0])+1)]
    put(d/'PROVENANCE.json',json.loads(scrub(json.dumps(rec,ensure_ascii=False))))
    print('OK',name,patched)
(S/'tools').mkdir(); shutil.copy2(R3/'handspinroll/attach_respinroll.py',S/'tools/attach_respinroll_r3.py'); shutil.copy2(R3/'handspinroll/com_fix.py',S/'tools/com_fix_r3.py')
shutil.copy2(W/'R3_NOTES.md',S/'R3_NOTES.md')
files=[p for p in sorted(S.rglob('*')) if p.is_file() and not p.name.startswith('._')]
stray=[str(p) for p in S.rglob('._*')]; print('apple double files skipped',len(stray))
leak=[str(p) for p in files if p.suffix in ('.json','.md','.csv','.log') and any(x in p.read_text(errors='ignore') for x in ['/Volumes/','/root/','185.216.','223.109.'])]; assert not leak,leak[:5]
bad=[str(p) for p in files if p.suffix in ('.orig',) or p.name=='.relocated']; assert not bad,bad[:5]
put(W/'UPLOAD_DRY_RUN_overlay_r3.json',dict(state='DRY_RUN_READY',repo=REPO,prefix=PREFIX,stage=str(S),files=[dict(relative=str(p.relative_to(S)),sha256=sha(p),bytes=p.stat().st_size) for p in files]))
print('files',len(files),'bytes',sum(p.stat().st_size for p in files))
