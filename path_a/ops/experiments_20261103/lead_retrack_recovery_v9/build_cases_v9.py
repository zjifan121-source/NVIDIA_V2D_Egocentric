"""Batch 9 (2026-11-04): late-anchor tracking for ep25 wooden pieces (own mesh, pose provided at the LAST frame with measured depth,
backward tracking) with plain and continuity recovery; nothing else was left untried for ep25 (no donor meshes exist)."""
from pathlib import Path
import json, hashlib, shutil, os, sys, subprocess, datetime, tarfile
import numpy as np
ROOT=Path('/Volumes/E/Code/V2D'); OPS=ROOT/'NVIDIA_V2D_Egocentric/path_a/ops'
X=OPS/'experiments_20261103/lead_retrack_recovery_v9'
B=OPS/'experiments_20261030/scored20_dual_anchor_v1'; REL=ROOT/'work/scored11-fulltrack-20261103/release'; FT=ROOT/'work/scored11-fulltrack-20261103'
SCRIPTS=ROOT/'work/scored20-pipeline-release-20261102/baseline/reconstruction/scripts'
sys.path.insert(0,str(SCRIPTS/'scored20_pipeline')); import inputs
sys.path.insert(0,str(X)); import run_recovery_v5 as run
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest(); read=lambda p:json.loads(Path(p).read_text())
def put(p,x): Path(p).parent.mkdir(parents=True,exist_ok=True); Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
inv={int(r['episode']):r['objects'] for r in read(REL/'inventory.json')}
KEN_MERGED=OPS/'experiments_20261029/scored_scene_consistency_v1/inputs/night_2026-10-24/scenes/episode_000033/ken_c_scored_attach/bundle_merged'
REP=OPS/'experiments_20261103/lead_fb11_repaired'
def rel(ep):
    r=REP/f'scenes/episode_{ep:06d}/bundle'  # contract-valid copies for the 7 fb11 bundles with missing materials
    return r if r.exists() else REL/f'scenes/episode_{ep:06d}/bundle'
# (ep, obj, mesh bundle, mesh slot, init: 'provided'|'register', init bundle, variant, label, priority)
# ep36 spoon mesh proved a better asset (ep33 +0.20, ep05 +0.046): try it on the remaining spoon episodes; ep37 donor for ep32
T=[(25,'wooden_piece_1',rel(25),0,'provided_last',None,'recovery_cont','last',1),(25,'wooden_piece_2',rel(25),1,'provided_last',None,'recovery_cont','last',2),
   (25,'wooden_piece_1',rel(25),0,'provided_last',None,'recovery','last',3),(25,'wooden_piece_2',rel(25),1,'provided_last',None,'recovery','last',4)]
template=read(FT/'ep33/protocol.json'); KEEP=['seed','register_iteration','track_iteration','mask_depth','scale_search','EKF','referee','GPU_window_seconds','expected_source_sha256','expected_backend_runtime']
runner_sha=sha(X/'run_recovery_v5.py'); batch=[]; skipped=[]; MIN_PX=1000
def build(ep,obj,mesh_bundle,mesh_slot,init,init_bundle,variant,label,prio):
    objects=inv[ep]; slot=objects.index(obj)
    origin=read(B/'cases'/f'ep{ep:02d}_{obj}'/'INPUT_MANIFEST.json'); cache=Path(origin['payload_local']); frames=origin['frames']; n=len(frames)
    Kd=read(cache/'K.json'); anchor=None
    for r in (frames[::-1] if init=='provided_last' else frames):
        _,d,_=run.decode_frame(cache,r,Kd)
        if int((d>0).sum())>=MIN_PX: anchor=r['frame_id']; break
    if anchor is None: raise ValueError('no frame with measured object depth')
    mesh_src=Path(mesh_bundle)/inputs.mesh_name(mesh_slot)
    case=X/'cases'/f'ep{ep:02d}_{obj}__{variant}_{label}'
    if case.exists(): shutil.rmtree(case)
    payload=case/'payload'; asset=payload/'asset'; asset.mkdir(parents=True)
    broot=mesh_src.resolve().parent
    for f in inputs.dependencies(mesh_src):
        t=asset/'mesh.obj' if f==mesh_src.resolve() else asset/f.relative_to(broot); t.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(f,t)
    files={('asset/'+f.relative_to(asset).as_posix()):sha(f) for f in sorted(asset.rglob('*')) if f.is_file()}
    shutil.copy2(cache/'K.json',payload/'K.json'); files['K.json']=sha(payload/'K.json')
    for r in frames:
        for k in ['rgb','depth','mask']:
            rp=r[k]; t=payload/rp; t.parent.mkdir(parents=True,exist_ok=True); os.link(cache/rp,t); files[rp]=origin['files'][rp]
    provided=None
    if init in ('provided','provided_last'):
        src=Path(init_bundle) if init_bundle else rel(ep); z=np.load(src/'result.npz',allow_pickle=False); pre='object' if slot==0 else f'object_{slot}'
        pose=np.asarray(z[pre+'_to_camera_transform'][anchor],float) if pre+'_to_camera_transform' in z.files else np.linalg.inv(np.asarray(z['camera_to_world_transform'][anchor],float))@np.asarray(z[pre+'_to_world_transform'][anchor],float)
        run.validate_transform(pose); np.save(case/'initial_pose.npy',pose)
        provided=dict(file='initial_pose.npy',sha256=sha(case/'initial_pose.npy'),mesh_sha256=files['asset/mesh.obj'],source_frame=anchor)
    manifest=dict(episode=ep,object=obj,object_slot=slot,frame_count=n,anchor_frame=anchor,metric_mesh_reviewed=True,anchor_selection_reviewed=True,
        review_scope=('Donor metric mesh of the same physical object class from another Scored episode; rank-0 registration at the anchor' if init=='register' else 'Current mesh and its pose at the LAST frame with measured depth (backward tracking); numerical validity only'),
        mesh_file='asset/mesh.obj',K_file='K.json',payload_root='payload',frames=[{k:r[k] for k in ['frame_id','rgb','depth','mask']} for r in frames],files=files)
    put(case/'manifest.json',manifest)
    protocol={k:template[k] for k in KEEP}; protocol.update(variant=variant,**run.VARIANTS[variant])
    protocol.update(episode=ep,object=obj,frame_count=n,anchor_frame=anchor,runner_sha256=runner_sha,input_manifest_sha256=sha(case/'manifest.json'),
        provenance=dict(mesh_source=str(mesh_src),mesh_source_sha256=sha(mesh_src),init=init,init_bundle=str(init_bundle or rel(ep)),observations=str(cache)),
        quality_veto_disabled_by_user=True,GT_use='none',purpose='batch 9: late-anchor backward tracking for ep25 wooden pieces (plain and continuity recovery)')
    if provided: protocol['provided_initial_pose']=provided
    put(case/'protocol.json',protocol)
    run.load_package(case/'manifest.json',case/'protocol.json',payload)
    batch.append(dict(case=case.name,episode=ep,object=obj,slot=slot,variant=variant,label=label,init=init,anchor_frame=anchor,frames=n,priority=prio,enabled=True,
        timeout_seconds=900,expected_seconds=300,mesh_source=str(mesh_src),protocol_sha256=sha(case/'protocol.json'),manifest_sha256=sha(case/'manifest.json'),mesh_sha256=files['asset/mesh.obj']))
    print('BUILT',case.name,n,'anchor',anchor,flush=True)
for t in T:
    try: build(*t)
    except Exception as e: skipped.append(dict(target=[str(x) for x in t],reason=repr(e)[:300])); print('SKIPPED',t[:2],t[6],t[7],repr(e)[:200],flush=True)
batch.sort(key=lambda c:c['priority'])
put(X/'FROZEN_BATCH.json',dict(state='FROZEN_AWAITING_GPU',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),runner_sha256=runner_sha,driver_sha256=sha(X/'driver.py'),
    tracker_sha256=sha(X/'foundation_pose_tracker.py'),contact_sha256=sha(X/'contact.py'),total_budget_seconds=3600,
    selection='pre-registered: strict full-frame proxy gain + symmetry-aware motion sanity (cups: axis tilt step; others: full rotation; max step <= max(current,30deg), trans p95 <= 1.5x current) + zero invalid frames; donor-mesh candidates swap mesh and poses together; no GT',cases=batch,skipped=skipped))
val={}
for c in batch:
    case=X/'cases'/c['case']; out=case/'VALIDATION.json'
    if out.exists(): out.unlink()
    r=subprocess.run([sys.executable,str(X/'run_recovery_v5.py'),'--manifest',str(case/'manifest.json'),'--protocol',str(case/'protocol.json'),'--payload',str(case/'payload'),'--validate-only','--validation-output',str(out)],capture_output=True,text=True)
    val[c['case']]=dict(exit=r.returncode,state=(read(out)['state'] if out.exists() else None),stderr=r.stderr[-300:]); print('VALIDATE',c['case'],val[c['case']]['exit'],val[c['case']]['state'],flush=True)
put(X/'VALIDATION_SUMMARY.json',val)
members=[]
for p in sorted(X.rglob('*')):
    if not p.is_file() or p.name in ('deploy.tar.gz','PACKAGE_SHA256.json','build_cases_v9.py','build_cases_v9.log') or p.suffix=='.pyc': continue
    rp=p.relative_to(X)
    if len(rp.parts)>=4 and rp.parts[0]=='cases' and rp.parts[2]=='payload' and rp.parts[3] in ('rgb','depth','masks'): continue
    if rp.parts[0]=='run_v1': continue
    members.append(p)
put(X/'PACKAGE_SHA256.json',{str(p.relative_to(X)):sha(p) for p in members})
pkg=X/'deploy.tar.gz'; pkg.unlink(missing_ok=True)
with tarfile.open(pkg,'w:gz') as t:
    for p in members+[X/'PACKAGE_SHA256.json']: t.add(p,arcname=str(p.relative_to(X)))
print('PACKAGE',pkg.stat().st_size,len(members),flush=True)
