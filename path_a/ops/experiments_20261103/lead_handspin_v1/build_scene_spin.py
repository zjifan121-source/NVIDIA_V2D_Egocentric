"""Build a training scene from a prepared bundle with the team loader (tools/track3/run_load_retarget.py), using
haisenk's recipe (2026-11-03): control = bundle as delivered (--no-rest-hold --no-hand-clean --no-object-clean --no-level);
rawhands = hand-only cleanup against OUR objects (--no-rest-hold --no-object-clean --no-level --attach-hands --smooth-hands 2).
This CPU environment (.venv-bundle-cpu: loader, MANO, retarget; no USD bindings) cannot run the loader's own support step
or its USD-based output check, so after load + retarget + URDF the SEATED support_v1 spec (lead_support_seating_v1/out)
is installed as the scene's support USD (text writer, same prim layout as the loader's) and an equivalent output check
(task error, link-object penetration, frame-0 seating, hand-object proximity on lifted frames) is written to
OUTPUT_CHECK.json. CPU only, no GT. Usage: build_scene.py <ep> <control|rawhands>"""
from pathlib import Path
import os
import sys, os, json, subprocess, time, shutil, importlib.util, hashlib, re
import numpy as np
SRC=Path(__file__).resolve().parent; O=Path(os.environ.get('LEAD_HAND_OUT',str(SRC))); import os
ROOT=Path(os.environ.get('V2D_ROOT','/Volumes/E/Code/V2D')); REPO=ROOT/'NVIDIA_V2D_Egocentric'
LOADER=Path(os.environ.get('LEAD_LOADER',str(REPO/'tools/track3/run_load_retarget.py'))); SEAT=Path(os.environ.get('LEAD_SEAT_ROOT',str(ROOT/'NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261103/lead_support_seating_v1/out')))
INSTALL=SRC/'install_support.py'
RUNTIME=ROOT/'NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261029/scored_scene_consistency_v1/runtime'  # v2d loader + robotic_grounding packages (same runtime as the consumer checks)
FPS=20.254119873046875; sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ep,mode=int(sys.argv[1]),sys.argv[2]; B=O/'bundles'/f'ep{ep:02d}_{"rawhands" if mode=="edmattach" else mode}'; OUT=O/'scenes'/f'ep{ep:02d}_{mode}'; SEQ=f'ep{ep}_lead_{mode}'
FLAGS={'control':['--no-rest-hold','--no-hand-clean','--no-object-clean','--no-level'],
       'rawhands':['--no-rest-hold','--no-object-clean','--no-level','--attach-hands','--smooth-hands','2'],
       'rawclean':['--no-rest-hold','--no-object-clean','--no-level','--smooth-hands','2'],          # cleaned + smoothed raw hands, NOT attached
       'rawexcl':['--no-rest-hold','--no-hand-clean','--no-object-clean','--no-level'],             # input already cleaned + exclusively attached (attach_exclusive.py)
       'kenclean':['--no-rest-hold','--no-object-clean','--no-level','--smooth-hands','2'],          # Ken-corrected raw hands (their bundle/ hand arrays), cleaned + smoothed, not attached
       'kenexcl':['--no-rest-hold','--no-hand-clean','--no-object-clean','--no-level'],'rawspin':['--no-rest-hold','--no-hand-clean','--no-object-clean','--no-level'],              # kenclean + exclusive attach
       'edmattach':['--no-rest-hold','--no-object-clean','--no-level','--attach-hands','--smooth-hands','2']}  # Edmond's de66fea loader (hold_owner: one object per hand) on the rawhands bundle
FL=FLAGS[mode]
if OUT.exists(): shutil.rmtree(OUT)
OUT.mkdir(parents=True); (O/'logs').mkdir(exist_ok=True)
env=dict(os.environ,PYTHONPATH=str(RUNTIME),V2D_EGO_FPS=repr(FPS),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
cmd=[sys.executable,str(LOADER),'--bundle',str(B),'--episode',str(ep),*FL,'--skip-video','--device','cpu','--fps',repr(FPS),'--sequence-id',SEQ,'--out',str(OUT)]
t0=time.monotonic(); log=O/'logs'/f'ep{ep:02d}_{mode}_pass1.log'
with log.open('w') as f: r=subprocess.run(cmd,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1500)
print('loader exit',r.returncode,round(time.monotonic()-t0,1),'s (exit 1 at the support step is expected here)',flush=True)
text=log.read_text(); m=re.search(r'== cleaned bundle: (\{.*\})',text); cleaning=eval(m.group(1),{'__builtins__':{}},{'nan':float('nan'),'inf':float('inf'),'true':True,'false':False,'None':None}) if m else None
proc=next((OUT/'assets/human_motion_data/ego_recon/processed'/f'sequence_id={SEQ}/robot_name=sharpa_wave').glob('*.parquet'),None)
assert proc is not None,'no processed parquet: '+text[-3000:]
assert 'support surface' in text,'pipeline stopped before the support step: '+text[-2000:]
# ---- install the seated support (common transform from the cleaned bundle the loader consumed; hand-only cleaning leaves object arrays unchanged)
spec=importlib.util.spec_from_file_location('install_support',INSTALL); inst=importlib.util.module_from_spec(spec); spec.loader.exec_module(inst)
import pyarrow.parquet as pq, trimesh
from scipy.spatial.transform import Rotation
cleaned=OUT/'bundle_cleaned' if (OUT/'bundle_cleaned/result.npz').exists() else B
table=pq.read_table(proc); record={k:v[0] for k,v in table.to_pydict().items()}
raw=np.load(cleaned/'result.npz',allow_pickle=False); G,pe,re_=inst.common_transform(raw,record)
seated=json.loads((SEAT/f'episode_{ep:06d}/support_v1/support_in_bundle_world.json').read_text())
STAGE_B=Path(os.environ.get('LEAD_BUNDLES_ROOT',str(ROOT/'work/lead-sprint-release-20261103/stage/scenes')))/f'episode_{ep:06d}/bundle'
for n,h in seated['bundle_sha256'].items():
    if n.endswith('.npz'):  # rawhands bundles differ from the staged bundle only in hand_* arrays: the seating (objects at frame 0) still applies
        a=np.load(B/n,allow_pickle=False); s=np.load(STAGE_B/n,allow_pickle=False)
        keys=[k for k in s.files if k.endswith('_to_world_transform') and not k.startswith('hand_')]  # object + camera world poses (the loader recomputes object_to_camera in float64 -> not byte-equal)
        assert sha(STAGE_B/n)==h and all(np.allclose(a[k][0],s[k][0],atol=1e-6) for k in keys),('seated spec objects differ from this bundle',n)
    else: assert sha(B/n)==h,('seated spec was made for another bundle',n)
usda=OUT/'assets/human_motion_data/ego_recon/reconstructed_stage'/f'{SEQ}_sharpa_wave_support.usda'; usda.parent.mkdir(parents=True,exist_ok=True)
if usda.exists(): usda.rename(usda.with_name(usda.name+'.generated_by_loader'))
cyl_sim=[]; prims=''
for i,c in enumerate(seated['cylinders']):
    ctr=(G@np.r_[c['center_bundle_world_m'],1.0])[:3]; name=f"{SEQ}_{c.get('kind','table')}_{i}"; cyl_sim.append((ctr,c['radius_m'],c['height_m'],c.get('kind','table')))
    prims+=f'''    def Cylinder "{name}" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {{
        uniform token axis = "Z"
        double height = {c['height_m']!r}
        double radius = {c['radius_m']!r}
        double3 xformOp:translate = ({float(ctr[0])!r}, {float(ctr[1])!r}, {float(ctr[2])!r})
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}

'''
usda.write_text('#usda 1.0\n(\n    defaultPrim = "support_surfaces"\n    metersPerUnit = 1\n    upAxis = "Z"\n)\n\ndef Xform "support_surfaces"\n{\n'+prims+'}\n')
(OUT/'SUPPORT_V1_SEATED_INSTALLED.json').write_text(json.dumps(dict(spec=str(SEAT/f'episode_{ep:06d}/support_v1/support_in_bundle_world.json'),spec_sha256=sha(SEAT/f'episode_{ep:06d}/support_v1/support_in_bundle_world.json'),world_to_consumer=G.tolist(),pose_error_m=pe,rot_error=re_,cylinders=len(seated['cylinders']),usda=str(usda),usda_sha256=sha(usda)),indent=1)+'\n')
# ---- output check (same quantities as the loader's check_output, without USD bindings) + hand-object proximity
INV={int(r['episode']):r['objects'] for r in json.loads((SRC/'inventory_v1.json').read_text())}
names=[str(n) for n in raw['object_names']] if 'object_names' in raw.files else INV[ep]
processed=proc.parents[2]; bodies=[f'{SEQ}_{n}' for n in names] if (processed/f'{SEQ}_{names[0]}.obj').exists() else [SEQ]  # single-object bundles without object_names are one body named after the sequence
P=np.asarray(record['object_body_position'],float); Q=np.asarray(record['object_body_wxyz'],float)
errors={side:np.asarray(record[f'robot_{side}_frame_task_errors'],float) for side in ('right','left')}
links=np.concatenate([np.asarray(record[f'robot_{side}_frames'],float)[:,2:,:3] for side in ('right','left')],axis=1)
depths=[]; seating=[]; pen0=[]
for b,body in enumerate(bodies):
    mesh=trimesh.load(processed/f'{body}.obj',force='mesh',process=False); rot=Rotation.from_quat(Q[:,b][:,[1,2,3,0]]).as_matrix()
    local=np.einsum('tji,tkj->tki',rot,links-P[:,b][:,None,:]); pts=local.reshape(-1,3)
    depths.append(np.concatenate([trimesh.proximity.signed_distance(mesh,pts[i:i+2000]) for i in range(0,len(pts),2000)]).reshape(local.shape[:2]).max(axis=1))
    w0=np.asarray(mesh.vertices)@rot[0].T+P[0,b]; centre=w0[:,:2].mean(0)
    under=[c[2]+h/2 for c,rr,h,k in cyl_sim if np.linalg.norm(centre-c[:2])<=rr]; seating.append(round((float(w0[:,2].min())-max(under))*100,2) if under else None)
    worst=0.0
    for c,rr,h,k in cyl_sim:
        ins=(np.linalg.norm(w0[:,:2]-c[:2],axis=1)<=rr)&(w0[:,2]<=c[2]+h/2)&(w0[:,2]>=c[2]-h/2)
        if ins.any(): worst=max(worst,float((c[2]+h/2-w0[ins,2]).max()*100))
    pen0.append(worst)
depth=np.max(depths,axis=0)
# hand-object proximity while an object is lifted (bundle world, cleaned bundle): wrist position vs object surface
def wrist_world(data,side):
    return data[f'hand_{side}_wrist_to_world_transform'].astype(float)[:,:3,3]
prox={}
for b,n in enumerate(names):
    key='object_to_world_transform' if b==0 else f'object_{b}_to_world_transform'; T=raw[key].astype(float)
    V=np.asarray(trimesh.load(B/('mesh.obj' if b==0 else f'mesh_{b}.obj'),force='mesh',process=False).vertices,float); V=V[::max(1,len(V)//600)]
    z=T[:,2,3]; lifted=z>z[:10].min()+0.02
    surf=np.einsum('tij,vj->tvi',T[:,:3,:3],V)+T[:,None,:3,3]
    for side in ('left','right'):
        d=np.linalg.norm(surf-wrist_world(raw,side)[:,None,:],axis=-1).min(axis=1)
        prox[f'{n}/{side}']=dict(lifted_frames=int(lifted.sum()),wrist_to_surface_cm_median_lifted=round(float(np.median(d[lifted])*100),2) if lifted.any() else None,
            within_5cm_lifted=round(float(np.mean(d[lifted]<0.05)),3) if lifted.any() else None,within_12cm_lifted=round(float(np.mean(d[lifted]<0.12)),3) if lifted.any() else None)
check=dict(episode=ep,mode=mode,sequence_id=SEQ,frames=int(len(P)),fps=float(record['fps']),loader_exit=r.returncode,loader_seconds=round(time.monotonic()-t0,1),cleaning=cleaning,
    task_error_cm={s:dict(mean=round(float(e.mean())*100,2),max=round(float(e.max())*100,2)) for s,e in errors.items()},
    penetration_frames=dict(any=round(float(np.mean(depth>0)),3),over_1cm=round(float(np.mean(depth>0.01)),3),over_2cm=round(float(np.mean(depth>0.02)),3)),
    support_shapes=len(cyl_sim),frame0_seating_cm=seating,frame0_support_penetration_cm=pen0,hand_object_proximity=prox,
    processed=str(proc),support_usda=str(usda),bundle_cleaned=str(cleaned),policy_tested=False)
(OUT/'OUTPUT_CHECK.json').write_text(json.dumps(check,indent=1)+'\n'); print(json.dumps({k:check[k] for k in ['cleaning','task_error_cm','penetration_frames','frame0_seating_cm','frame0_support_penetration_cm','hand_object_proximity']},indent=1),flush=True)
