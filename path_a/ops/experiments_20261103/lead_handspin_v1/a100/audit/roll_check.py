import glob, hashlib, json, os
import numpy as np, pyarrow.parquet as pq, trimesh
from scipy.spatial.transform import Rotation as Rt
A="/root/v2d/lead_v3/handspin33com/scenes/ep33_rawspin"; B="/root/v2d/lead_v3/handspinroll/scenes/ep33_rawspinroll"
P="assets/human_motion_data/ego_recon/processed"
fq=glob.glob(f"{B}/{P}/sequence_id=*/robot_name=*/*.parquet"); print("parquets:",len(fq),"seq",fq[0].split("sequence_id=")[1].split("/")[0])
ta=pq.read_table(glob.glob(f"{A}/{P}/sequence_id=*/robot_name=*/*.parquet")[0]); tb=pq.read_table(fq[0])
cols=[c for c in ta.column_names if c in tb.column_names]
num=[]
for c in cols:
    x=ta.column(c)[0].as_py(); y=tb.column(c)[0].as_py()
    if x==y: continue
    try:
        X=np.array(x,float); Y=np.array(y,float); num.append(f"{c}: max {np.abs(X-Y).max():.4g}" if X.shape==Y.shape else f"{c}: SHAPE {X.shape} {Y.shape}")
    except Exception: num.append(f"{c}: strings")
print("columns differing:",len(num)); print("\n".join("  "+n for n in num[:34]))
us=tb.column('object_urdf_paths')[0].as_py()+tb.column('object_mesh_paths')[0].as_py()
print("paths under own scene:",all(p.startswith(B) for p in us),"exist:",all(os.path.exists(p) for p in us),"score meshes:",len(glob.glob(B+"/score_input/meshes/episode_*/*.obj")),"apple:",len(glob.glob(B+"/**/._*",recursive=True)))
for u in sorted(glob.glob(f"{B}/{P}/*_rigid.urdf")):
    n=os.path.basename(u); print(n,[l.strip() for l in open(u) if '<origin' in l][:2])
print("files in", os.listdir(B))
pa=np.array(ta.column('object_body_position')[0].as_py(),float); pb=np.array(tb.column('object_body_position')[0].as_py(),float)
qa=np.array(ta.column('object_body_wxyz')[0].as_py(),float); qb=np.array(tb.column('object_body_wxyz')[0].as_py(),float)
print("object pos max diff per body cm:",(np.abs(pa-pb).max(axis=(0,2))*100).round(3).tolist())
names=tb.column('object_body_names')[0].as_py()
for i,n in enumerate(names):
    Ra=Rt.from_quat(qa[:,i][:,[1,2,3,0]]); Rb=Rt.from_quat(qb[:,i][:,[1,2,3,0]])
    d=np.degrees((Ra.inv()*Rb).magnitude())
    print(f"{n}: rotation change deg median {np.median(d):.1f} p95 {np.percentile(d,95):.1f} max {d.max():.1f}")
    obj=glob.glob(f"{B}/{P}/*{n.split('rawspinroll_')[-1]}.obj")[0]
    m=trimesh.load(obj,force="mesh",process=False); v=np.asarray(m.vertices); c=v.mean(0); w,U=np.linalg.eigh(np.cov((v-c).T)); ax=U[:,2]
    wa=Ra.apply(ax); wb=Rb.apply(ax); tilt=np.degrees(np.arccos(np.clip(np.abs((wa*wb).sum(1)),-1,1)))
    print(f"   long-axis direction change deg: median {np.median(tilt):.2f} max {tilt.max():.2f}; mesh extent along axis cm {np.ptp((v-c)@ax)*100:.1f}")
    rv=(Ra.inv()*Rb).as_rotvec(); roll=np.degrees(rv@ax)
    print(f"   rotation about long axis deg: min {roll.min():.1f} max {roll.max():.1f}")
