import glob, hashlib, json, os, re, sys
import numpy as np, pyarrow.parquet as pq, trimesh
A="/root/v2d/lead_v3/handspin28r2/scenes/ep28_hold"; B="/root/v2d/lead_v3/handspin28r2com/scenes/ep28_hold"
P="assets/human_motion_data/ego_recon/processed"
print("SCENE_COM_FIX:",open(B+"/SCENE_COM_FIX.json").read()[:900])
fa={os.path.relpath(f,A) for f in glob.glob(A+"/**/*",recursive=True) if os.path.isfile(f)}
fb={os.path.relpath(f,B) for f in glob.glob(B+"/**/*",recursive=True) if os.path.isfile(f)}
print("only in r2:",sorted(fa-fb)[:8]); print("only in com:",sorted(fb-fa)[:8])
sha=lambda p:hashlib.sha256(open(p,'rb').read()).hexdigest()
diff=[f for f in sorted(fa&fb) if sha(A+"/"+f)!=sha(B+"/"+f)]
print("files differing:",len(diff)); print("\n".join("  "+d for d in diff[:20]))
pa=glob.glob(f"{A}/{P}/sequence_id=*/robot_name=*/*.parquet"); pb=glob.glob(f"{B}/{P}/sequence_id=*/robot_name=*/*.parquet")
print("parquets",len(pa),len(pb))
ta=pq.read_table(pa[0]); tb=pq.read_table(pb[0])
cd=[c for c in ta.column_names if ta.column(c)[0].as_py()!=tb.column(c)[0].as_py()]
print("parquet columns differing:",cd)
print("paths exist:",all(os.path.exists(p) for c in ('object_mesh_paths','object_urdf_paths') for p in tb.column(c)[0].as_py()),"score meshes:",len(glob.glob(B+"/score_input/meshes/episode_*/*.obj")))
for u in sorted(glob.glob(f"{B}/{P}/*_rigid.urdf")):
    n=os.path.basename(u); a=open(f"{A}/{P}/{n}").read(); b=open(u).read()
    la=a.splitlines(); lb=b.splitlines()
    ch=[(x.strip(),y.strip()) for x,y in zip(la,lb) if x!=y]
    print(n,"lines",len(la),len(lb),"changed:",ch[:4])
    obj=u.replace("_rigid.urdf",".obj"); m=trimesh.load(obj,force="mesh",process=False)
    pts,_=trimesh.sample.sample_surface(m,40000,seed=0)
    vc=m.center_mass if m.is_volume else None
    print("   mesh bounds center",np.round(m.bounds.mean(0),4).tolist(),"surface centroid",np.round(pts.mean(0),4).tolist(),"volume com",None if vc is None else np.round(vc,4).tolist(),"|origin to centroid| cm",round(float(np.linalg.norm(pts.mean(0)))*100,2))
