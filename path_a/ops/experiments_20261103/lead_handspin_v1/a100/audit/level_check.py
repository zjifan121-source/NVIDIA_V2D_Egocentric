import glob, hashlib, json, os
import numpy as np, pyarrow.parquet as pq
P="assets/human_motion_data/ego_recon/processed"
def pqs(s): return sorted(glob.glob(f"{s}/{P}/sequence_id=*/robot_name=*/*.parquet"))
def load(f):
    t=pq.read_table(f); return {c:t.column(c)[0].as_py() for c in t.column_names}
for ep,new,base in ((1,"/root/v2d/lead_v3/level/scenes/ep01_hold","/root/v2d/lead_v3/stage/scenes/episode_000001/scene"),(30,"/root/v2d/lead_v3/level/scenes/ep30_hold","/root/v2d/lead_v3/stage/scenes/episode_000030/scene")):
    fn=pqs(new); fb=pqs(base); print(f"== ep{ep} new parquets={len(fn)} base parquets={len(fb)}")
    A=load(fn[0]); B=load(fb[0])
    us=A['object_urdf_paths']+A['object_mesh_paths']
    print("  seq",fn[0].split('sequence_id=')[1].split('/')[0],"paths under own scene:",all(p.startswith(new) for p in us),"exist:",all(os.path.exists(p) for p in us),"score meshes:",len(glob.glob(new+"/score_input/meshes/episode_*/*.obj")),"apple:",len(glob.glob(new+"/**/._*",recursive=True)))
    diff=[]
    for c in sorted(set(A)&set(B)):
        x,y=A[c],B[c]
        if x==y: continue
        try:
            X=np.array(x,float); Y=np.array(y,float); diff.append(f"{c}: max {np.abs(X-Y).max():.4g}" if X.shape==Y.shape else f"{c}: SHAPE {X.shape} {Y.shape}")
        except Exception: diff.append(f"{c}: strings")
    print("  columns differing vs lead_v3:",len(diff)); print("   "+"\n   ".join(diff[:40]))
    pa=np.array(A['object_body_position'],float); pb=np.array(B['object_body_position'],float)
    if pa.shape==pb.shape: print("  object pos diff cm per body: max",(np.abs(pa-pb).max(axis=(0,2))*100).round(2).tolist(),"frame0",(np.abs(pa[0]-pb[0]).max(axis=1)*100).round(2).tolist(),"z shift median cm",((pa-pb)[:,:,2].mean(0)*100).round(2).tolist())
    for u in sorted(glob.glob(f"{new}/{P}/*_rigid.urdf")):
        print("  ",os.path.basename(u),[l.strip() for l in open(u) if '<origin' in l][:1])
