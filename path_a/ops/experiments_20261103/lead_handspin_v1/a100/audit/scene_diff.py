import sys, glob, json
import numpy as np, pyarrow.parquet as pq
def load(s):
    f=glob.glob(s+"/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=*/*.parquet")[0]
    t=pq.read_table(f); return {c:t.column(c)[0].as_py() for c in t.column_names}
SC="/root/v2d/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261103/lead_hand_reattach_v1/scenes"
pairs=[("ep33",f"{SC}/ep33_rawexcl","/root/v2d/lead_v3/stage/scenes/episode_000033/scene"),
       ("ep25",f"{SC}/ep25_rawexcl","/root/v2d/lead_v3/stage/scenes/episode_000025/scene"),
       ("ep30",f"{SC}/ep30_rawexcl","/root/v2d/lead_v3/stage/scenes/episode_000030/scene"),
       ("ep28new","/root/v2d/lead_winners_v8/mini_stage/hand_stage/scenes/ep28_rawexcl","/root/v2d/lead_v3/stage/scenes/episode_000028/scene")]
for ep,a,b in pairs:
    A,B=load(a),load(b); same=[]; diff=[]
    print("==",ep,"cols",len(A),len(B),"only-a",sorted(set(A)-set(B)),"only-b",sorted(set(B)-set(A)))
    for c in sorted(set(A)&set(B)):
        x,y=A[c],B[c]
        if x==y: same.append(c); continue
        try:
            X=np.asarray(x,dtype=float); Y=np.asarray(y,dtype=float)
            d=np.abs(X-Y); diff.append(f"{c}: shape{X.shape}/{Y.shape} maxabs={d.max():.4g} mean={d.mean():.4g}" if X.shape==Y.shape else f"{c}: SHAPE {X.shape}/{Y.shape}")
        except Exception:
            s=str(x)[:110]; u=str(y)[:110]
            if c.endswith('paths') or 'file' in c or 'id' in c: diff.append(f"{c}: strings differ ({s.split('/')[-1]} | {u.split('/')[-1]})")
            else: diff.append(f"{c}: {s} | {u}")
    print(" same",len(same)); print("\n".join("  "+d for d in diff))
