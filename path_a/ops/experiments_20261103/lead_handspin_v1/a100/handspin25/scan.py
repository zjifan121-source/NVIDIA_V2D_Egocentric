import numpy as np, pyarrow.parquet as pq, trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation
S=Path("/root/v2d/lead_v3/handspin/scenes/ep25_hold")
p=next(S.glob("assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=sharpa_wave/*.parquet")); t=pq.read_table(p)
cell=lambda k:np.asarray(t[k][0].as_py())
names=t["object_body_names"][0].as_py(); print(names)
pos,q=cell("object_body_position"),cell("object_body_wxyz")
for obj in ["wooden_piece_2","wooden_piece_1"]:
    slot=next(i for i,n in enumerate(names) if n.endswith("_"+obj)); R=Rotation.from_quat(q[:,slot][:,[1,2,3,0]]).as_matrix()
    mesh=trimesh.load(S/f"score_input/meshes/episode_000025/{obj}.obj",force="mesh",process=False); v=np.asarray(mesh.vertices)
    c=v.mean(0); _,sv,ax=np.linalg.svd(v-c,full_matrices=False); co=(v-c)@ax.T; lo,hi=co.min(0),co.max(0)
    print(obj,"pca extents cm",np.round((hi-lo)*100,1).tolist(),"thin check",round(float((hi[2]-lo[2])/(hi[1]-lo[1])),2),"z range",round(float(pos[:,slot,2].min()),3),round(float(pos[:,slot,2].max()),3))
    for side in ["right","left"]:
        fn=t[f"{side}_robot_frame_names"][0].as_py(); tips=[fn.index(f"{side}_{d}_tip_site") for d in ("thumb","index")]; F=cell(f"robot_{side}_frames")
        rows=[]
        for a in range(150,333,10):
            lt=(F[a,tips,:3]-pos[a,slot])@R[a]; ax_=lt[0]-lt[1]; al=abs(ax_@ax[2])/np.linalg.norm(ax_)
            mid=lt.mean(0); tg=(mid-c)@ax.T; tg[:2]=np.clip(tg[:2],lo[:2]+.015,hi[:2]-.015); tg[2]=(lo[2]+hi[2])/2; d=c+tg@ax-mid
            rows.append((a,round(float(al),2),round(float(np.linalg.norm(d))*100,1),round(float(np.linalg.norm(ax_))*100,1)))
        print(" ",side,"(frame, pinch-axis alignment, move cm, tip gap cm):",rows)
