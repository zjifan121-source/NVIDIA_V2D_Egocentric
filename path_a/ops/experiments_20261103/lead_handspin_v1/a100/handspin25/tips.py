import numpy as np, pyarrow.parquet as pq, trimesh
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
S=Path("/root/v2d/lead_v3/handspin25/scenes/ep25_hold")
p=next(S.glob("assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=sharpa_wave/*.parquet")); t=pq.read_table(p)
c=lambda k:np.asarray(t[k][0].as_py()); names=t["object_body_names"][0].as_py(); pos,q=c("object_body_position"),c("object_body_wxyz")
slot=next(i for i,n in enumerate(names) if n.endswith("_wooden_piece_2")); R=Rotation.from_quat(q[:,slot][:,[1,2,3,0]]).as_matrix()
m=trimesh.load(next(S.glob("assets/human_motion_data/ego_recon/processed/*wooden_piece_2.obj")),force="mesh",process=False); pts,_=trimesh.sample.sample_surface(m,40000,seed=0); tr=cKDTree(pts)
fn=t["right_robot_frame_names"][0].as_py(); F=c("robot_right_frames"); idx=[fn.index(f"right_{d}_tip_site") for d in ("thumb","index","middle")]
print("frame: thumb/index/middle tip to board 2 surface (cm), thumb-index gap, signed side")
for a in list(range(150,333,20)):
    lt=(F[a,idx,:3]-pos[a,slot])@R[a]; d=tr.query(lt)[0]*100; print(a,np.round(d,1).tolist(),round(float(np.linalg.norm(lt[0]-lt[1])*100),1))
