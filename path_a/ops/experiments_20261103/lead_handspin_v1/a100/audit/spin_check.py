import glob
import numpy as np, pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as Rt
def load(s):
    f=glob.glob(s+"/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=*/*.parquet")[0]
    t=pq.read_table(f,columns=['object_body_wxyz','object_body_names'])
    return np.array(t.column(0)[0].as_py(),float),t.column(1)[0].as_py()
for name,s in (("plain","/root/v2d/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261103/lead_hand_reattach_v1/scenes/ep33_rawexcl"),("r1 spin","/root/lead_current/experiments/wentacc/scored20_lead_v3_hands_r1_20261104/scenes/episode_000033/scene")):
    q,names=load(s); w=q[:,0]  # blue cup, wxyz
    R=Rt.from_quat(w[:,[1,2,3,0]])
    rel=R[:-1].inv()*R[1:]; rv=rel.as_rotvec()
    ang=np.degrees(np.linalg.norm(rv,axis=1))
    # dominant body-frame axis of the frame-to-frame rotation = the cup's spin axis
    u,sv,vt=np.linalg.svd(rv,full_matrices=False); ax=vt[0]; signed=np.degrees(rv@ax)
    print(f"{name}: {names[0]} frames={len(w)} summed |step rotation|={ang.sum():.0f} deg, net signed about dominant axis={signed.sum():.0f} deg, cumulative abs about axis={np.abs(signed).sum():.0f} deg, max step={ang.max():.1f} deg")
