import sys,json,numpy as np
from scipy.spatial.transform import Rotation as R
H="/root/v2d/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261103/lead_hand_reattach_v1"
def steps(M): return np.degrees(R.from_matrix(np.einsum("nji,njk->nik",M[:-1],M[1:])).magnitude())
for ep in (33,32):
    raw=np.load(f"{H}/scenes/ep{ep}_rawclean/bundle_cleaned/result.npz")
    Wr=raw["camera_to_world_transform"].astype(float)@raw["hand_right_wrist_to_camera_transform"].astype(float)
    for name,b in [("tracked","/root/v2d/lead_v3/handspin/bundles/ep%d_rawspin"%ep)]+[(r,f"/root/v2d/lead_v3/handspinroll/try_{r}/bundles/ep{ep}_rawspinroll") for r in ("position","orientation","frozen")]:
        z=np.load(b+"/result.npz"); W=z["camera_to_world_transform"].astype(float)@z["hand_right_wrist_to_camera_transform"].astype(float); T=z["object_1_to_world_transform"].astype(float)
        hold=json.load(open(b+"/ATTACH_RESPIN.json"))["attached"]["object_1"][0]; a,b_=hold[1],hold[2]
        d=np.linalg.norm(W[a:b_,:3,3]-Wr[a:b_,:3,3],axis=1)*100; ang=np.degrees(R.from_matrix(np.einsum("nji,njk->nik",W[a:b_,:3,:3],Wr[a:b_,:3,:3])).magnitude())
        print(ep,f"{name:12s} hold {a}-{b_} wrist moved median {np.median(d):.1f} p95 {np.percentile(d,95):.1f} max {d.max():.1f} cm | hand rot vs raw median {np.median(ang):.0f} p95 {np.percentile(ang,95):.0f} deg | spoon rot step p95 {np.percentile(steps(T[:,:3,:3]),95):.1f} max {steps(T[:,:3,:3]).max():.1f} | hand rot step p95 {np.percentile(steps(W[a:b_,:3,:3]),95):.1f} (raw {np.percentile(steps(Wr[a:b_,:3,:3]),95):.1f})")
