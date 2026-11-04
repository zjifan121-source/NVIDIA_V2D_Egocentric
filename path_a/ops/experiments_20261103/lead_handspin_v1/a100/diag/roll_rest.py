import numpy as np,trimesh
from scipy.spatial.transform import Rotation as R
H="/root/v2d/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261103/lead_hand_reattach_v1"
for ep,(a0,b0) in {33:(57,296),32:(1,259)}.items():
    p=f"{H}/scenes/ep{ep}_rawclean/bundle_cleaned"; z=np.load(p+"/result.npz"); T=z["object_1_to_world_transform"].astype(float); n=len(T)
    m=trimesh.load(p+"/mesh_1.obj",force="mesh",process=False); pts,_=trimesh.sample.sample_surface(m,20000,seed=0); w,U=np.linalg.eigh(np.cov((pts-pts.mean(0)).T)); a=U[:,2]
    q=R.from_matrix(np.einsum("ji,njk->nik",T[0,:3,:3],T[:,:3,:3])).as_quat(); th=np.degrees(2*np.arctan2(q[:,:3]@a,q[:,3])); th=(th+180)%360-180
    print("ep",ep,"n",n,"tracked roll vs f0, every 12 frames:",[int(x) for x in th[::12]])
    Wr=z["camera_to_world_transform"].astype(float)@z["hand_right_wrist_to_camera_transform"].astype(float)
    print("  spoon z every 12:",[round(float(x),3) for x in T[::12,2,3]])
    c=pts.mean(0); cw=np.einsum("nij,j->ni",T[:,:3,:3],c)+T[:,:3,3]; print("  right wrist to spoon centroid cm every 12:",[int(x) for x in np.linalg.norm(Wr[:,:3,3]-cw,axis=1)[::12]*100])
    if b0<n:
        rest=th[b0+5:]; ang=np.radians(rest); print("  rest after hold: circular mean",round(float(np.degrees(np.arctan2(np.sin(ang).mean(),np.cos(ang).mean()))),1),"spread deg",round(float(np.degrees(np.sqrt(-2*np.log(np.hypot(np.sin(ang).mean(),np.cos(ang).mean()))))),1))
    pre=th[:max(a0-3,1)]; print("  pre hold spread",round(float(np.ptp(pre)),1))
