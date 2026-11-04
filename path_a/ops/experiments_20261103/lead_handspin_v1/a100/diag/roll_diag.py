import sys,numpy as np,trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation as R
H=Path("/root/v2d/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261103/lead_hand_reattach_v1")
for ep,slot in [(33,1),(32,1)]:
    raw=H/f"scenes/ep{ep}_rawclean/bundle_cleaned"
    z=np.load(raw/"result.npz"); print("EP",ep,[str(x) for x in z["object_names"]] if "object_names" in z.files else None)
    k=f"object_{slot}_to_world_transform"
    print(sorted(x for x in z.files if "to_world" in x), z[k].shape)
    T=z[k].astype(float); n=len(T)
    m=trimesh.load(raw/f"mesh_{slot}.obj",force="mesh",process=False); s=float(z[f"object_{slot}_scale"]) if f"object_{slot}_scale" in z.files else 1.0
    pts,_=trimesh.sample.sample_surface(m,20000,seed=0); pts=pts*s; c=pts.mean(0); w,U=np.linalg.eigh(np.cov((pts-c).T)); print("eigs cm2",np.round(w*1e4,2),"scale",s); a=U[:,2]
    print("extents along PCs cm",np.round((pts-c).dot(U).max(0)-(pts-c).dot(U).min(0),1)*100)
    ax=np.einsum("nij,j->ni",T[:,:3,:3],a); sw=np.degrees(np.arccos(np.clip(ax@ax[0],-1,1))); print("axis swing from f0 max",round(sw.max(),1),"at",int(sw.argmax()))
    st=np.degrees(np.array([R.from_matrix(T[i,:3,:3].T@T[i+1,:3,:3]).magnitude() for i in range(n-1)])); dp=np.linalg.norm(np.diff(T[:,:3,3],axis=0),axis=1)*100
    print("rot step p95",round(np.percentile(st,95),1),"max",round(st.max(),1),"at",int(st.argmax()),"; trans step p95 cm",round(np.percentile(dp,95),2),"max",round(dp.max(),2),"at",int(dp.argmax()))
    print("z range",np.round([T[:,2,3].min(),T[:,2,3].max()],3))
    for side in ("left","right"):
        W=z["camera_to_world_transform"].astype(float)@z[f"hand_{side}_wrist_to_camera_transform"].astype(float)
        w_=W[:,:3,3]; cw=np.einsum("nij,j->ni",T[:,:3,:3],c)+T[:,:3,3]; d=np.linalg.norm(w_-cw,axis=1); held=d<0.15
        if held.sum()<10: print(side,"far"); continue
        f=np.where(held)[0]; f0,f1=f[0],f[-1]+1
        Rw=T[f0:f1,:3,:3]; x=np.einsum("nji,nj->ni",Rw,w_[f0:f1]-T[f0:f1,:3,3])-c; along=x@a; perp=x-np.outer(along,a); r=np.linalg.norm(perp,axis=1)*100
        print(side,"f",f0,f1,"perp radius cm median",round(np.median(r),1),"p10",round(np.percentile(r,10),1),"along cm median",round(np.median(along)*100,1))
