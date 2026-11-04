"""Audit of exclusive attach: was each 'held' hand actually near the object before attach? Read-only, no GT.
For each rawexcl scene: compare rawclean (pre-attach) and rawexcl (post-attach) wrists against the object on frames
where the post-attach hand sits rigidly on that object (attach window)."""
import json, glob, os, sys
from pathlib import Path
import numpy as np, trimesh
H=Path('/root/v2d/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261103/lead_hand_reattach_v1')
def objs(d, bdir):
    names=[str(n) for n in d['object_names']] if 'object_names' in d else ['object']
    out={}
    for i,n in enumerate(names):
        pk='object_to_world_transform' if i==0 else f'object_{i}_to_world_transform'
        out[n]=d[pk].astype(float)
    return out
def mesh_for(b, i):
    p=b/('mesh.obj' if i==0 else f'mesh_{i}.obj')
    return trimesh.load(p,force='mesh',process=False)
rows=[]
for sc in sorted((H/'scenes').glob('ep*_rawexcl')):
    ep=sc.name[2:4]; b1=H/'bundles'/sc.name; b0=sc.parent/f'ep{ep}_rawclean'/'bundle_cleaned'
    if not (b1/'result.npz').exists() or not (b0/'result.npz').exists(): rows.append(dict(ep=ep,skip=True)); continue
    d1=dict(np.load(b1/'result.npz',allow_pickle=False)); d0=dict(np.load(b0/'result.npz',allow_pickle=False))
    O=objs(d1,b1)
    for side in ('left','right'):
        w1=d1[f'hand_{side}_wrist_to_world_transform'].astype(float); w0=d0[f'hand_{side}_wrist_to_world_transform'].astype(float)
        c2w1=d1['camera_to_world_transform'].astype(float); c2w0=d0['camera_to_world_transform'].astype(float)
        moved=np.linalg.norm(w1[:,:3,3]-w0[:,:3,3],axis=1)
        attached=moved>0.005
        for i,(n,T) in enumerate(O.items()):
            m=mesh_for(b1,i); s=float(np.asarray(d1.get('object_scale' if i==0 else f'object_{i}_scale',1.0)).reshape(-1)[0]); v=np.asarray(m.vertices)[::max(1,len(m.vertices)//1500)]*s
            pts=np.einsum('tij,nj->tni',T[:,:3,:3],v)+T[:,None,:3,3]
            d_pre=np.linalg.norm(pts-w0[:,None,:3,3],axis=2).min(1); d_post=np.linalg.norm(pts-w1[:,None,:3,3],axis=2).min(1)
            lifted=np.linalg.norm(T[:,:3,3]-T[0,:3,3],axis=1)>0.02
            held=attached&(d_post<0.12)&lifted
            if held.sum()<10: continue
            rows.append(dict(ep=ep,object=n,side=side,held_frames=int(held.sum()),pre_median_cm=round(float(np.median(d_pre[held]))*100,1),post_median_cm=round(float(np.median(d_post[held]))*100,1),pre_min_cm=round(float(d_pre[held].min())*100,1),moved_median_cm=round(float(np.median(moved[held]))*100,1)))
json.dump(rows,open('/root/probes/hold_audit.json','w'),indent=1)
print('ep object side held pre_med post_med pre_min moved_med')
for r in rows:
    if r.get('skip'): print(r['ep'],'no rawclean/rawexcl bundle'); continue
    flag='  <-- pre>12cm' if r['pre_median_cm']>12 else ''
    print(r['ep'],r['object'],r['side'],r['held_frames'],r['pre_median_cm'],r['post_median_cm'],r['pre_min_cm'],r['moved_median_cm'],flag)
