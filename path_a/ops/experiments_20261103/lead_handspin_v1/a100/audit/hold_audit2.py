"""Second pass: for every hand in every rawexcl scene, how far did attach drag it, on all frames, regardless of the
post-attach distance? Dragged frames = wrist moved >5 cm by attach. Pre = distance to the nearest object before attach."""
import json
from pathlib import Path
import numpy as np, trimesh
H=Path('/root/v2d/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261103/lead_hand_reattach_v1')
rows=[]
print('ep side n_frames dragged_frames moved_median_cm moved_p95_cm pre_near_median_cm(dragged) post_near_median_cm(dragged)')
for sc in sorted((H/'scenes').glob('ep*_rawexcl')):
    ep=sc.name[2:4]; b1=H/'bundles'/sc.name; b0=sc.parent/f'ep{ep}_rawclean'/'bundle_cleaned'
    if not (b1/'result.npz').exists() or not (b0/'result.npz').exists(): continue
    d1=dict(np.load(b1/'result.npz',allow_pickle=False)); d0=dict(np.load(b0/'result.npz',allow_pickle=False))
    names=[str(n) for n in d1['object_names']] if 'object_names' in d1 else ['object']
    P=[]
    for i,n in enumerate(names):
        T=d1['object_to_world_transform' if i==0 else f'object_{i}_to_world_transform'].astype(float)
        m=trimesh.load(b1/('mesh.obj' if i==0 else f'mesh_{i}.obj'),force='mesh',process=False)
        s=float(np.asarray(d1.get('object_scale' if i==0 else f'object_{i}_scale',1.0)).reshape(-1)[0]); v=np.asarray(m.vertices)[::max(1,len(m.vertices)//1500)]*s
        P.append(np.einsum('tij,nj->tni',T[:,:3,:3],v)+T[:,None,:3,3])
    for side in ('left','right'):
        w1=d1[f'hand_{side}_wrist_to_world_transform'].astype(float)[:,:3,3]; w0=d0[f'hand_{side}_wrist_to_world_transform'].astype(float)[:,:3,3]
        mv=np.linalg.norm(w1-w0,axis=1); dr=mv>0.05
        dpre=np.min([np.linalg.norm(p-w0[:,None],axis=2).min(1) for p in P],axis=0); dpost=np.min([np.linalg.norm(p-w1[:,None],axis=2).min(1) for p in P],axis=0)
        r=dict(ep=ep,side=side,n=len(mv),dragged=int(dr.sum()),moved_med=round(float(np.median(mv[dr]))*100,1) if dr.any() else 0,moved_p95=round(float(np.percentile(mv,95))*100,1),pre=round(float(np.median(dpre[dr]))*100,1) if dr.any() else None,post=round(float(np.median(dpost[dr]))*100,1) if dr.any() else None)
        rows.append(r); print(ep,side,r['n'],r['dragged'],r['moved_med'],r['moved_p95'],r['pre'],r['post'],'  <-- FAR' if (r['pre'] or 0)>12 and r['dragged']>=30 else '')
json.dump(rows,open('/root/probes/hold_audit2.json','w'),indent=1)
