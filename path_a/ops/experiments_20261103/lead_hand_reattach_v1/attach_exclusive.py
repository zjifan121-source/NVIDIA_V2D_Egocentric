"""Exclusive hand attach: the team loader's attach_hands (tools/track3/run_load_retarget.py) attaches EVERY hand that comes
within 12 cm of a lifted object, object after object, so a hand holding the cup is re-attached to the spoon stirring in
that cup (ep32/ep33: left hand leaves the cup). This variant assigns each hand to ONE object — the object whose surface
is closest to the wrist (median over that object's lifted frames) — and then runs the loader's own attach math only for
that (hand, object) pair. Input: the loader's cleaned, smoothed, UNATTACHED raw-hands bundle (scenes/epXX_rawclean/
bundle_cleaned). Output: bundles/epXX_rawexcl/ (real files) + ATTACH_EXCLUSIVE.json. CPU only, no GT.
Usage: attach_exclusive.py <ep> [...]"""
from pathlib import Path
import sys, json, shutil
import numpy as np
from scipy.spatial.transform import Rotation
import os
ROOT=Path(os.environ.get('V2D_ROOT','/Volumes/E/Code/V2D')); sys.path.insert(0,str(ROOT/'NVIDIA_V2D_Egocentric/tools/track3'))
import run_load_retarget as L
import os
O=Path(os.environ.get('LEAD_HAND_OUT',str(Path(__file__).resolve().parent)))
def attach_exclusive(data,bundle,lift=0.02,reach=0.12,blend=5):
    c2w=data['camera_to_world_transform'].astype(np.float64); Rc,tc=c2w[:,:3,:3],c2w[:,:3,3]; mirror=np.diag([1.0,-1.0,-1.0]); hands={}
    for side in ('left','right'):
        j0=L.mano_rest_wrist(side,data[f'hand_{side}_betas'].reshape(10)); R=Rotation.from_rotvec(data[f'hand_{side}_wrist_orient_in_camera'].astype(np.float64)).as_matrix()
        if side=='left': R=mirror@R@mirror
        H=np.tile(np.eye(4),(len(c2w),1,1)); H[:,:3,:3]=Rc@R; H[:,:3,3]=np.einsum('nij,nj->ni',Rc,data[f'hand_{side}_wrist_trans_in_camera']+j0)+tc; hands[side]=(H,j0)
    slots=L.object_slots(data); objs=[]
    for i,(pose_key,scale_key,mesh_file) in enumerate(slots):
        T=data[pose_key].astype(np.float64); v=L.object_vertices(bundle,data,scale_key,mesh_file); v=v[::max(1,len(v)//500)]
        surface=np.einsum('tij,vj->tvi',T[:,:3,:3],v)+T[:,None,:3,3]; segs=L.lifted_segments(T[:,2,3],lift); objs.append(dict(i=i,T=T,surface=surface,segs=segs))
    # assignment: each hand -> the object whose surface is closest (median over that object's lifted frames); only if within reach somewhere
    assign={}; table={}
    for side,(H,_) in hands.items():
        best=None
        for o in objs:
            idx=np.concatenate([np.arange(a,b) for a,b in o['segs']]) if o['segs'] else np.array([],int)
            if len(idx)==0: continue
            d=np.linalg.norm(o['surface'][idx]-H[idx,None,:3,3],axis=-1).min(axis=1); med=float(np.median(d)); table[f'{side}->object_{o["i"]}']=dict(lifted_frames=int(len(idx)),wrist_to_surface_median_cm=round(med*100,2),min_cm=round(float(d.min())*100,2))
            if d.min()<=reach and (best is None or med<best[1]): best=(o['i'],med)
        if best: assign[side]=best[0]
    info=dict(assignment=assign,candidates=table,attached={}); moved={side:np.zeros(len(c2w),bool) for side in hands}
    for o in objs:
        T=o['T']
        for a,b in o['segs']:
            for side,(H,_) in hands.items():
                if assign.get(side)!=o['i']: continue
                dist=np.linalg.norm(o['surface'][a:b]-H[a:b,None,:3,3],axis=-1).min(axis=1)
                if dist.min()>reach: continue
                k=a+int(np.argmin(dist)); grip=np.linalg.inv(T[k])@H[k]
                for t in range(a,b):
                    w=min(1.0,(t-a+1)/(blend+1),(b-t)/(blend+1)); target=T[t]@grip
                    q=Rotation.from_matrix([H[t,:3,:3],target[:3,:3]]); H[t,:3,:3]=(q[0]*Rotation.from_rotvec(w*(q[0].inv()*q[1]).as_rotvec())).as_matrix(); H[t,:3,3]=(1-w)*H[t,:3,3]+w*target[:3,3]
                moved[side][a:b]=True; info['attached'].setdefault(f'object_{o["i"]}',[]).append([side,int(a),int(b),round(float(dist.min())*100,1)])
    for side,(H,j0) in hands.items():
        if not moved[side].any(): continue
        R=np.einsum('nji,njk->nik',Rc,H[:,:3,:3])
        if side=='left': R=mirror@R@mirror
        cam_t=np.einsum('nji,nj->ni',Rc,H[:,:3,3]-tc)-j0; m=moved[side]
        data[f'hand_{side}_wrist_orient_in_camera'][m]=Rotation.from_matrix(R[m]).as_rotvec(); data[f'hand_{side}_wrist_trans_in_camera'][m]=cam_t[m]
        w2c=data[f'hand_{side}_wrist_to_camera_transform'].astype(np.float64); w2c[m,:3,:3]=Rotation.from_rotvec(data[f'hand_{side}_wrist_orient_in_camera'][m].astype(np.float64)).as_matrix(); w2c[m,:3,3]=data[f'hand_{side}_wrist_trans_in_camera'][m]
        data[f'hand_{side}_wrist_to_camera_transform']=w2c.astype(np.float32); data[f'hand_{side}_wrist_to_world_transform']=(c2w@w2c).astype(np.float32); info[f'hand_{side}_attached_frames']=int(m.sum())
    return info
SRC,DST=('rawclean','rawexcl') if '--ken' not in sys.argv else ('kenclean','kenexcl')
for ep in [int(a) for a in sys.argv[1:] if a!='--ken']:
    src=O/'scenes'/f'ep{ep:02d}_{SRC}'/'bundle_cleaned'; out=O/'bundles'/f'ep{ep:02d}_{DST}'
    if out.exists(): shutil.rmtree(out)
    shutil.copytree(src,out,symlinks=False)  # materialize the loader's mesh symlinks
    data=dict(np.load(out/'result.npz',allow_pickle=False)); info=attach_exclusive(data,out)
    np.savez(out/'result.npz',**data); (out/'ATTACH_EXCLUSIVE.json').write_text(json.dumps(dict(episode=ep,source=str(src),**info),indent=1)+'\n')
    print('EXCL',ep,json.dumps(info['assignment']),json.dumps(info['attached']),flush=True)
