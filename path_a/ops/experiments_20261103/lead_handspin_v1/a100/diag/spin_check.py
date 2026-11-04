"""Spin/orbit check: for each held object, (a) tracked axial twist of the object during the hold (symmetric objects only make
sense), (b) the holding hand's azimuth around the object axis in the object frame (raw hands vs. the bundle's hands),
(c) how far attach moved the wrist, (d) max rotation step of every object (flip detector). Usage: spin_check.py <bundle.npz> [<raw_hands.npz>] --attach <ATTACH_EXCLUSIVE.json or 'auto'>"""
import sys, json, numpy as np, trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation as R
b=Path(sys.argv[1]); raw=Path(sys.argv[2]) if len(sys.argv)>2 and not sys.argv[2].startswith('--') else None
z=np.load(b/'result.npz'); zr=np.load(raw/'result.npz') if raw else None
names=[str(x) for x in z['object_names']] if 'object_names' in z.files else None
def axis(mesh):
    v=mesh.vertices-mesh.vertices.mean(0); w,vec=np.linalg.eigh(v.T@v); return vec[:,-1]
def twist(Rrel,a):
    q=R.from_matrix(Rrel).as_quat(); return np.degrees(np.unwrap(np.radians(np.degrees(2*np.arctan2(q[:,:3]@a,q[:,3])))))
slots=[k for k in z.files if k.endswith('_to_world_transform') and k.startswith('object')]
for nk in sorted(slots):
    slot=0 if nk=='object_to_world_transform' else int(nk.split('_')[1]); name=names[slot] if names else nk
    T=z[nk]; Rw=T[:,:3,:3]; t=T[:,:3,3]; n=len(T)
    step=np.degrees(np.array([R.from_matrix(Rw[i].T@Rw[i+1]).magnitude() for i in range(n-1)]))
    mesh=trimesh.load(b/('mesh.obj' if slot==0 else f'mesh_{slot}.obj'),force='mesh'); a=axis(mesh)
    print(f'{name}: rot step p95 {np.percentile(step,95):.1f} max {step.max():.1f} deg; frames>60deg: {int((step>60).sum())}')
    for side in ['left','right']:
        W=z['camera_to_world_transform']@z[f'hand_{side}_wrist_to_camera_transform']; w=W[:,:3,3]
        d=np.linalg.norm(w-t,axis=1); held=d<0.15
        if held.sum()<10: continue
        f=np.where(held)[0]; f0,f1=f[0],f[-1]+1
        Rrel=np.einsum('ij,nik->njk',Rw[f0],Rw[f0:f1]); tw=twist(Rrel,a); tw-=tw[0]
        pc=np.einsum('nji,nj->ni',Rw[f0:f1],(w[f0:f1]-t[f0:f1])); perp=pc-np.outer(pc@a,a); e1=perp[0]/np.linalg.norm(perp[0]); e2=np.cross(a,e1)
        az=np.degrees(np.unwrap(np.arctan2(perp@e2,perp@e1))); az-=az[0]
        line=f'  {side} near f{f0}-{f1} ({held.sum()} frames<15cm): object twist range {tw.max()-tw.min():.0f} deg; bundle-hand azimuth in object frame range {az.max()-az.min():.0f} deg'
        if zr is not None:
            Wr=zr['camera_to_world_transform']@zr[f'hand_{side}_wrist_to_camera_transform']; wr=Wr[:,:3,3]
            pcr=np.einsum('nji,nj->ni',Rw[f0:f1],(wr[f0:f1]-t[f0:f1])); perpr=pcr-np.outer(pcr@a,a)
            azr=np.degrees(np.unwrap(np.arctan2(perpr@e2,perpr@e1))); azr-=azr[0]
            disp=np.linalg.norm(w[f0:f1]-wr[f0:f1],axis=1)*100
            line+=f'; raw-hand azimuth range {azr.max()-azr.min():.0f} deg, corr(raw az,-twist) {np.corrcoef(azr,-tw)[0,1]:.2f}; wrist moved by attach median {np.median(disp):.1f} max {disp.max():.1f} cm'
        print(line)
