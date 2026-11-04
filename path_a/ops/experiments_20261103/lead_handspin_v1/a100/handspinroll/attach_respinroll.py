"""Exclusive hand attach + cup respin + spoon roll (lead_v3 rawspinroll mode, 2026-11-04). Spoon rule: the same wrist-azimuth rule about the spoon long axis, frame-0 roll when not held. Rest as rawspin: attach_exclusive's assignment and attach math, plus
HaisenKang's respin rule (hand_fixes_wentao_v1 tools/ep32/sym_common.py, mode "respin" with frozen blends): for axially
symmetric cups (kit: blue_cup, beige_cup, whose axial spin the official AUC ignores) the reference spin, which is
FoundationPose noise, is removed and replaced by the holding wrist's azimuth change about the cup axis; constant outside
holds. Cup position and tilt are unchanged, frame 0 is unchanged, every non-cup object is unchanged, meshes are unchanged.
Input: <H>/scenes/epXX_{rawclean|kenclean}/bundle_cleaned (cleaned, smoothed, unattached). Output: $LEAD_HAND_OUT/bundles/epXX_rawspin.
CPU only, no GT. Usage: attach_respin.py [--ken] <ep> [...]"""
from pathlib import Path
import sys, json, shutil, os
import numpy as np, trimesh
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d
ROOT=Path(os.environ['V2D_ROOT']); sys.path.insert(0,str(ROOT/'NVIDIA_V2D_Egocentric/tools/track3'))
import run_load_retarget as L
H_DIR=ROOT/'NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261103/lead_hand_reattach_v1'; O=Path(os.environ['LEAD_HAND_OUT'])
SYM={'blue_cup','beige_cup'}
ROLL={'wooden_spoon'}  # thin object: the tracked roll about the long axis is spurious, take it from the holding wrist
RULE=os.environ.get('ROLL_RULE','frozen')
RAMP=15  # frames at the end of a hold that ease the roll back to the frame-0 roll
def sym_axis(v,f):
    m=trimesh.Trimesh(v,f,process=False); pts,_=trimesh.sample.sample_surface(m,20000,seed=0); c=pts.mean(0)
    w,U=np.linalg.eigh(np.cov((pts-c).T)); sep=[min(abs(w[j]-w[(j+1)%3]),abs(w[j]-w[(j+2)%3])) for j in range(3)]
    a=U[:,int(np.argmax(sep))]; return a/np.linalg.norm(a),c,[float(x) for x in w]
def twist(Rrel,a):
    q=Rotation.from_matrix(Rrel).as_quat(); th=2.0*np.arctan2(float(np.dot(q[:3],a)),float(q[3])); return (th+np.pi)%(2*np.pi)-np.pi
def spin(a,c,phi):
    S=np.eye(4); S[:3,:3]=Rotation.from_rotvec(a*phi).as_matrix(); S[:3,3]=c-S[:3,:3]@c; return S
def despun(T,a,c,k):
    out=T.copy(); th=np.zeros(len(T))
    for t in range(len(T)):
        th[t]=twist(T[k,:3,:3].T@T[t,:3,:3],a); out[t]=T[t]@spin(a,c,-th[t])
    return out,th
def azimuth(T,P,a,c):
    e1=np.cross(a,[1.0,0.0,0.0])
    if np.linalg.norm(e1)<0.1: e1=np.cross(a,[0.0,1.0,0.0])
    e1/=np.linalg.norm(e1); e2=np.cross(a,e1); x=np.einsum('tji,tj->ti',T[:,:3,:3],P-T[:,:3,3])-c
    return np.unwrap(np.arctan2(x@e2,x@e1))
def hand_roll(T,H,a):
    # roll of the hand about the object axis, from the hand orientation: G_t G_0^T is a rotation about a if the hand keeps its grip up to roll. The wrist position azimuth is ill-conditioned here because the wrist can pass close to a thin object axis.
    G=np.einsum('tji,tjk->tik',T[:,:3,:3],H[:,:3,:3]); rel=np.einsum('tij,kj->tik',G,G[0]); q=Rotation.from_matrix(rel).as_quat()
    th=np.unwrap(2.0*np.arctan2(q[:,:3]@a,q[:,3])); rest=np.array([Rotation.from_rotvec(-a*th[t]).as_matrix()@rel[t] for t in range(len(rel))])
    return th,np.degrees(np.arccos(np.clip((np.trace(rest,axis1=1,axis2=2)-1)/2,-1,1)))
def rot_steps(T):
    R=T[:,:3,:3]; rel=np.einsum('nij,njk->nik',R[:-1].transpose(0,2,1),R[1:]); return np.degrees(np.arccos(np.clip((np.trace(rel,axis1=1,axis2=2)-1)/2,-1,1)))
def attach_respin(data,bundle,names,lift=0.02,reach=0.12,blend=5):
    c2w=data['camera_to_world_transform'].astype(np.float64); Rc,tc=c2w[:,:3,:3],c2w[:,:3,3]; mirror=np.diag([1.0,-1.0,-1.0]); hands={}; n=len(c2w)
    for side in ('left','right'):
        j0=L.mano_rest_wrist(side,data[f'hand_{side}_betas'].reshape(10)); R=Rotation.from_rotvec(data[f'hand_{side}_wrist_orient_in_camera'].astype(np.float64)).as_matrix()
        if side=='left': R=mirror@R@mirror
        H=np.tile(np.eye(4),(n,1,1)); H[:,:3,:3]=Rc@R; H[:,:3,3]=np.einsum('nij,nj->ni',Rc,data[f'hand_{side}_wrist_trans_in_camera']+j0)+tc; hands[side]=(H,j0)
    objs=[]
    for i,(pose_key,scale_key,mesh_file) in enumerate(L.object_slots(data)):
        T=data[pose_key].astype(np.float64); v=L.object_vertices(bundle,data,scale_key,mesh_file); v=v[::max(1,len(v)//500)]
        zlift=T[:,2,3].copy()
        if i<len(names) and names[i] in ROLL: zlift[0]=np.median(zlift[1:11])  # frame 0 is anchored separately and can sit centimetres below the resting track, which would read as lifted from frame 1
        objs.append(dict(i=i,T=T,v=v,surface=np.einsum('tij,vj->tvi',T[:,:3,:3],v)+T[:,None,:3,3],segs=L.lifted_segments(zlift,lift),pose_key=pose_key,scale_key=scale_key,mesh_file=mesh_file))
    assign={}; table={}
    for side,(H,_) in hands.items():
        best=None
        for o in objs:
            idx=np.concatenate([np.arange(a,b) for a,b in o['segs']]) if o['segs'] else np.array([],int)
            if len(idx)==0: continue
            d=np.linalg.norm(o['surface'][idx]-H[idx,None,:3,3],axis=-1).min(axis=1); med=float(np.median(d)); table[f'{side}->object_{o["i"]}']=dict(lifted_frames=int(len(idx)),wrist_to_surface_median_cm=round(med*100,2),min_cm=round(float(d.min())*100,2))
            if d.min()<=reach and (best is None or med<best[1]): best=(o['i'],med)
        if best: assign[side]=best[0]
    info=dict(assignment=assign,candidates=table,attached={},respin={})
    for o in objs:
        i=o['i']
        if i>=len(names) or names[i] not in SYM|ROLL: continue
        mesh=trimesh.load(bundle/o['mesh_file'],force='mesh',process=False); a_ax,c_ax,w=sym_axis(np.asarray(mesh.vertices)*float(data.get(o['scale_key'],1.0)),np.asarray(mesh.faces))
        T=o['T']; before=rot_steps(T); Tds,th0=despun(T,a_ax,c_ax,0); psi=np.zeros(n); last,cur=0,0.0; holds=[]; ends=[]; grip_rest=[]
        for s0,s1 in sorted(o['segs']):
            for side,(H,_) in hands.items():
                if assign.get(side)!=i: continue
                d=np.linalg.norm(o['surface'][s0:s1]-H[s0:s1,None,:3,3],axis=-1).min(axis=1)
                if d.min()<=reach: holds.append((s0,s1,side))
        for s0,s1,side in holds:
            if s0<last: continue  # overlapping hold of the other hand: the first one drives the spin
            psi[last:s0]=cur; az=azimuth(Tds[s0:s1],hands[side][0][s0:s1,:3,3],a_ax,c_ax)
            if names[i] in ROLL:
                azo,rest=hand_roll(Tds[s0:s1],hands[side][0][s0:s1],a_ax); grip_rest.append([int(s0),int(s1),round(float(np.median(rest)),1),round(float(np.percentile(rest,95)),1)])
                if RULE=='orientation': az=azo
                if RULE=='frozen': az=np.zeros(s1-s0)
            if s1-s0>2*blend+2: az=az[np.clip(np.arange(s1-s0),blend,s1-s0-blend-1)]  # spin only while fully held
            psi[s0:s1]=cur+(az-az[0]); cur,last=float(psi[s1-1]),s1
            if names[i] in ROLL:  # roll back to the frame-0 roll (nearest whole turn) over the end of the hold, so it is the frame-0 roll whenever not held
                end=float(psi[s1-1]); turn=2*np.pi*np.round(end/(2*np.pi)); ends.append([int(s1),round(float(np.degrees(end-turn)),1)])
                if s1<n:  # released before the sequence ends; a spoon still held at the last frame keeps the hand roll
                    r=min(RAMP,s1-s0); ww=np.ones(s1-s0); ww[-r:]=np.linspace(1,0,r); psi[s0:s1]=psi[s0:s1]*ww+turn*(1-ww); cur=turn
        psi[last:]=cur; psi=gaussian_filter1d(psi,2.0,mode='nearest')
        psi-=psi[0]  # frame 0 stays exactly the v1 pose
        Tn=np.stack([Tds[t]@spin(a_ax,c_ax,psi[t]) for t in range(n)])
        t_ok=np.linalg.norm((T[:,:3,:3]@c_ax+T[:,:3,3])-(Tn[:,:3,:3]@c_ax+Tn[:,:3,3]),axis=1).max(); ax_w=T[:,:3,:3]@a_ax; ax_n=Tn[:,:3,:3]@a_ax
        info['respin'][names[i]]=dict(rule='roll' if names[i] in ROLL else 'spin',roll_error_at_release_deg=ends,grip_residual_deg_median_p95=grip_rest,tracked_twist_range_deg=round(float(np.degrees(np.ptp(np.unwrap(th0)))),1),axis=[round(float(x),4) for x in a_ax],center_cm=[round(float(x)*100,2) for x in c_ax],cov_eigs=[round(x*1e4,2) for x in w],holds=[[int(a),int(b),s] for a,b,s in holds],
            psi_range_deg=[round(float(np.degrees(psi.min())),1),round(float(np.degrees(psi.max())),1)],rot_path_deg_before=round(float(before.sum()),0),rot_path_deg_after=round(float(rot_steps(Tn).sum()),0),
            rot_p95_before=round(float(np.percentile(before,95)),2),rot_p95_after=round(float(np.percentile(rot_steps(Tn),95)),2),
            axis_point_moved_max_mm=round(float(t_ok)*1000,3),axis_direction_changed_max_deg=round(float(np.degrees(np.arccos(np.clip(np.einsum('ti,ti->t',ax_w,ax_n),-1,1)).max())),4),frame0_max_abs=float(np.abs(Tn[0]-T[0]).max()))
        assert t_ok<1e-5 and np.abs(Tn[0]-T[0]).max()<1e-9,(names[i],t_ok)
        data[o['pose_key']]=Tn.astype(data[o['pose_key']].dtype); cam=o['pose_key'].replace('to_world','to_camera')
        if cam in data: data[cam]=(np.linalg.inv(c2w)@Tn).astype(data[cam].dtype)
        o['T']=Tn; o['surface']=np.einsum('tij,vj->tvi',Tn[:,:3,:3],o['v'])+Tn[:,None,:3,3]
    moved={side:np.zeros(n,bool) for side in hands}
    for o in objs:
        T=o['T']; is_cup=o['i']<len(names) and names[o['i']] in SYM|ROLL
        for a,b in o['segs']:
            for side,(H,_) in hands.items():
                if assign.get(side)!=o['i']: continue
                dist=np.linalg.norm(o['surface'][a:b]-H[a:b,None,:3,3],axis=-1).min(axis=1)
                if dist.min()>reach: continue
                k=a+int(np.argmin(dist)); grip=np.linalg.inv(T[k])@H[k]; prev=None
                for t in range(a,b):
                    w=min(1.0,(t-a+1)/(blend+1),(b-t)/(blend+1)); target=T[t]@grip; q0=Rotation.from_matrix(H[t,:3,:3]); rv=(q0.inv()*Rotation.from_matrix(target[:3,:3])).as_rotvec(); ang=np.linalg.norm(rv)
                    if is_cup and prev is not None and ang>1e-9:  # continuous branch, as in the respin source
                        alt=rv-2*np.pi*rv/ang
                        if np.linalg.norm(alt-prev)<np.linalg.norm(rv-prev): rv=alt
                    prev=rv; H[t,:3,:3]=(q0*Rotation.from_rotvec(w*rv)).as_matrix(); H[t,:3,3]=(1-w)*H[t,:3,3]+w*target[:3,3]
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
SRC='kenclean' if '--ken' in sys.argv else 'rawclean'
CTL='--ctl' in sys.argv  # no raw hands for this episode: re-seat the delivered (control) hands on the current tracks
INV={int(r['episode']):r['objects'] for r in json.loads((H_DIR/'inventory_v1.json').read_text())}
for ep in [int(a) for a in sys.argv[1:] if not a.startswith('--')]:
    src=(H_DIR/'bundles'/f'ep{ep:02d}_control') if CTL else H_DIR/'scenes'/f'ep{ep:02d}_{SRC}'/'bundle_cleaned'; out=O/'bundles'/f'ep{ep:02d}_rawspinroll'
    if out.exists(): shutil.rmtree(out)
    shutil.copytree(src,out,symlinks=False)
    data=dict(np.load(out/'result.npz',allow_pickle=False)); names=[str(x) for x in data['object_names']] if 'object_names' in data else INV[ep]
    info=attach_respin(data,out,names); np.savez(out/'result.npz',**data); (out/'ATTACH_RESPIN.json').write_text(json.dumps(dict(episode=ep,source=str(src),**info),indent=1)+'\n')
    print('RESPIN',ep,json.dumps(info['assignment']),json.dumps(info['respin']),flush=True)
