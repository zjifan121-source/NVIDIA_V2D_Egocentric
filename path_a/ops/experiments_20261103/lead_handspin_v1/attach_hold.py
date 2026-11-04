"""Per-frame hold attach + cup respin (lead_v3 hold mode, 2026-11-04). attach_exclusive attached a hand for a whole lifted segment
if it passed within reach once, which forced far hands onto objects (ep28 left: median 40.7 cm before attach). This version uses
the no-mask fallback of Edmond's de66fea attach_hands: a hand holds an object only in frames where its wrist is within reach of
the lifted object's surface, one object per hand at a time (hold_owner, holds copied from de66fea), and lifted frames that still
have no holder go to the last holder, because a lifted object must be held and those are hand-tracking misses, except a final gap
where the object stays still, which is a release (object set down above its start height). Reports how far each hold
moved the hand, which unlike post-attach contact is not circular. Plus
HaisenKang's respin rule (hand_fixes_wentao_v1 tools/ep32/sym_common.py, mode "respin" with frozen blends): for axially
symmetric cups (kit: blue_cup, beige_cup, whose axial spin the official AUC ignores) the reference spin, which is
FoundationPose noise, is removed and replaced by the holding wrist's azimuth change about the cup axis; constant outside
holds. Cup position and tilt are unchanged, frame 0 is unchanged, every non-cup object is unchanged, meshes are unchanged.
Input: <H>/scenes/epXX_{rawclean|kenclean}/bundle_cleaned (cleaned, smoothed, unattached). Output: $LEAD_HAND_OUT/bundles/epXX_hold.
CPU only, no GT. Usage: attach_hold.py [--ken|--ctl] <ep> [...]"""
from pathlib import Path
import sys, json, shutil, os
import numpy as np, trimesh
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d
ROOT=Path(os.environ['V2D_ROOT']); sys.path.insert(0,str(ROOT/'NVIDIA_V2D_Egocentric/tools/track3'))
import run_load_retarget as L
H_DIR=ROOT/'NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261103/lead_hand_reattach_v1'; O=Path(os.environ['LEAD_HAND_OUT'])
SYM={'blue_cup','beige_cup'}
def hold_owner(D,switch,gap,min_len,ratio=1.0):
    owner=np.full(D.shape[1],-1); cur=-1
    for t in range(D.shape[1]):
        best=int(np.argmin(D[:,t]))
        if not np.isfinite(D[best,t]): cur=-1
        elif cur<0 or not np.isfinite(D[cur,t]) or (D[best,t]<D[cur,t]-switch and D[best,t]*ratio<D[cur,t]): cur=best
        owner[t]=cur
    for o,a,b in holds(owner):
        nxt=np.flatnonzero(owner[b:b+gap+1]>=0)
        if len(nxt) and owner[b+nxt[0]]==o: owner[b:b+nxt[0]]=o
    for o,a,b in holds(owner):
        if b-a<min_len: owner[a:b]=-1
    return owner
def holds(owner):
    edges=np.flatnonzero(np.diff(np.concatenate([[-1],owner,[-1]])))
    return [(int(owner[a]),int(a),int(b)) for a,b in zip(edges[:-1],edges[1:]) if owner[a]>=0]
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
        objs.append(dict(i=i,T=T,v=v,surface=np.einsum('tij,vj->tvi',T[:,:3,:3],v)+T[:,None,:3,3],segs=L.lifted_segments(T[:,2,3],lift),pose_key=pose_key,scale_key=scale_key,mesh_file=mesh_file))
    dist={}; table={}; owners={}
    for o in objs:
        lifted=np.zeros(n,bool)
        for s0,s1 in o['segs']: lifted[s0:s1]=True
        if 'LEAD_LIFT_END' in os.environ: lifted[int(os.environ['LEAD_LIFT_END']):]=False  # a set-down object can stay above its start height
        o['lifted']=lifted
    def detect():
        for side,(H,_) in hands.items():
            for o in objs:
                dist[(side,o['i'])]=np.linalg.norm(o['surface']-H[:,None,:3,3],axis=-1).min(axis=1)
            D=np.stack([np.where(o['lifted']&(dist[(side,o['i'])]<=reach),dist[(side,o['i'])],np.inf) for o in objs])
            owners[side]=hold_owner(D,switch=0.02,gap=10,min_len=2*blend+1)
        # a lifted object must be held: lifted frames with no holder go to the hand that just held it (else the one about to
        # hold it, else the nearest), only if that hand is free for the whole gap and comes within reach during the lift
        gaps_filled.clear()
        for o in objs:
            if not o['lifted'].any(): continue
            i=o['i']; held=np.zeros(n,bool)
            for side in hands: held|=owners[side]==i
            for _,a,b in holds((o['lifted']&~held).astype(int)-1):
                T_=o['T']; net=float(np.linalg.norm(T_[b-1,:3,3]-T_[a,:3,3]))
                later=any((owners[side][b:]==i).any() for side in hands)
                if net<0.01 and not later:  # a set-down object can stay above its start height: that is a release, not a tracking miss
                    gaps_filled.append(dict(object=i,frames=[int(a),int(b)],side=None,reason='release: object static and no later hold',net_cm=round(net*100,2))); continue
                cands=[side for side in hands if (owners[side][a:b]<0).all() and dist[(side,i)][o['lifted']].min()<=reach]
                if not cands: gaps_filled.append(dict(object=i,frames=[a,b],side=None)); continue
                prev=[side for side in cands if a>0 and owners[side][a-1]==i]; nxt=[side for side in cands if b<n and owners[side][b]==i]
                pool=prev or nxt or cands; side=min(pool,key=lambda x:float(np.median(dist[(x,i)][a:b])))
                owners[side][a:b]=i; gaps_filled.append(dict(object=i,frames=[int(a),int(b)],side=side,median_cm=round(float(np.median(dist[(side,i)][a:b]))*100,1)))
    gaps_filled=[]
    detect()
    for side in hands:
        for o in objs:
            if o['lifted'].any():
                d=dist[(side,o['i'])][o['lifted']]; table[f'{side}->object_{o["i"]}']=dict(lifted_frames=int(o['lifted'].sum()),wrist_to_surface_median_cm=round(float(np.median(d))*100,2),min_cm=round(float(d.min())*100,2),frames_within_reach=int((d<=reach).sum()))
    info=dict(rule='per-frame hold within reach (de66fea fallback) + lifted frames with no holder go to the last holder',reach_m=reach,candidates=table,gaps_filled=gaps_filled,holds={side:[[o,a,b] for o,a,b in holds(owners[side])] for side in hands},attached={},respin={},hand_moved_cm={})
    for o in objs:
        i=o['i']
        if i>=len(names) or names[i] not in SYM: continue
        mesh=trimesh.load(bundle/o['mesh_file'],force='mesh',process=False); a_ax,c_ax,w=sym_axis(np.asarray(mesh.vertices)*float(data.get(o['scale_key'],1.0)),np.asarray(mesh.faces))
        T=o['T']; before=rot_steps(T); Tds,th0=despun(T,a_ax,c_ax,0); psi=np.zeros(n); last,cur=0,0.0
        cup_holds=sorted((s0,s1,side) for side in hands for q,s0,s1 in holds(owners[side]) if q==i)
        for s0,s1,side in cup_holds:
            if s0<last: continue  # overlapping hold of the other hand: the first one drives the spin
            psi[last:s0]=cur; az=azimuth(Tds[s0:s1],hands[side][0][s0:s1,:3,3],a_ax,c_ax)
            if s1-s0>2*blend+2: az=az[np.clip(np.arange(s1-s0),blend,s1-s0-blend-1)]  # spin only while fully held
            psi[s0:s1]=cur+(az-az[0]); cur,last=float(psi[s1-1]),s1
        psi[last:]=cur; psi=gaussian_filter1d(psi,2.0,mode='nearest')
        psi-=psi[0]  # frame 0 stays exactly the v1 pose
        Tn=np.stack([Tds[t]@spin(a_ax,c_ax,psi[t]) for t in range(n)])
        t_ok=np.linalg.norm((T[:,:3,:3]@c_ax+T[:,:3,3])-(Tn[:,:3,:3]@c_ax+Tn[:,:3,3]),axis=1).max(); ax_w=T[:,:3,:3]@a_ax; ax_n=Tn[:,:3,:3]@a_ax
        info['respin'][names[i]]=dict(axis=[round(float(x),4) for x in a_ax],center_cm=[round(float(x)*100,2) for x in c_ax],cov_eigs=[round(x*1e4,2) for x in w],holds=[[int(a),int(b),s] for a,b,s in cup_holds],
            psi_range_deg=[round(float(np.degrees(psi.min())),1),round(float(np.degrees(psi.max())),1)],rot_path_deg_before=round(float(before.sum()),0),rot_path_deg_after=round(float(rot_steps(Tn).sum()),0),
            rot_p95_before=round(float(np.percentile(before,95)),2),rot_p95_after=round(float(np.percentile(rot_steps(Tn),95)),2),
            axis_point_moved_max_mm=round(float(t_ok)*1000,3),axis_direction_changed_max_deg=round(float(np.degrees(np.arccos(np.clip(np.einsum('ti,ti->t',ax_w,ax_n),-1,1)).max())),4),frame0_max_abs=float(np.abs(Tn[0]-T[0]).max()))
        assert t_ok<1e-5 and np.abs(Tn[0]-T[0]).max()<1e-9,(names[i],t_ok)
        data[o['pose_key']]=Tn.astype(data[o['pose_key']].dtype); cam=o['pose_key'].replace('to_world','to_camera')
        if cam in data: data[cam]=(np.linalg.inv(c2w)@Tn).astype(data[cam].dtype)
        o['T']=Tn; o['surface']=np.einsum('tij,vj->tvi',Tn[:,:3,:3],o['v'])+Tn[:,None,:3,3]
    moved={side:np.zeros(n,bool) for side in hands}
    detect()  # cup surfaces changed only by spin; hold windows stay comparable
    info['holds_final']={side:[[o,a,b] for o,a,b in holds(owners[side])] for side in hands}
    for side,(H,_) in hands.items():
        H0=H.copy()
        for q,a,b in holds(owners[side]):
            o=objs[q]; T=o['T']; is_cup=q<len(names) and names[q] in SYM
            d=dist[(side,q)]; k=a+int(np.argmin(d[a:b])); grip=np.linalg.inv(T[k])@H0[k]; prev=None
            for t in range(a,b):
                w=min(1.0,(t-a+1)/(blend+1),(b-t)/(blend+1)); target=T[t]@grip; q0=Rotation.from_matrix(H0[t,:3,:3]); rv=(q0.inv()*Rotation.from_matrix(target[:3,:3])).as_rotvec(); ang=np.linalg.norm(rv)
                if is_cup and prev is not None and ang>1e-9:
                    alt=rv-2*np.pi*rv/ang
                    if np.linalg.norm(alt-prev)<np.linalg.norm(rv-prev): rv=alt
                prev=rv; H[t,:3,:3]=(q0*Rotation.from_rotvec(w*rv)).as_matrix(); H[t,:3,3]=(1-w)*H0[t,:3,3]+w*target[:3,3]
            moved[side][a:b]=True; shift=np.linalg.norm(H[a:b,:3,3]-H0[a:b,:3,3],axis=1)*100
            info['attached'].setdefault(f'object_{q}',[]).append([side,int(a),int(b),round(float(d[k])*100,1)]); info['hand_moved_cm'].setdefault(side,[]).append(dict(object=q,frames=[a,b],median=round(float(np.median(shift)),2),max=round(float(shift.max()),2)))
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
    src=Path(os.environ['LEAD_HOLD_SRC']) if 'LEAD_HOLD_SRC' in os.environ else ((H_DIR/'bundles'/f'ep{ep:02d}_control') if CTL else H_DIR/'scenes'/f'ep{ep:02d}_{SRC}'/'bundle_cleaned'); out=O/'bundles'/f'ep{ep:02d}_hold'
    if out.exists(): shutil.rmtree(out)
    shutil.copytree(src,out,symlinks=False)
    data=dict(np.load(out/'result.npz',allow_pickle=False)); names=[str(x) for x in data['object_names']] if 'object_names' in data else INV[ep]
    info=attach_respin(data,out,names); np.savez(out/'result.npz',**data); (out/'ATTACH_HOLD.json').write_text(json.dumps(dict(episode=ep,source=str(src),**info),indent=1)+'\n')
    print('HOLD',ep,json.dumps(info['holds_final']),'GAPS',json.dumps(info['gaps_filled']),json.dumps(info['hand_moved_cm']),json.dumps({k:{x:v[x] for x in ('rot_path_deg_before','rot_path_deg_after')} for k,v in info['respin'].items()}),flush=True)
