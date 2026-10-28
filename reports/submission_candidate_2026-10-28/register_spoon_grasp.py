"""Align a successful spoon's body frame to the target using reconstructed surfaces only."""
from pathlib import Path
import json,itertools,numpy as np,trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
P=Path(__file__).resolve().parent;plan=json.loads((P/'plan.json').read_text());meshes=[trimesh.load(Path(plan[str(ep)]['scene'])/f'score_input/meshes/episode_{ep:06d}/wooden_spoon.obj',force='mesh') for ep in [37,33]];np.random.seed(0);X,Y=[m.sample(8000) for m in meshes];cx,cy=X.mean(0),Y.mean(0);Ax=np.linalg.svd(X-cx,full_matrices=False)[2];Ay=np.linalg.svd(Y-cy,full_matrices=False)[2];tree=cKDTree(Y);results={}
for mode in ['rigid','similarity']:
 best=None
 for signs in itertools.product([-1,1],repeat=3):
  R=Ay.T@np.diag(signs)@Ax
  if np.linalg.det(R)<0:continue
  scale=1. if mode=='rigid' else np.ptp((Y-cy)@Ay.T,axis=0)[0]/np.ptp((X-cx)@Ax.T,axis=0)[0];translation=cy-scale*R@cx
  for iteration in range(60):
   Z=scale*X@R.T+translation;distance,idx=tree.query(Z);keep=distance<=np.quantile(distance,.8);a=Z[keep];b=Y[idx[keep]];ca,cb=a.mean(0),b.mean(0);H=(a-ca).T@(b-cb);u,s,vt=np.linalg.svd(H);dR=vt.T@u.T
   if np.linalg.det(dR)<0:vt[-1]*=-1;dR=vt.T@u.T
   ds=1. if mode=='rigid' else np.clip(s.sum()/np.sum((a-ca)**2),.98,1.02)
   if not .8<=scale*ds<=1.2:ds=1.
   dt=cb-ds*dR@ca;translation=ds*dR@translation+dt;R=dR@R;scale*=ds
  Z=scale*X@R.T+translation;d=tree.query(Z)[0];reverse=cKDTree(Z).query(Y)[0];score=float(np.mean(np.r_[np.sort(d)[:6400],np.sort(reverse)[:6400]]**2)**.5)
  row=dict(scale=float(scale),rotation=R.tolist(),translation_m=translation.tolist(),symmetric_trimmed_rmse_m=score,mean_surface_distance_m=float(d.mean()))
  if best is None or score<best['symmetric_trimmed_rmse_m']:best=row
 results[mode]=best;print(mode,json.dumps(best),flush=True)
z=np.load(P/'ep37_successful_grasp_trace.npz');meta=json.loads((P/'ep37_successful_grasp_trace.json').read_text());frame=60;body=z['body_q'][frame];obj=body[int(z['object_body_ids'][0])];wr=next(i for i,n in enumerate(meta['body_labels']) if n.endswith('/right_hand_C_MC'));Robj=Rotation.from_quat(obj[3:]).as_matrix();Rhand=Rotation.from_quat(body[wr,3:]).as_matrix();local_pos=Robj.T@(body[wr,:3]-obj[:3]);local_rot=Robj.T@Rhand;ids=[i for i,n in enumerate(meta['joint_labels']) if n.startswith('right_sharpa_wave/')];names=[meta['joint_labels'][i].split('/')[-1] for i in ids];angles=z['joint_q'][frame,z['joint_q_start'][ids]]
widths=np.diff(z['joint_q_start']);dofs=np.array([3 if w==4 else 6 if w==7 else w for w in widths]);starts=np.r_[0,np.cumsum(dofs)];assert starts[-1]==z['joint_target'].shape[1];command_angles=z['joint_target'][frame,starts[ids]]
for mode,r in results.items():
 R=np.array(r['rotation']);r['wrist_position_target_object_m']=(r['scale']*R@local_pos+np.array(r['translation_m'])).tolist();r['wrist_wxyz_target_object']=Rotation.from_matrix(R@local_rot).as_quat()[[3,0,1,2]].tolist()
out=dict(source_episode=37,target_episode=33,source_frame=frame,source_policy_trace=str(P/'ep37_successful_grasp_trace.npz'),finger_joint_names=names,finger_joint_positions=angles.tolist(),finger_joint_command_positions=command_angles.tolist(),registrations=results);(P/'spoon_grasp_registration.json').write_text(json.dumps(out,indent=2)+'\n')
