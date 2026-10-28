"""Experimental robot-space grasp refinement; retains object and wrist references."""
from pathlib import Path
import sys,json,shutil,hashlib,xml.etree.ElementTree as ET
import numpy as np,pyarrow as pa,pyarrow.parquet as pq,trimesh
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
sys.path.insert(0,'path_b');from pathb import fk
ROOT=Path('reports/submission_candidate_2026-10-28')
S=ROOT/'ep25_grip3_scene';D=ROOT/'ep25_pinch_scene'
if D.exists():raise FileExistsError(D)
p=next(S.glob('assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=sharpa_wave/*.parquet'));t=pq.read_table(p)
c=lambda k:np.asarray(t[k][0].as_py())
side='right';jn=t['right_robot_finger_joint_names'][0].as_py();q=c('robot_right_finger_joints');original=q.copy();wp=c('robot_right_wrist_position');wq=c('robot_right_wrist_wxyz');op=c('object_body_position');oq=c('object_body_wxyz');names=t['object_body_names'][0].as_py()
up=Path(str(fk.FLASH_URDF).format(side=side));xml=ET.parse(up).getroot();hand=fk.sharpa(side,'urdf');frame_hand=fk.sharpa(side)
assert frame_hand.names==t['right_robot_frame_names'][0].as_py()
assert np.allclose(frame_hand.frames(wp,wq,dict(zip(jn,q.T))),c('robot_right_frames'),atol=2e-6)
indices=[i for i,n in enumerate(jn) if '_thumb_' in n or '_index_' in n]
limits=[xml.find(f"joint[@name='{jn[i]}']/limit") for i in indices];lo=np.array([float(x.get('lower')) for x in limits]);hi=np.array([float(x.get('upper')) for x in limits]);audit=[]
for slot,start,end,anchor in [(1,150,333,220)]:
 R=Rotation.from_quat(oq[:,slot][:,[1,2,3,0]]).as_matrix();mesh=trimesh.load(next(z for z in (S/'score_input/meshes/episode_000025').glob('*.obj') if names[slot].endswith(z.stem)),force='mesh',process=False)
 poses=hand.poses(wp[anchor],wq[anchor],dict(zip(jn,q[anchor])));pads=[];targets=[];normals=[];anchors=[]
 for digit in ['thumb','index']:
  name='right_'+digit+'_elastomer';link=xml.find(f"link[@name='{name}']");pad=trimesh.load((up.parent/link.find('collision/geometry/mesh').get('filename')).resolve(),force='mesh');T=poses[name][0];v=(pad.vertices@T[:3,:3].T+T[:3,3]-op[anchor,slot])@R[anchor]
  closest,dist,face=trimesh.proximity.closest_point_naive(mesh,v)
  # For an intersecting pad, use its deepest interior vertex to push it back outside.
  inside=mesh.contains(v);idx=np.where(inside,dist,-1).argmax() if inside.any() else dist.argmin()
  normal=mesh.face_normals[face[idx]];target=closest[idx]+normal*0.0003
  pads.append((name,pad.vertices[idx]));targets.append(target);normals.append(normal);anchors.append(float(dist[idx]))
 if np.dot(*normals)>0:raise ValueError('Selected pads do not oppose the object')
 # Optimize only the anchor. Then preserve that actual robot/object grasp, rather
 # than assuming MANO-level rigid wrists survive the downstream retarget solver.
 target_world=np.asarray(targets)@R[anchor].T+op[anchor,slot]
 def residual(x):
  angles=original[anchor].copy();angles[indices]=x;poses=hand.poses(wp[anchor],wq[anchor],dict(zip(jn,angles)))
  points=np.array([poses[name][0,:3,:3]@v+poses[name][0,:3,3] for name,v in pads])
  return np.r_[(points-target_world).ravel(),0.0005*(x-original[anchor,indices])]
 fit=least_squares(residual,np.clip(original[anchor,indices],lo+1e-8,hi-1e-8),bounds=(lo,hi),max_nfev=150,gtol=1e-10)
 target_q=original[anchor].copy();target_q[indices]=fit.x
 err=np.linalg.norm(residual(fit.x)[:6].reshape(2,3),axis=1).max()
 if err>0.002:raise ValueError('Anchor grasp is not reachable')
 wrist_local=(wp[anchor]-op[anchor,slot])@R[anchor]
 wrist_r_local=R[anchor].T@Rotation.from_quat(wq[anchor,[1,2,3,0]]).as_matrix()
 for f in range(start,end):
  weight=min(1,(f-start+1)/13,(end-f)/13)
  target_wp=R[f]@wrist_local+op[f,slot]
  target_wr=R[f]@wrist_r_local
  old_wr=Rotation.from_quat(wq[f,[1,2,3,0]]).as_matrix()
  delta=Rotation.from_matrix(target_wr@old_wr.T).as_rotvec()
  wp[f]=(1-weight)*wp[f]+weight*target_wp
  wq[f]=Rotation.from_matrix(Rotation.from_rotvec(weight*delta).as_matrix()@old_wr).as_quat()[[3,0,1,2]]
  q[f]=(1-weight)*original[f]+weight*target_q
 audit.append(dict(object=names[slot],hold=[start,end],anchor=anchor,opposing_normal_dot=float(np.dot(*normals)),max_anchor_target_error_m=float(err),targets_local=np.asarray(targets).tolist(),robot_wrist_rigid=True))
 print(audit[-1],flush=True)
frames=frame_hand.frames(wp,wq,dict(zip(jn,q.T)))
for key,value in [('robot_right_finger_joints',q),('robot_right_frames',frames),('robot_right_wrist_position',wp),('robot_right_wrist_wxyz',wq)]:
 idx=t.column_names.index(key);t=t.set_column(idx,t.schema.field(idx),pa.array([value.tolist()],type=t.schema.field(idx).type))
shutil.copytree(S,D);pq.write_table(t,D/p.relative_to(S))
new=pq.read_table(D/p.relative_to(S));old=pq.read_table(p)
for key in old.column_names:
 if key not in ['robot_right_finger_joints','robot_right_frames','robot_right_wrist_position','robot_right_wrist_wxyz']:assert old[key].equals(new[key]),key
(D/'pinch_refinement.json').write_text(json.dumps(dict(source=str(S),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),changed_columns=['robot_right_finger_joints','robot_right_frames','robot_right_wrist_position','robot_right_wrist_wxyz'],holds=audit),indent=2)+'\n')
print('wrote',D,flush=True)
